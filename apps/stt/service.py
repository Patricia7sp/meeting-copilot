"""STT service realtime: captura por fonte -> segmentação contínua -> WS persistente com ack.

Etapas (fidelidade antes de insights):
1. Gate por fonte (noise floor calibrado + histerese + VAD): decide fala/silêncio
   com resolução fina (~250ms) — `YOU` só do mic, `OTHERS` só do loopback.
2. PROVISÓRIO de baixa latência: janelas sobrepostas (~1.6s) sobre o enunciado,
   deduplicadas por texto+tempo. Em nenhum momento finaliza por chunk.
3. FINAL consolidado: só silêncio real (500-800ms) encerra o enunciado; o PCM
   inteiro é re-transcrito de uma vez (=1 `final` por enunciado, sem lacunas).

Transporte: WsClient persistente (fila com backpressure, IDs idempotentes por
event_id, retries com backoff, remoção SÓ após ack, disco p/ queda do WS).
API responde {"type":"ack","id":...} e só gera insights em eventos "final"
consolidados (janela consolidada + limiar de confiança + dedup).

Sobrecarga: locutor por fonte (mic->YOU, loopback->OTHERS), calibração de ruído
por fonte — nunca por `RMS > limiar` fixo (ruído ambiente do mic não vira fala).

Uso:
  MOCK_MODE=false python3 apps/stt/service.py           # realtime
  python3 apps/stt/service.py --mock-once "texto"       # teste sem áudio
  MAX_ITERATIONS=300 python3 apps/stt/service.py        # smoke test (para sozinho)
"""
from __future__ import annotations
import asyncio
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../audio"))
from capture import (Mixer, NoiseFloorCalibrator, SourceAudio, detect_setup,  # noqa: E402
                     rms_db, SAMPLE_RATE)
from wsclient import WsClient  # noqa: E402
from stream import GateResult, Segmenter  # noqa: E402

API_WS_URL = os.getenv("API_WS_URL", "ws://localhost:8000/ws")
MODEL = os.getenv("WHISPER_MODEL", "small")
MOCK = os.getenv("MOCK_MODE", "true").lower() == "true"
LOOPBACK_DEVICE = os.getenv("LOOPBACK_DEVICE")
MIC_DEVICE = os.getenv("MIC_DEVICE")

# --- parâmetros (captura continua + segmentação + duas etapas) --------------
FRAME_SECONDS = float(os.getenv("FRAME_SECONDS", "0.25"))       # resolução do silêncio
CHUNK_SECONDS = float(os.getenv("CHUNK_SECONDS", "1.5"))        # compat (chunk base legado)
OVERLAP_SECONDS = float(os.getenv("OVERLAP_SECONDS", "0.5"))    # sobreposição p/ fronteira
WINDOW_SECONDS = float(os.getenv("WINDOW_SECONDS", "1.6"))      # janela do provisório
PROVISIONAL_EVERY_SECONDS = float(os.getenv("PROVISIONAL_EVERY_SECONDS", "0.8"))
PAUSE_SECONDS = float(os.getenv("PAUSE_SECONDS", "0.7"))        # silêncio real que encerra a fala
MAX_UTTERANCE_SECONDS = float(os.getenv("MAX_UTTERANCE_SECONDS", "10.0"))  # refresh do provisório
MAX_UTTERANCE_CAP_SECONDS = float(os.getenv("MAX_UTTERANCE_CAP_SECONDS", "120.0"))
INITIAL_FLOOR_DB = float(os.getenv("INITIAL_FLOOR_DB", "-46"))
SPEECH_DELTA_DB_MIC = float(os.getenv("SPEECH_DELTA_DB_MIC", "6"))
SPEECH_DELTA_DB_LOOPBACK = float(os.getenv("SPEECH_DELTA_DB_LOOPBACK", "6"))
SILENCE_DELTA_DB = float(os.getenv("SILENCE_DELTA_DB", "3"))
GATE_WARMUP_FRAMES = int(os.getenv("GATE_WARMUP_FRAMES", "4"))
NO_SPEECH_THRESHOLD = float(os.getenv("NO_SPEECH_THRESHOLD", "0.6"))
LOG_PROB_THRESHOLD = float(os.getenv("LOG_PROB_THRESHOLD", "-0.8"))   # provisório (estrito)
LOG_PROB_THRESHOLD_FINAL = float(os.getenv("LOG_PROB_THRESHOLD_FINAL", "-1.0"))  # revisão (fala real)
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.25"))
MIN_WEBRTC_SPEECH_RATIO = float(os.getenv("MIN_SPEECH_RATIO", "0.10"))
VAD_MODE = os.getenv("VAD_MODE", "auto").lower()
TRANSLATE_TARGET = (os.getenv("TRANSLATE_TARGET", "").strip().lower() or None)
WS_QUEUE = int(os.getenv("WS_QUEUE", "200"))
WS_ACK_TIMEOUT = float(os.getenv("WS_ACK_TIMEOUT", "8.0"))
WS_METRICS_EVERY = float(os.getenv("WS_METRICS_EVERY", "10.0"))
WS_PENDING_DIR = os.getenv("WS_PENDING_DIR", "data/ws_pending")

FILLER_ONLY = {"um", "uh", "uhm", "ah", "mm", "hmm", "é", "huh", "the", "a", "yeah", "ok"}

_LEVEL = "[stt]"


# ---------- VAD por fonte (locutor por energia, nunca por device) -----------

def _webrtc_speech_ratio(pcm: bytes, rate: int = SAMPLE_RATE) -> float | None:
    try:
        import webrtcvad  # type: ignore
    except Exception:
        return None
    try:
        vad = webrtcvad.Vad(2)
        frame_bytes = 2 * int(rate * 0.030)
        n_frames = len(pcm) // frame_bytes
        if n_frames < 3:
            return None
        frames = [pcm[i * frame_bytes:(i + 1) * frame_bytes] for i in range(n_frames)]
        speech = sum(1 for f in frames if vad.is_speech(f, rate))
        return speech / n_frames
    except Exception:
        return None


class NoiseFloorGate:
    """Gate por fonte: noise floor calibrado + delta acima do piso + histerese + VAD.

    Nunca decide por `RMS > limiar` fixo: o piso de ruído é aprendido por fonte,
    e a fala só entra com delta acima do piso e histerese na saída (evita que o
    ruído ambiente do mic vire `YOU` ou que a fonte fique chaveando a cada frame).
    """

    def __init__(self, source: str, *, initial_floor_db: float = INITIAL_FLOOR_DB,
                 speech_delta_db: float | None = None, silence_delta_db: float | None = None,
                 min_webrtc_ratio: float = MIN_WEBRTC_SPEECH_RATIO,
                 warmup_frames: int = GATE_WARMUP_FRAMES, log: object = None):
        self.source = source
        self.cal = NoiseFloorCalibrator(initial_floor_db=initial_floor_db)
        self.speech = False
        self.speech_delta_db = (speech_delta_db if speech_delta_db is not None
                                else (SPEECH_DELTA_DB_MIC if source == "mic" else SPEECH_DELTA_DB_LOOPBACK))
        self.silence_delta_db = silence_delta_db if silence_delta_db is not None else SILENCE_DELTA_DB
        self.min_webrtc_ratio = min_webrtc_ratio
        self.warmup = 0
        self.warmup_frames = max(warmup_frames, 1)
        self._log = log or (lambda *a, **k: None)

    def evaluate(self, src: SourceAudio) -> GateResult:
        rms = rms_db(src.rms)
        floor = self.cal.floor_db
        if src.mock:
            return GateResult(False, "mock_backend", rms, floor, None, src.active)
        if not src.active:
            return GateResult(False, "device_inactive", rms, floor, None, False)
        ratio = _webrtc_speech_ratio(src.pcm) if VAD_MODE in ("webrtc", "auto") else None
        if self.speech:
            if rms < floor + self.silence_delta_db:
                self.speech = False
                self.cal.observe(src.rms, is_speech=False)
                return GateResult(False, f"silence_floor (rms {rms:.0f}dB < floor+{self.silence_delta_db:.0f}dB)",
                                  rms, self.cal.floor_db, ratio, src.active)
            return GateResult(True, "speech_active", rms, floor, ratio, src.active)
        self.warmup += 1
        self.cal.observe(src.rms, is_speech=False)
        floor = self.cal.floor_db
        if rms < floor + self.speech_delta_db:
            return GateResult(False, f"below_floor (delta {rms - floor:.0f}dB)",
                              rms, floor, ratio, src.active)
        if ratio is not None and ratio < self.min_webrtc_ratio:
            return GateResult(False, f"no_speech_webrtc (ratio {ratio:.2f})",
                              rms, floor, ratio, src.active)
        if self.warmup < self.warmup_frames:
            return GateResult(False, f"gate_warmup ({self.warmup}/{self.warmup_frames})",
                              rms, floor, ratio, src.active)
        self.speech = True
        return GateResult(True, "speech_enter", rms, floor, ratio, src.active)


def vad_gate(src: SourceAudio) -> tuple[bool, str]:
    """Compat legado: gate único sem estado (testes antigos). Use NoiseFloorGate no realtime."""
    g = NoiseFloorGate(src.source, warmup_frames=1)
    r = g.evaluate(src)
    return r.speech, r.reason


# ---------- métricas dos segmentos ----------

def _seg_metrics(segs) -> dict:
    total = sum(max(s.end - s.start, 1e-6) for s in segs) or 1e-6
    wavg = sum((s.end - s.start) * float(s.avg_logprob) for s in segs) / total
    wns = sum((s.end - s.start) * float(s.no_speech_prob) for s in segs) / total
    conf = min(max(math.exp(min(wavg, 0.0)), 0.0), 1.0)
    return {"avg_logprob": wavg, "no_speech_prob": wns, "confidence": conf,
            "low": (wns > 0.4 or wavg < -1.2 or conf < MIN_CONFIDENCE)}


def _repetitive(text: str) -> bool:
    tt = re.sub(r"[^\w' é]+", " ", text.lower()).strip()
    words = tt.split()
    if len(set(words)) == 1 and len(words) >= 3:
        return True
    return bool(re.fullmatch(r"(\w[\w'-]*)\s+\1(?:\s+\1)+", tt))


def _post_filter(text: str) -> str | None:
    t = text.strip()
    if len(t) < 2:
        return None
    low = t.lower().strip(" .,!?")
    if low in FILLER_ONLY:
        return None
    if _repetitive(t):
        return None
    return t


# ---------- transcrição (multilíngue, mantém idioma falado) ----------------

def transcribe_audio(audio, model, *, final: bool = False) -> tuple[str | None, dict]:
    """Transcreve PCM (bytes) ou float32. final=True => revisão da fala inteira.

    Retorna (texto, meta) ou (None, {"discard": motivo}).
    meta: {lang, language_probability, confidence, low, translation?}
    """
    if model is None:
        return None, {"discard": "model_unavailable"}
    import numpy as np  # type: ignore
    if isinstance(audio, (bytes, bytearray, memoryview)):
        x = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        x = np.asarray(audio, dtype=np.float32)
    kw = dict(language=None, task="transcribe", vad_filter=True,
              vad_parameters={"min_silence_duration_ms": 400, "speech_pad_ms": 300},
              no_speech_threshold=NO_SPEECH_THRESHOLD,
              log_prob_threshold=LOG_PROB_THRESHOLD_FINAL if final else LOG_PROB_THRESHOLD,
              condition_on_previous_text=final,       # provisório: off (anti-alucinação)
              )
    try:
        segs, info = model.transcribe(x, **kw)
    except Exception as e:
        return None, {"discard": f"transcribe_error: {e}"}
    segs = list(segs)
    lang = getattr(info, "language", "unknown") or "unknown"
    lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    text = _post_filter(" ".join(s.text.strip() for s in segs).strip())
    if not text:
        return None, {"discard": f"no_speech (lang={lang})"}
    m = _seg_metrics(segs)
    if lang_prob < 0.6:
        m["low"] = True
    m.update(lang=lang, language_probability=round(lang_prob, 3))
    # Tradução é recurso separado e só vem na revisão final.
    if final and TRANSLATE_TARGET:
        if TRANSLATE_TARGET == "en":
            try:
                tsegs, _translation_info = model.transcribe(x, language=lang, task="translate",
                                         vad_filter=True,
                                         no_speech_threshold=NO_SPEECH_THRESHOLD,
                                         log_prob_threshold=LOG_PROB_THRESHOLD_FINAL,
                                         condition_on_previous_text=True)
                tr = " ".join(s.text.strip() for s in tsegs).strip()
                m["translation"] = tr if tr else None
            except Exception as e:
                m["translation"] = None
                m["translation_error"] = str(e)
        else:
            m["translation"] = None
            m["translation_error"] = f"destino '{TRANSLATE_TARGET}' não nativo; use TRANSLATE_TARGET=en"
    return text, m


def transcribe_source(src: SourceAudio, model) -> tuple[str | None, dict]:
    """Compat: transcreve um chunk simples como provisório."""
    return transcribe_audio(src.pcm, model, final=False)


# ---------- infra ----------

def _speaker_of(source: str) -> str:
    return "YOU" if source == "mic" else "OTHERS"


def _log_rms(sources: dict[str, SourceAudio]) -> None:
    mic, lb = sources["mic"], sources["loopback"]
    print(f"{_LEVEL} rms mic={rms_db(mic.rms):.1f}dB up={mic.active} "
          f"loopback={rms_db(lb.rms):.1f}dB up={lb.active}")


def load_model():
    try:
        from faster_whisper import WhisperModel  # type: ignore
        m = WhisperModel(MODEL, device="cpu", compute_type="int8")
        print(f"{_LEVEL} modelo={MODEL} (cpu int8). frame={int(FRAME_SECONDS * 1000)}ms "
              f"overlap={OVERLAP_SECONDS}s window={WINDOW_SECONDS}s "
              f"pause={PAUSE_SECONDS}s max_utterance={MAX_UTTERANCE_SECONDS}s "
              f"floor={INITIAL_FLOOR_DB}dB+{SPEECH_DELTA_DB_MIC}/{SPEECH_DELTA_DB_LOOPBACK}dB "
              f"log_prob_prev={LOG_PROB_THRESHOLD} final={LOG_PROB_THRESHOLD_FINAL} "
              f"no_speech={NO_SPEECH_THRESHOLD} cond_prev(final)=True "
              f"translate={TRANSLATE_TARGET or '-'} ws_queue={WS_QUEUE}")
        return m
    except ImportError:
        print(f"{_LEVEL} faster-whisper NÃO instalado — modo escuta (frames descartados). "
              "Instale: pip install -r apps/stt/requirements.txt")
        return None


async def run_realtime(max_iter: int | None = None) -> None:
    setup = detect_setup()
    print(f"{_LEVEL} OS={setup.os_name} loopback={setup.loopback_kind} ok={setup.loopback_available}")
    print(f"{_LEVEL} {setup.hint}")
    model = load_model()
    mixer = Mixer(loopback_device=LOOPBACK_DEVICE, mic_device=MIC_DEVICE)
    ws = WsClient(API_WS_URL, max_queue=WS_QUEUE, ack_timeout=WS_ACK_TIMEOUT,
                  pending_dir=WS_PENDING_DIR)
    await ws.start()
    gates = {name: NoiseFloorGate(name) for name in ("mic", "loopback")}

    def gate(src: SourceAudio) -> GateResult:
        return gates[src.source].evaluate(src)

    seg = Segmenter(
        gate=gate,
        provisional=lambda a: transcribe_audio(a, model, final=False),
        final=lambda a: transcribe_audio(a, model, final=True),
        pause_seconds=PAUSE_SECONDS,
        overlap_seconds=OVERLAP_SECONDS,
        max_utterance_seconds=MAX_UTTERANCE_SECONDS,
        max_utterance_cap_seconds=MAX_UTTERANCE_CAP_SECONDS,
        provisional_every_seconds=PROVISIONAL_EVERY_SECONDS,
        window_seconds=WINDOW_SECONDS,
        frame_seconds=FRAME_SECONDS,
        log=print,
    )
    print(f"{_LEVEL} capturando mic={MIC_DEVICE or 'default'} loopback={LOOPBACK_DEVICE or 'default'} "
          f"(frame {int(FRAME_SECONDS * 1000)}ms / overlap {OVERLAP_SECONDS}s / "
          f"pause {PAUSE_SECONDS}s / floor {INITIAL_FLOOR_DB}dB "
          f"+{SPEECH_DELTA_DB_MIC}/{SPEECH_DELTA_DB_LOOPBACK}dB sinais histerese "
          f"{SILENCE_DELTA_DB}dB)... Ctrl+C p/ parar.")
    n = 0
    last_metrics = time.monotonic()
    try:
        async for sources in mixer.stream_frame(FRAME_SECONDS, max_iter=max_iter):
            n += 1
            now = time.monotonic()
            for name in ("mic", "loopback"):
                src = sources[name]
                for ev in seg.on_audio(src, now, dt_seconds=FRAME_SECONDS):
                    stage = ev.get("stage")
                    uid = ev.get("utterance_id", "?")
                    if stage == "provisional":
                        print(f"{_LEVEL} PROV [{ev['speaker']}] u{uid} "
                              f"lang={ev['language']} conf={ev['confidence']} "
                              f"dur={ev['duration_ms']}ms floor={ev['noise_floor_db']:.0f}dB "
                              f"rms={ev['rms_db']:.0f}dB :: {ev['delta']!r}")
                    elif stage == "final":
                        print(f"{_LEVEL} FINAL [{ev['speaker']}] u{uid} "
                              f"lang={ev['language']} conf={ev['confidence']} "
                              f"low={ev['low_confidence']} dur={ev['duration_ms']}ms :: {ev['text']}")
                    else:
                        continue
                    await ws.send(ev)
            if now - last_metrics >= WS_METRICS_EVERY:
                await ws.send(ws.metrics_event())
                print(f"{_LEVEL} {ws.report()}")
                last_metrics = now
            elif n % 40 == 0:
                _log_rms(sources)
        if max_iter:
            print(f"{_LEVEL} max_iter={max_iter} atingido; encerrando.")
    except KeyboardInterrupt:
        print(f"\n{_LEVEL} parado.")
    finally:
        print(f"{_LEVEL} {ws.report()}")
        await ws.stop()


if __name__ == "__main__":
    if "--mock-once" in sys.argv:
        idx = sys.argv.index("--mock-once")
        text = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "What did you do during the weekend?"
        print(json.dumps({"stage": "final", "speaker": "OTHERS", "text": text,
                          "language": "en", "confidence": 0.99, "low_confidence": False},
                         ensure_ascii=False))
    elif MOCK:
        print(f"{_LEVEL} MOCK_MODE=true (modelo={MODEL}). Defina MOCK_MODE=false p/ realtime.")
        print(f"{_LEVEL} use scripts/simulate_meeting.py para validar o pipeline.")
    else:
        asyncio.run(run_realtime(max_iter=int(os.getenv("MAX_ITERATIONS", "0") or "0") or None))