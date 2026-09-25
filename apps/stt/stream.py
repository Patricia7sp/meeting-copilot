"""Segmentação contínua de fala por fonte (mic/loopback).

Modelo de operação (frames contínuos ~250ms por fonte):

- gate por fonte (noise floor + histerese + VAD) decide fala/silêncio com
  resolução fina; calibração de ruído por FONTE, nunca por `RMS > limiar` fixo.
- fala acumula PCM do enunciado; a única coisa que finaliza é:
    1. silêncio REAL >= `pause_seconds` (500-800ms por default); ou
    2. fala > `max_utterance_cap_seconds` (limite de memória — 120s normalmente
       não é atingido numa aula; a fala finalizada cobre o enunciado inteiro).
- provisório roda janelas sobrepostas de ~`window_seconds` sobre o enunciado,
  deduplicando a fronteira por texto+tempo (`prov_len - overlap` + `dedup_delta`);
  `max_utterance_seconds` só FORÇA um refresh do provisório (limita o contexto do
  whisper em janelas de 8-12s), não finaliza — garantindo UM `final` consolidado
  por enunciado, sem lacunas causadas por chunking.
- no silêncio, o enunciado inteiro é re-transcrito de uma vez (`final`) e o
  evento final carrega `consolidated=True` com o texto completo.

Nenhuma decisão de locutor aqui: mic -> YOU, loopback -> OTHERS (definido pelo
gate de quem usa esta classe). O gate recebe `SourceAudio` e devolve
`GateResult` (fala?, motivo, RMS dB, piso dB, ratio VAD).
"""
from __future__ import annotations
import itertools
import re
from dataclasses import dataclass, field
from typing import Callable

from capture import SourceAudio, SAMPLE_RATE  # type: ignore


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u00C0-\u024F']+", text.lower())


def dedup_delta(text: str, prev_tail_words: list[str]) -> tuple[str, list[str]]:
    """Remove do texto as palavras que já apareceram no fim da janela anterior.

    Retorna (delta_substituido, tail_words). A sobreposição faz o Whisper repetir
    a fronteira; esta função deduplica por TEXTO o que já foi emitido. O texto
    definitivo do enunciado vem da re-transcrição final (sem dependência aqui).
    """
    words = _tokens(text)
    if not words:
        return text, prev_tail_words[-3:]
    tail = [t for t in prev_tail_words if t]
    drop = None
    for cnt in range(min(len(tail), len(words)), 0, -1):
        if words[:cnt] == tail[-cnt:]:
            drop = cnt
    kept = words[drop:] if drop else words
    return " ".join(kept), words[-3:]


@dataclass
class GateResult:
    """Resultado do gate de UMA fonte para UM frame."""
    speech: bool
    reason: str = ""
    rms_db: float = -99.0
    floor_db: float = -46.0
    vad_ratio: float | None = None
    active: bool = True


@dataclass
class _SrcState:
    source: str
    utterance_id: int = 0
    utt_buf: bytearray = field(default_factory=bytearray)   # PCM total (fala + silêncio-pad)
    utt_dur: float = 0.0                                    # duração de FALA acumulada
    started_ts: float = 0.0
    last_speech_ts: float = 0.0
    speech: bool = False
    finalized: bool = False
    prov_len: int = 0                                       # bytes já cobertos pelo provisório
    prov_deltas: list[str] = field(default_factory=list)
    prov_tail: list[str] = field(default_factory=list)
    last_prov_dur: float = 0.0
    rms_vals: list[float] = field(default_factory=list)
    vad_ratios: list[float] = field(default_factory=list)


class Segmenter:
    """Acumula frames por fonte e emite `provisional`/`final` consolidado."""

    def __init__(self, *, gate: Callable, provisional: Callable, final: Callable,
                 pause_seconds: float = 0.7, overlap_seconds: float = 0.5,
                 max_utterance_seconds: float = 10.0,
                 max_utterance_cap_seconds: float = 120.0,
                 provisional_every_seconds: float = 0.8,
                 window_seconds: float = 1.6, frame_seconds: float = 0.25,
                 log: Callable = print):
        self.gate = gate
        self.provisional = provisional
        self.final = final
        self.pause_seconds = pause_seconds
        self.overlap_seconds = overlap_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self.max_utterance_cap_seconds = max_utterance_cap_seconds
        self.provisional_every_seconds = provisional_every_seconds
        self.window_seconds = window_seconds
        self.frame_seconds = frame_seconds
        self.log = log
        self._states: dict[str, _SrcState] = {}
        self._utt_seq = itertools.count(1)
        self._speaker = {"mic": "YOU", "loopback": "OTHERS"}

    # ---------- entrada ----------

    def on_audio(self, src: SourceAudio, now: float, dt_seconds: float | None = None) -> list[dict]:
        """Processa UM frame de UMA fonte. Retorna eventos (provisional|final)."""
        dt = dt_seconds or self.frame_seconds
        st = self._states.setdefault(src.source, _SrcState(source=src.source))
        res = self.gate(src)
        if res.speech:
            return self._on_speech(st, src, now, dt, res)
        return self._on_silence(st, src, now, dt, res)

    # ---------- fala ----------

    def _on_speech(self, st: _SrcState, src: SourceAudio, now: float,
                   dt: float, res: GateResult) -> list[dict]:
        events: list[dict] = []
        if st.finalized or not st.rms_vals:
            st.utterance_id = next(self._utt_seq)
            st.started_ts = now
        elif not st.speech:
            self.log(f"[stream] {src.source}: fala ativa (dur={st.utt_dur:.1f}s "
                     f"rms={res.rms_db:.0f}dB floor={res.floor_db:.0f}dB vad={res.vad_ratio})")
        st.speech = True
        st.last_speech_ts = now
        st.utt_buf.extend(src.pcm)
        st.utt_dur += dt
        st.rms_vals.append(src.rms)
        if res.vad_ratio is not None:
            st.vad_ratios.append(res.vad_ratio)

        new_speech = st.utt_dur - st.last_prov_dur
        force = st.utt_dur >= self.max_utterance_seconds
        if new_speech >= self.provisional_every_seconds or force:
            prov = self._make_provisional(st, src, res)
            if prov:
                events.append(prov)
            if force and not st.finalized:
                self.log(f"[stream] {src.source}: fala {st.utt_dur:.1f}s >= "
                         f"{self.max_utterance_seconds}s — refresh provisório (não finaliza)")
        if st.utt_dur >= self.max_utterance_cap_seconds:
            self.log(f"[stream] {src.source}: cap de memória "
                     f"{self.max_utterance_cap_seconds}s atingido; finalizando por cap")
            fin = self._finalize(st, src, now, res, reason="max_utterance_cap")
            if fin:
                events.append(fin)
        return events

    # ---------- silêncio ----------

    def _on_silence(self, st: _SrcState, src: SourceAudio, now: float,
                    dt: float, res: GateResult) -> list[dict]:
        events: list[dict] = []
        if st.finalized or not st.rms_vals:
            if not st.rms_vals and res.reason and "speech_enter" not in res.reason:
                self.log(f"[stream] {src.source}: descarte {res.reason} "
                         f"(rms={res.rms_db:.0f}dB floor={res.floor_db:.0f}dB "
                         f"vad={res.vad_ratio})")
            return events
        if st.speech:
            self.log(f"[stream] {src.source}: silêncio início (dur={st.utt_dur:.1f}s "
                     f"rms={res.rms_db:.0f}dB floor={res.floor_db:.0f}dB vad={res.vad_ratio})")
            st.speech = False
        idle = now - st.last_speech_ts
        if idle >= self.pause_seconds:
            fin = self._finalize(st, src, now, res, reason="silence")
            if fin:
                events.append(fin)
        return events

    # ---------- provisório (janela sobreposta + dedup por texto/tempo) ----------

    def _make_provisional(self, st: _SrcState, src: SourceAudio,
                          res: GateResult) -> dict | None:
        overlap_bytes = max(int(self.overlap_seconds * SAMPLE_RATE) * 2, SAMPLE_RATE)
        start = max(0, st.prov_len - overlap_bytes)
        cand = bytes(st.utt_buf[start:])
        max_bytes = int(self.window_seconds * SAMPLE_RATE) * 2
        if len(cand) > max_bytes:
            cand = cand[-max_bytes:]
        try:
            text, meta = self.provisional(cand)
        except Exception as e:
            self.log(f"[stream] erro provisional [{src.source}]: {e}")
            text, meta = None, {"discard": "provisional_error"}
        st.prov_len = len(st.utt_buf)
        st.last_prov_dur = st.utt_dur
        if not text:
            return None
        delta, tail = dedup_delta(text, st.prov_tail)
        st.prov_tail = [w for w in tail if w][-3:]
        if not delta:
            return None
        running = " ".join(st.prov_deltas + [delta]).strip()
        st.prov_deltas.append(delta)
        return {
            "type": "transcript",
            "stage": "provisional",
            "source": src.source,
            "speaker": self._speaker[src.source],
            "utterance_id": st.utterance_id,
            "delta": delta,
            "text": running,
            "language": meta.get("lang", "unknown"),
            "language_probability": meta.get("language_probability", 0.0),
            "confidence": round(float(meta.get("confidence", 0.0)), 3),
            "low_confidence": bool(meta.get("low", False)),
            "rms": round(src.rms, 4),
            "rms_db": round(res.rms_db, 1),
            "noise_floor_db": round(res.floor_db, 1),
            "vad_ratio": round(float(res.vad_ratio), 3) if res.vad_ratio is not None else None,
            "duration_ms": int(st.utt_dur * 1000),
            "discard_reason": res.reason,
        }

    # ---------- final consolidado ----------

    def _finalize(self, st: _SrcState, src: SourceAudio, now: float,
                  res: GateResult, *, reason: str) -> dict | None:
        if st.finalized or not st.rms_vals:
            return None
        st.finalized = True
        audio = bytes(st.utt_buf)
        dur = len(audio) / 2 / SAMPLE_RATE
        try:
            text, meta = self.final(audio)
        except Exception as e:
            self.log(f"[stream] erro revisão final [{src.source}]: {e}")
            text, meta = None, {"discard": "final_error"}
        self._states.pop(src.source, None)
        if not text:
            self.log(f"[stream] DISCARD final [{src.source}]: {meta.get('discard')}")
            return None
        rms_mean = (sum(st.rms_vals) / len(st.rms_vals)) if st.rms_vals else 0.0
        conf = round(float(meta.get("confidence", 0.0)), 3)
        self.log(f"[stream] FINAL [{self._speaker[src.source]}] {reason} "
                 f"dur={dur * 1000:.0f}ms conf={conf} :: {text[:160]}")
        return {
            "type": "transcript",
            "stage": "final",
            "consolidated": True,
            "source": src.source,
            "speaker": self._speaker[src.source],
            "utterance_id": st.utterance_id,
            "text": text,
            "provisional_text": " ".join(st.prov_deltas).strip(),
            "language": meta.get("lang", "unknown"),
            "language_probability": meta.get("language_probability", 0.0),
            "confidence": conf,
            "low_confidence": bool(meta.get("low", False)),
            "duration_ms": int(dur * 1000),
            "rms_mean": round(rms_mean, 3),
            "rms_db": round(res.rms_db, 1),
            "noise_floor_db": round(res.floor_db, 1),
            "vad_ratio": round(float(res.vad_ratio), 3) if res.vad_ratio is not None else None,
            "discard_reason": reason,
            "translation": meta.get("translation"),
        }

    def reset(self) -> None:
        self._states.clear()