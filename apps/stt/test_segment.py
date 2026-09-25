"""Testes da segmentação contínua (stream.py) sem hardware/modelo.

Uso: python3 apps/stt/test_segment.py

Cobre: silêncio descartado, fala contínua 60s sem lacunas, troca de fonte
(mic->YOU / loopback->OTHERS), borda entre janelas sobrepostas (dedup por
texto+tempo), refresca sem finalizar no max_utterance e reset entre enunciados.
"""
from __future__ import annotations
import array
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../audio"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from capture import SourceAudio, SAMPLE_RATE  # noqa: E402
from stream import Segmenter, GateResult, dedup_delta  # noqa: E402

FRAME = 0.25


def pcm(dur_s: float, level: int = 0) -> bytes:
    n = int(SAMPLE_RATE * dur_s)
    return array.array("h", [level] * n).tobytes()


def src(name: str, dur_s: float = FRAME, level: int = 0, active: bool = True) -> SourceAudio:
    rms = min(level / 32768.0, 1.0)
    return SourceAudio(name, pcm(dur_s, level), rms, active)


def rms_gate(speak_level: int):
    """Gate determinístico: nível de amostra >= `speak_level` é fala."""
    def g(s: SourceAudio) -> GateResult:
        lvl = int(s.pcm[0]) if s.pcm else 0
        if s.active and lvl >= speak_level:
            return GateResult(True, "speech_enter", -26.0, -46.0, None, s.active)
        return GateResult(False, "below_floor (delta -9dB)", -55.0, -46.0, None, s.active)
    return g


def gate_sticky(speak_level: int):
    """Gate com histerese simples: uma vez fala, sai só abaixo da metade do nível."""
    def g(s: SourceAudio) -> GateResult:
        if not s.pcm:
            return GateResult(False, "mock_backend", -99, -46, None, False)
        return GateResult(s.active and int(s.pcm[0]) >= speak_level, "",
                          -26.0, -46.0, None, s.active)
    return g


class WordStepper:
    """Fake whisper: devolve palavras da frase em janelas com fronteira repetida.

    A cada chamada entrega um bloco de `step` palavras terminando com as últimas
    `repeat` palavras já emitidas (comportamento real do Whisper na sobreposição),
    exercitando o dedup por texto+tempo do Segmenter.
    """

    def __init__(self, words, step: int = 3, repeat: int = 3):
        self.words = [w for w in words]
        self.step = step
        self.repeat = repeat
        self.prov = 0
        self.final_calls = 0

    def provisional(self, audio):
        self.prov += 1
        delivered = min(self.prov * self.step, len(self.words))
        start = max(0, delivered - self.repeat)
        return (" ".join(self.words[start:delivered]),
                {"lang": "en", "language_probability": 0.98,
                 "confidence": 0.9, "low": False})

    def final(self, audio):
        self.final_calls += 1
        return (" ".join(self.words),
                {"lang": "en", "language_probability": 0.99,
                 "confidence": 0.96, "low": False})

    @property
    def expected(self) -> str:
        return " ".join(self.words)


def make_segmenter(model, *, pause=0.7, max_utt=10.0, cap=60.0, log=None):
    return Segmenter(
        gate=rms_gate(1),
        provisional=model.provisional,
        final=model.final,
        pause_seconds=pause,
        overlap_seconds=0.5,
        max_utterance_seconds=max_utt,
        max_utterance_cap_seconds=cap,
        provisional_every_seconds=0.8,
        window_seconds=1.6,
        frame_seconds=FRAME,
        log=log or (lambda *a, **k: None),
    )


def feed_speech(seg, source: str, seconds: float, level: int = 8000, now: float = 0.0):
    """Alimenta fala contínua (nível fixo) frame a frame. Retorna (eventos, agora)."""
    evs = []
    n = int(round(seconds / FRAME))
    for _ in range(n):
        evs.extend(seg.on_audio(src(source, FRAME, level), now, dt_seconds=FRAME))
        now += FRAME
    return evs, now


def feed_silence(seg, source: str, seconds: float, now: float = 0.0):
    """Alimenta silêncio (nível 0). Retorna (eventos, agora)."""
    evs = []
    n = int(round(seconds / FRAME))
    for _ in range(n):
        evs.extend(seg.on_audio(src(source, FRAME, 0), now, dt_seconds=FRAME))
        now += FRAME
    return evs, now


def test_silence_no_events():
    model = WordStepper(["hello", "world"])
    seg = make_segmenter(model)
    evs, _ = feed_silence(seg, "mic", 5.0)
    assert evs == [], evs
    assert model.final_calls == 0


def test_continuous_speech_60s_one_consolidated_final():
    words = [f"w{i}" for i in range(200)]
    model = WordStepper(words)
    seg = make_segmenter(model, max_utt=10.0, cap=120.0)

    prov_evs, now = feed_speech(seg, "mic", 60.0)
    finals_during = [e for e in prov_evs if e.get("stage") == "final"]
    assert finals_during == [], ("não deve finalizar durante fala contínua", finals_during)
    prov_evs = [e for e in prov_evs if e.get("stage") == "provisional"]
    assert prov_evs, "deve emitir provisório durante a fala"

    fin_evs, _ = feed_silence(seg, "mic", 1.5, now=now)
    finals = [e for e in fin_evs if e.get("stage") == "final"]
    assert len(finals) == 1, ("um único final consolidado por enunciado", finals)
    f = finals[0]
    assert f["consolidated"] is True
    assert f["speaker"] == "YOU"
    assert f["text"] == model.expected
    assert f["duration_ms"] >= 60_000 - 400, ("60s completos, sem lacunas", f["duration_ms"])
    # provisório corretamente deduplicado: nenhuma palavra repetida/garada
    last_prov = prov_evs[-1]["text"]
    assert last_prov == model.expected, (last_prov, model.expected)


def test_brief_dip_does_not_split_utterance():
    words = ["one", "two", "three", "four", "five", "six"]
    model = WordStepper(words)
    seg = make_segmenter(model, pause=0.7)
    evs, now = feed_speech(seg, "mic", 2.0)
    assert not any(e.get("stage") == "final" for e in evs)
    # 0.5s de dip (< pause 0.7s) NÃO finaliza
    dip, now = feed_silence(seg, "mic", 0.5, now=now)
    assert not any(e.get("stage") == "final" for e in dip)
    _, now = feed_speech(seg, "mic", 2.0, now=now)
    fin, _ = feed_silence(seg, "mic", 1.0, now=now)
    finals = [e for e in fin if e.get("stage") == "final"]
    assert len(finals) == 1, finals
    assert finals[0]["duration_ms"] >= 4_000, finals[0]


def test_source_switching_you_and_others():
    words = ["a", "b", "c", "d"]
    model = WordStepper(words)
    seg = make_segmenter(model)
    mic_events, _ = feed_speech(seg, "mic", 2.0)
    mic_prov = [e for e in mic_events if e.get("stage") == "provisional"]
    assert mic_prov and all(e["speaker"] == "YOU" and e["source"] == "mic" for e in mic_prov)

    lb_events, _ = feed_speech(seg, "loopback", 2.0)
    lb_prov = [e for e in lb_events if e.get("stage") == "provisional"]
    assert lb_prov and all(e["speaker"] == "OTHERS" and e["source"] == "loopback" for e in lb_prov)


def test_chunk_boundary_dedup_no_loss():
    words = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    model = WordStepper(words, step=2, repeat=2)  # janela pequena: fronteira a cada 0.5s de info
    seg = make_segmenter(model, max_utt=10.0, cap=60.0)
    evs, now = feed_speech(seg, "mic", 5.0)
    prov = [e for e in evs if e.get("stage") == "provisional"]
    running = prov[-1]["text"] if prov else ""
    # dedup por fronteira NÃO pode perder palavra nem duplicar
    parts = running.split()
    assert set(parts) == set(model.words), (parts, model.words)
    assert len(parts) == len(model.words), (parts, model.words)


def test_max_utterance_refreshes_but_does_not_finalize():
    words = [f"p{i}" for i in range(150)]
    model = WordStepper(words)
    seg = make_segmenter(model, max_utt=3.0, cap=120.0)
    evs, now = feed_speech(seg, "mic", 10.0)
    assert not any(e.get("stage") == "final" for e in evs), \
        "max_utterance só força refresh do provisório, nunca finaliza"
    fin, _ = feed_silence(seg, "mic", 1.0, now=now)
    finals = [e for e in fin if e.get("stage") == "final"]
    assert len(finals) == 1 and finals[0]["duration_ms"] >= 9_000, finals


def test_utterance_resets_after_final():
    words = ["ok", "then", "good", "bye"]
    model = WordStepper(words)
    seg = make_segmenter(model)
    _, now = feed_speech(seg, "mic", 1.0)
    fin1, now = feed_silence(seg, "mic", 1.0, now=now)
    finals1 = [e for e in fin1 if e.get("stage") == "final"]
    assert len(finals1) == 1
    uid1 = finals1[0]["utterance_id"]
    _, now = feed_speech(seg, "mic", 1.0, now=now)
    fin2, _ = feed_silence(seg, "mic", 1.0, now=now)
    finals2 = [e for e in fin2 if e.get("stage") == "final"]
    assert len(finals2) == 1 and finals2[0]["utterance_id"] != uid1


def test_dedup_delta():
    assert dedup_delta("bom dia", ["bom"]) == ("dia", ["bom", "dia"])
    assert dedup_delta("dia equipe", ["bom", "dia"]) == ("equipe", ["dia", "equipe"])
    assert dedup_delta("novo texto", []) == ("novo texto", ["novo", "texto"])


def test_mock_and_inactive_never_speech():
    seg = Segmenter(gate=lambda s: GateResult(False, "device_inactive", -99, -46, None, False),
                    provisional=lambda a: (None, {}), final=lambda a: (None, {}),
                    frame_seconds=FRAME, log=lambda *a, **k: None)
    evs1 = seg.on_audio(SourceAudio("mic", pcm(FRAME, 8000), 0.24, True, mock=True), 0.0, dt_seconds=FRAME)
    assert evs1 == []
    evs2 = seg.on_audio(SourceAudio("loopback", pcm(FRAME, 8000), 0.24, False), 0.0, dt_seconds=FRAME)
    assert evs2 == []


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()