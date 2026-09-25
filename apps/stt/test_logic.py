"""Testes de lógica do STT sem hardware/modelo (mock). Uso: python3 apps/stt/test_logic.py

Cobre: locutor por energia, silêncio descartado, anti-alucinação pós-whisper,
transcrição multilíngue com language/confidence no payload.
"""
from __future__ import annotations
import array
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../audio"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from capture import SourceAudio, SAMPLE_RATE  # noqa: E402
import service as stt  # noqa: E402


def fake_model(segment_text: str, lang: str = "en", avg_logprob: float = -0.4,
               no_speech_prob: float = 0.1, translation_text: str | None = None):
    """Modelo fake: um segmento fixo + info. Imita faster-whisper."""
    class Seg:
        def __init__(self, text, avg_logprob, no_speech_prob):
            self.text = text
            self.avg_logprob = avg_logprob
            self.no_speech_prob = no_speech_prob
            self.start, self.end = 0.0, 1.6
    class Info:
        language = lang
        language_probability = 0.98
    class M:
        def transcribe(self, audio, **kw):
            txt = translation_text if kw.get("task") == "translate" and translation_text else segment_text
            segs = iter([Seg(txt, avg_logprob, no_speech_prob)])
            if kw.get("task") == "translate":
                return segs, Info()
            return segs, Info()
    return M()


def pcm_silence():
    n = int(SAMPLE_RATE * stt.CHUNK_SECONDS)
    return array.array("h", [0] * n).tobytes()


def pcm_rms(rms: float):
    n = int(SAMPLE_RATE * stt.CHUNK_SECONDS)
    amp = int(32768 * min(max(rms, 0.0), 1.0))
    return array.array("h", [amp] * n).tobytes()


def test_silence_discarded():
    s = SourceAudio("mic", pcm_silence(), 0.0, active=True)
    ok, reason = stt.vad_gate(s)
    assert not ok and "floor" in reason, reason


def test_no_device_never_decides():
    # mesmo com rms alto, device_inactive manda descartar
    s = SourceAudio("mic", pcm_rms(0.05), 0.05, active=False)
    ok, reason = stt.vad_gate(s)
    assert not ok and "inactive" in reason, reason


def test_speaker_by_source():
    assert stt._speaker_of("mic") == "YOU"
    assert stt._speaker_of("loopback") == "OTHERS"


def test_gate_noise_floor_learns_ambient():
    # ruído ambiente constante (~-38dB) não vira fala após calibração do piso.
    # warmup default (GATE_WARMUP_FRAMES) cobre os primeiros frames de aprendizado.
    g = stt.NoiseFloorGate("mic")
    ambient = 0.0125
    for _ in range(60):
        r = g.evaluate(SourceAudio("mic", pcm_rms(ambient), ambient, active=True))
        assert not r.speech, (ambient, r)
    # piso convergido para perto do ruído ambiente (~-38dB)
    assert -39.0 < g.cal.floor_db < -37.5, g.cal.floor_db
    # fala real (~-20dB) acima do piso + histerese dispara
    r = g.evaluate(SourceAudio("mic", pcm_rms(0.1), 0.1, active=True))
    assert r.speech, r
    # histerese: queda para o piso (~-38dB) desliga a fala
    r = g.evaluate(SourceAudio("mic", pcm_rms(ambient), ambient, active=True))
    assert not r.speech and "silence_floor" in r.reason, r


def test_repetitive_hallucination_discarded():
    assert stt._post_filter("bye bye bye") is None
    assert stt._post_filter("Bye, bye, bye.") is None
    assert stt._post_filter("um um um") is None


def test_filler_discarded():
    assert stt._post_filter("uh") is None
    assert stt._post_filter("hmm.") is None


def test_emit_multilingual_with_meta():
    s = SourceAudio("mic", pcm_rms(0.05), 0.05, active=True)
    text, meta = stt.transcribe_source(s, fake_model("Ontem eu fui ao mercado", lang="pt"))
    assert text == "Ontem eu fui ao mercado"
    assert meta["lang"] == "pt"
    assert "confidence" in meta and meta["low"] is False


def test_translate_target_en_separate():
    s = SourceAudio("loopback", pcm_rms(0.05), 0.05, active=True)
    old = stt.TRANSLATE_TARGET
    stt.TRANSLATE_TARGET = "en"
    try:
        text, meta = stt.transcribe_audio(s.pcm, fake_model("Bom dia", lang="pt",
                                                            translation_text="Good morning"),
                                          final=True)
        assert text == "Bom dia"           # transcrição mantém o idioma
        assert meta["lang"] == "pt"
        assert meta["translation"] == "Good morning"  # tradução é separada (só final)
    finally:
        stt.TRANSLATE_TARGET = old


def test_low_confidence_flagged():
    s = SourceAudio("mic", pcm_rms(0.05), 0.05, active=True)
    text, meta = stt.transcribe_source(s, fake_model("buzzzt", avg_logprob=-2.5))
    assert text is not None and meta["low"] is True


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()