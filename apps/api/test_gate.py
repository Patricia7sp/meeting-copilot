"""Testes do gate de confiança e da atribuição por origem (Feature 007).

Cobre os helpers usados por /ingest e /ws no servidor:
- locutor derivado da ORIGEM do áudio (source), vencendo o speaker enviado;
- gate MIN_FINAL_CONFIDENCE (finais abaixo do limiar não chegam ao Jev);
- status derivado: capturado (provisório) / em_revisao (final baixo) / confirmado.

Uso: python3 apps/api/test_gate.py
"""
from __future__ import annotations
import os
import sys

os.environ["MIN_FINAL_CONFIDENCE"] = "0.35"
os.environ["DATA_DIR"] = "/tmp/mc_gate_test"
os.makedirs(os.environ["DATA_DIR"], exist_ok=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import main as api  # noqa: E402


def test_speaker_from_source():
    assert api._speaker_from_source({"source": "mic"}) == "YOU"
    assert api._speaker_from_source({"source": "loopback"}) == "OTHERS"
    # a ORIGEM vence o speaker que o client enviar (nunca "YouTube como YOU")
    assert api._speaker_from_source({"source": "loopback", "speaker": "YOU"}) == "OTHERS"
    assert api._speaker_from_source({"source": "mic", "speaker": "OTHERS"}) == "YOU"
    assert api._speaker_from_source({}) == "OTHERS"
    assert api._speaker_from_source({"speaker": "YOU"}) == "YOU"
    print("ok: locutor por origem vence speaker")


def test_below_threshold():
    t = api._below_threshold
    assert t(low=False, confidence=0.5) is False, "acima do limiar passa"
    assert t(low=False, confidence=0.3) is True, "abaixo do limiar bloqueia"
    assert t(low=True, confidence=0.9) is True, "low_confidence bloqueia sempre"
    assert t(low=False, confidence=None) is False, "sem confiança = pass-through (compat)"
    assert t(low=False, confidence=0.35) is False, "no limiar passa"
    print("ok: gate MIN_FINAL_CONFIDENCE")


def test_status_derived():
    assert api._status_for("provisional", False) == "capturado"
    assert api._status_for("provisional", True) == "capturado"
    assert api._status_for("final", True) == "em_revisao"
    assert api._status_for("final", False) == "confirmado"
    print("ok: status capturado/em_revisao/confirmado")


def test_min_final_env():
    api.MIN_FINAL_CONFIDENCE = 0.5
    try:
        assert api._below_threshold(False, 0.4) is True, "env reflete no gate"
    finally:
        api.MIN_FINAL_CONFIDENCE = float(os.getenv("MIN_FINAL_CONFIDENCE", "0.35"))
    print("ok: MIN_FINAL_CONFIDENCE configurável por env")


def main():
    print("== apps/api/test_gate.py ==")
    for fn in (test_speaker_from_source, test_below_threshold,
               test_status_derived, test_min_final_env):
        fn()
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()