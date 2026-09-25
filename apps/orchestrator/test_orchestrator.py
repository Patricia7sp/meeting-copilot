"""Testes do Orchestrator: limiar de confiança, dedup, contexto consolidado e router.

Uso: python3 apps/orchestrator/test_orchestrator.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../packages/context"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from orchestrator import Orchestrator  # noqa: E402


def orch(**kw):
    return Orchestrator(mode=kw.get("mode", "auto"), cooldown_seconds=0)


def test_low_confidence_never_insight():
    o = orch()
    out = o.handle("OTHERS",
                   "what did you do during the weekend asking friend",
                   confidence=0.3, low_confidence=True)
    assert out["type"] == "noop" and out["reason"] == "low_confidence", out


def test_confidence_threshold():
    o = orch()
    out = o.handle("OTHERS", "Talvez essa pipeline esteja demorando porque fazemos join no BigQuery",
                   confidence=0.4)  # abaixo de 0.5
    assert out["type"] == "noop" and out["reason"] == "low_confidence", out
    o2 = orch()
    out2 = o2.handle("OTHERS", "Talvez essa pipeline esteja demorando porque fazemos join no BigQuery",
                     confidence=0.9)
    assert out2["type"] == "insight", out2


def test_dedup_same_utterance():
    o = orch()
    t = "Talvez essa pipeline esteja demorando porque fazemos join no BigQuery"
    first = o.handle("OTHERS", t, confidence=0.9)
    second = o.handle("OTHERS", t, confidence=0.9)
    assert first["type"] == "insight"
    assert second["type"] == "noop" and second["reason"] == "dedup", second


def test_consolidated_context_merges_same_speaker():
    o = orch()
    o.observe("PROFESSOR", "What did you do during the weekend?")
    o.observe("YOU", "I stayed home")
    o.observe("YOU", "and went out on sunday")     # mesmo locutor consecutivo
    blob = o.ctx.window_text(6)
    assert "I stayed home and went out on sunday" in blob, blob
    assert blob.count("YOU:") == 1, blob  # falas consecutivas do mesmo locutor foram mescladas


def test_low_confidence_turn_excluded_from_context():
    o = orch()
    o.observe("PROFESSOR", "What did you do during the weekend?")
    o.observe("YOU", "bzzzt noise", confidence=0.1, low=True)
    o.observe("YOU", "I stayed home")
    blob = o.ctx.window_text(6)
    assert "bzzzt noise" not in blob
    assert "I stayed home" in blob


def test_priority_blocks_insight():
    o = orch()
    # "muito curto" => priority 0.1, abaixo de 0.2 => bloqueado apesar de route decidir ignore
    out = o.handle("OTHERS", "oi", confidence=0.9)
    # "oi" é curto: rota local decide ignore -> noop
    assert out["type"] == "noop"
    assert out["reason"] in ("muito curto",)


def test_local_router_without_key():
    o = orch()
    assert o.classifier is None  # sem OPENROUTER_API_KEY
    out = o.handle("OTHERS", "E se a gente avaliar partition pruning e clustering no join do BigQuery?",
                   confidence=0.9)
    assert out["type"] == "insight" and out["kind"] == "work_copilot", out


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()