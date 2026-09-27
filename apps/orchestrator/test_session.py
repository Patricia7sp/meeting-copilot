"""Testes do Classificador de Sessão (Feature 005): CA-1..CA-7 + persistência.

Uso: python3 apps/orchestrator/test_session.py
"""
from __future__ import annotations
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../packages/context"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from orchestrator import Orchestrator  # noqa: E402
from session import SessionResult, SessionTracker  # noqa: E402

TMP = tempfile.mkdtemp(prefix="mc_session_")

EN_1 = "what did you do during the weekend"
EN_2 = "how do you say table in english"
EN_3 = "teacher how do you use present perfect"
TECH_1 = "o job está lendo muitas bytes por causa do join"
TECH_2 = "precisamos de cache e arquitetura melhor no api"
TECH_3 = "vamos avaliar partition pruning e clustering no join"
GEN_1 = "a janta estava ótima ontem na casa da minha mãe"
GEN_2 = "o trânsito hoje está horrível né"
GEN_3 = "vamos marcar de ver um filme sábado à noite"


def orch():
    d = tempfile.mkdtemp(prefix="mc_session_")
    return Orchestrator(mode="auto", cooldown_seconds=0, session_id="t", session_dir=d)


def feed(o, texts, conf=0.9):
    out = []
    for t in texts:
        out.append(o.handle("YOU", t, confidence=conf))
    return out


def prof(o):
    s = o.session
    assert s is not None
    return s.to_dict()


def tracker(**kw):
    d = tempfile.mkdtemp(prefix="mc_session_")
    return SessionTracker(session_id="t", session_dir=d, **kw)


def test_ca1_english_lesson_local():
    o = orch()  # sem OPENROUTER_API_KEY -> fallback local
    feed(o, [EN_1, EN_2, EN_3])
    assert prof(o)["type"] == "english_lesson", prof(o)
    assert prof(o)["source"] == "local"


def test_ca2_technical_meeting_local():
    o = orch()
    feed(o, [TECH_1, TECH_2, TECH_3])
    assert prof(o)["type"] == "work_meeting", prof(o)


def test_ca3_vague_conversation():
    o = orch()
    feed(o, [GEN_1, GEN_2, GEN_3])
    assert prof(o)["type"] == "general_conversation", prof(o)


def test_ca4_short_or_low_not_classified():
    o = orch()
    o.handle("YOU", "oi", confidence=0.9)                       # curto
    o.handle("YOU", EN_1, confidence=0.3, low_confidence=True) # baixa
    s = o.session
    assert s is not None
    assert len(s.window) == 0, s.window
    assert prof(o)["type"] == "unknown"


def test_ca5_isolated_topic_does_not_switch():
    o = orch()
    feed(o, [TECH_1, TECH_2, TECH_3])
    assert prof(o)["type"] == "work_meeting"
    r = o.handle("YOU", EN_1, confidence=0.9)                  # 1 fala isolada
    assert prof(o)["type"] == "work_meeting", prof(o)
    assert not (r.get("session") or {}).get("changed", False), r


def test_ca6_sustained_change_commits():
    o = orch()
    feed(o, [TECH_1, TECH_2, TECH_3])
    assert prof(o)["type"] == "work_meeting"
    o.handle("YOU", EN_1, confidence=0.9)       # 1 isolado -> não muda
    assert prof(o)["type"] == "work_meeting", prof(o)
    o.handle("YOU", EN_2, confidence=0.9)       # 2 últimos convergem + conf > atual -> commit
    assert prof(o)["type"] == "english_lesson", prof(o)


def test_jev_higher_confidence_swaps_immediately():
    t = tracker()
    for txt in (EN_1, EN_2, EN_3):
        t.observe_final(txt, speaker="YOU", confidence=0.9)
    assert t.to_dict()["type"] == "english_lesson"
    note = t.observe_final(TECH_1, speaker="YOU", confidence=0.9,
                           classification=SessionResult(type="work_meeting",
                                                        confidence=0.95, source="jev"))
    assert note["changed"] is True and t.to_dict()["type"] == "work_meeting"
    assert t.to_dict()["source"] == "jev"


def test_jev_low_confidence_does_not_swap():
    t = tracker()
    for txt in (EN_1, EN_2, EN_3):
        t.observe_final(txt, speaker="YOU", confidence=0.9)
    note = t.observe_final(TECH_1, speaker="YOU", confidence=0.9,
                           classification=SessionResult(type="work_meeting",
                                                        confidence=0.4, source="jev"))
    assert note["changed"] is False and t.to_dict()["type"] == "english_lesson"


def test_ca7_jev_failure_falls_back_local_without_blocking():
    o = orch()

    def boom(ctx, last):
        raise RuntimeError("jev unavailable")

    o.classifier = boom
    out = feed(o, [EN_1, EN_2, EN_3])
    assert all(x["type"] in ("insight", "noop") for x in out)   # pipeline não quebra
    assert prof(o)["type"] == "english_lesson"
    assert prof(o)["source"] == "local"


def test_window_limited_to_eight_with_summary():
    t = tracker()
    for i in range(12):
        t.observe_final(f"fala numero {i} sobre join e bigquery", speaker="YOU", confidence=0.9)
    assert len(t.window) == 8, len(t.window)
    assert t._summary, "resumo dos antigos deve existir"


def test_persistence_reload():
    d = tempfile.mkdtemp(prefix="mc_persist_")
    t1 = SessionTracker(session_id="p1", session_dir=d)
    for txt in (EN_1, EN_2, EN_3):
        t1.observe_final(txt, speaker="YOU", confidence=0.9)
    assert t1.to_dict()["type"] == "english_lesson"
    t2 = SessionTracker(session_id="p1", session_dir=d)
    assert t2.to_dict()["type"] == "english_lesson"
    assert os.path.exists(os.path.join(d, "p1_session.json"))


def test_interval_and_fingerprint_gate():
    t = tracker()
    fp = t.ctx_fingerprint()
    t.on_call_done(fp)
    assert t.should_call(fp) is False     # mesmo fingerprint (dedup) + intervalo
    assert t.should_call(None) is False   # ainda dentro do intervalo mínimo


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()