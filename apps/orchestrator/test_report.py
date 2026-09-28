"""Testes do Resumo Markdown por sessão (Feature 006): CA 7, CA 8 e qualidade.

Uso: python3 apps/orchestrator/test_report.py
"""
from __future__ import annotations
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../packages/context"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from orchestrator import Orchestrator  # noqa: E402
from report import (build_report, clean_entries, render_markdown,  # noqa: E402
                    save_report, display_path)

TMP = tempfile.mkdtemp(prefix="mc_report_")

WORK = [
    "a reunião de hoje é sobre o join no BigQuery e o custo do pipeline de dados",
    "o join atual está escaneando 2TB por causa da falta de partition pruning",
    "decidimos adotar particionamento por data e clustering nas chaves do join",
    "precisamos medir o custo antes do deploy e atualizar a documentação da arquitetura",
    "vamos revisar os próximos passos com o time de dados na sexta",
]
LESSON = [
    "teacher what did you do during the weekend",
    "how do you say table in english",
    "você deveria dizer it was nice rather than it was fine",
    "let's practice the past tense with the verb go",
    "phrasal verb carry on means continue",
]


def entries(texts, conf=0.92, duration_ms=20000):
    return [{"speaker": "OTHERS", "text": t, "confidence": conf, "duration_ms": duration_ms}
            for t in texts]


def render(entries_in, stype="work_meeting", conf=0.91):
    md, _ = build_report(session_id="2026-09-27_1430",
                         profile={"type": stype, "confidence": conf, "source": "local"},
                         entries=entries_in)
    return md


def test_ca7_meeting_has_decisions_and_pending():
    md = render(entries(WORK))
    assert "## Resumo executivo" in md
    assert "## Decisões" in md
    assert "## Pendências e próximos passos" in md
    body = md[md.index("## Decisões"):]
    assert "decidimos adotar" in body, body[:400]
    assert "precisamos medir o custo" in body, body[:400]
    assert "vamos revisar os próximos passos" in body, body[:600]
    assert "Reunião de trabalho" in md
    assert "91%" in md


def test_ca8_lesson_has_vocab_corrections_suggestions():
    md = render(entries(LESSON), stype="english_lesson")
    assert "## Vocabulário" in md
    assert "## Correções" in md
    assert "## Frases sugeridas" in md
    assert "## Pontos de aprendizado" in md
    assert "table" in md[md.index("## Vocabulário"):md.index("## Correções")]
    assert "rather than" in md[md.index("## Correções"):md.index("## Frases sugeridas")]
    frases = md[md.index("## Frases sugeridas"):md.index("## Pontos de aprendizado")]
    assert "how do you say" in frases
    assert "past tense" in md[md.index("## Pontos de aprendizado"):]
    assert "Aula de inglês" in md
    tx = md[md.index("## Transcrição consolidada"):]
    assert tx.count("\n- ") == len(LESSON)
    assert "\n- Outros: teacher what did you do during the weekend\n" in tx


def test_general_conversation_minimal():
    md = render(entries(["o trânsito hoje está horrível", "a janta estava ótima ontem"]),
                stype="general_conversation")
    assert "## Decisões" not in md
    assert "## Vocabulário" not in md
    assert "## Transcrição consolidada" in md
    assert "Conversa geral" in md


def test_unknown_without_content_has_no_transcript():
    md = render([], stype="unknown")
    assert "## Transcrição consolidada" not in md
    assert "sem falas finais consolidadas" in md


def test_naming_2026_09_27_1430_work_meeting():
    d = tempfile.mkdtemp(prefix="mc_report_name_")
    md = render(entries(WORK))
    p = save_report(d, "2026-09-27_1430", {"type": "work_meeting", "confidence": 0.91}, md)
    assert p.name == "2026-09-27_1430_work-meeting.md", p.name
    assert p.exists()
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# Resumo da Sessão\n")
    assert "decisionadas".lower() not in text.lower()


def test_atomic_write_no_tmp_leftovers():
    d = tempfile.mkdtemp(prefix="mc_report_atom_")
    md = render(entries(WORK))
    save_report(d, "sess_x", {"type": "work_meeting"}, md)
    left = [f for f in os.listdir(d) if f.endswith(".tmp")]
    assert left == [], left


def test_clean_entries_filters_provisional_low_and_duplicates():
    raw = [
        {"speaker": "YOU", "text": "o job escaneia 2TB por falta de partition pruning",
         "stage": "final", "consolidated": True, "confidence": 0.9},
        {"speaker": "YOU", "text": "o job escaneia 2TB por falta de partition pruning",
         "stage": "final", "consolidated": True, "confidence": 0.9},           # repetição
        {"speaker": "YOU", "text": "provável frase provisória", "stage": "provisional",
         "consolidated": False, "confidence": 0.7},                            # provisório
        {"speaker": "YOU", "text": "fala de baixa confiança", "stage": "final",
         "consolidated": True, "confidence": 0.3, "low_confidence": True},     # baixa conf
        {"speaker": "OTHERS", "text": "decidimos adotar clustering nas chaves",
         "stage": "final", "consolidated": True, "confidence": 0.8},
    ]
    out = clean_entries(raw)
    assert len(out) == 2, [(e["text"]) for e in out]
    assert all(e["text"] != "fala de baixa confiança" for e in out)
    assert out[0]["text"].startswith("o job escaneia")


def test_load_entries_from_jsonl():
    d = tempfile.mkdtemp(prefix="mc_report_jsonl_")
    import json
    path = os.path.join(d, "s.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"speaker": "YOU", "text": "decidimos adotar o cluster",
                             "stage": "final", "consolidated": True,
                             "confidence": 0.9, "duration_ms": 15000}) + "\n")
        fh.write(json.dumps({"speaker": "YOU", "text": "duplicado",
                             "stage": "final", "consolidated": True,
                             "confidence": 0.1, "low_confidence": True}) + "\n")
    from report import load_entries
    got = load_entries(path)
    assert len(got) == 1, got
    assert got[0]["text"] == "decidimos adotar o cluster"


def test_orchestrator_end_session_writes_markdown():
    d = tempfile.mkdtemp(prefix="mc_report_ork_")
    o = Orchestrator(mode="auto", cooldown_seconds=0,
                     session_id="2026-09-27_1900", session_dir=d)
    for t in WORK:
        o.handle("OTHERS", t, confidence=0.92, duration_ms=20000)
    done = o.end_session(None)
    assert done is not None and done["type"] == "session_end", done
    assert done["session_type"] == "work_meeting"
    path = done.get("report_path")
    assert path, done
    full = path if os.path.isabs(path) else os.path.join(
        os.getcwd(), path) if not path.startswith(os.path.join("data", "sessions")) \
        else os.path.join(os.path.dirname(__file__), "..", "..", path)
    assert os.path.exists(full), full
    md = open(full, encoding="utf-8").read()
    assert "## Decisões" in md and "decidimos adotar" in md
    assert "2026-09-27_1900" in md


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()