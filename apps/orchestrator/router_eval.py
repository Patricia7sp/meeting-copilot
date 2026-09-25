"""Avalia o classificador/roteador OpenRouter (Jev) num conjunto rotulado.

Uso:
  OPENROUTER_API_KEY=... python3 apps/orchestrator/router_eval.py --live
  python3 apps/orchestrator/router_eval.py            # dry: mostra os casos

Métricas: acurácia de ação, acurácia de "precisa de insight" e latências (p50/p95).
Corra com o mesmo ROUTER_MODEL usado em produção.
"""
from __future__ import annotations
import sys
import os
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from classifier import classify  # type: ignore

# (contexto, última fala, ação_esperada, need_esperado)
CASES = [
    ("PROFESSOR: What did you do during the weekend?", "I stay home... uhm...", "lang_coach", True),
    ("PROFESSOR: How do you say 'aproveitar' in English?", "How do you say aproveitar?", "lang_coach", True),
    ("ANA: o join do BigQuery está estourando bytes.", "E se a gente avaliar partition pruning?", "work_copilot", True),
    ("CARLOS: pipeline de ETL demorou 3h.", "Qual o comportamento esperado vs observado?", "work_copilot", True),
    ("", "uh", "ignore", False),
    ("", "hm ok", "ignore", False),
    ("", "Talvez essa pipeline esteja demorando porque estamos fazendo um join muito grande no BigQuery.", "work_copilot", True),
    ("PROFESSOR: Nice! And the past tense of 'go'?", "go went gone", "lang_coach", True),
]


def is_live() -> bool:
    return "--live" in sys.argv


def main():
    if not is_live() or not os.getenv("OPENROUTER_API_KEY"):
        print("[router-eval] sem chave/--live — apenas mostra os casos:")
        for ctx, last, want_a, want_n in CASES:
            print(f"  [{want_a}] {last[:60]!r}")
        print("\nCom chave rode: OPENROUTER_API_KEY=... python3 apps/orchestrator/router_eval.py --live")
        return
    import statistics
    n = ok_a = ok_n = 0
    lats = []
    for ctx, last, want_a, want_n in CASES:
        t0 = time.perf_counter()
        res = classify(ctx, last)
        dec: Any = res[0]
        used: Any = res[1]
        lat = time.perf_counter() - t0
        lats.append(lat)
        n += 1
        if dec is not None:
            got_a, got_n = dec.action, dec.needs_insight
            ok_a += got_a == want_a
            ok_n += got_n == want_n
            mark = "OK " if got_a == want_a else "x  "
            print(f"{mark} como={got_a:>11} prio={dec.priority:.2f} need={got_n} "
                  f"esperado={want_a:>11}/{want_n} :: {last[:50]!r}")
        else:
            print(f"x   classificador None (used={used}) :: {last[:50]!r}")
    print("\n--- resultado ---")
    print(f"acurácia ação = {ok_a}/{n} ({ok_a / n:.0%})")
    print(f"acurácia need = {ok_n}/{n} ({ok_n / n:.0%})")
    s = sorted(lats)
    p50 = s[len(s) // 2] if s else 0
    p95 = s[max(0, (len(s) * 95) // 100 - 1)] if s else 0
    print(f"latência por chamada: p50={p50 * 1000:.0f}ms p95={p95 * 1000:.0f}ms")
    print(f"(modelo ROUTER_MODEL={os.getenv('ROUTER_MODEL', '(cadeia free)')})")


if __name__ == "__main__":
    main()