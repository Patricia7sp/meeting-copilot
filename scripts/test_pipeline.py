"""Teste pipeline puro (stdlib only). Roda sem instalar nada: python3 scripts/test_pipeline.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../packages/context"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../apps/orchestrator"))
from orchestrator import Orchestrator

def main():
    for mode in ("auto", "english", "work"):
        orch = Orchestrator(mode=mode, cooldown_seconds=0)
        print(f"\n=== modo {mode} ===")
        convo = [
            ("OTHERS", "What did you do during the weekend?"),
            ("YOU", "uhm"),
            ("OTHERS", "Talvez essa pipeline esteja demorando porque estamos fazendo um join muito grande no BigQuery."),
            ("OTHERS", "é..."),
        ]
        insights = 0
        for sp, tx in convo:
            out = orch.handle(sp, tx)
            print(f"[{sp}] {tx}\n  -> {out['type']}:{out.get('kind', out.get('reason'))}")
            if out.get("type") == "insight":
                insights += 1
                print(f"     card: {list(out['card'].keys())}")
        print(f"contexto:\n{orch.ctx.to_prompt_block()[:400]}")
        assert insights >= 1, f"modo {mode} deveria gerar >=1 insight"
    print("\nOK: pipeline gerou insights nos 3 modos.")

if __name__ == "__main__":
    main()
