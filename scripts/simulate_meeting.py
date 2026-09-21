"""Simula reunião alimentando a API via HTTP ou WS. Uso:
  python3 scripts/simulate_meeting.py --mode english --via http
  python3 scripts/simulate_meeting.py --mode work --via ws
Sem API no ar, roda em dry-run local (só orchestrator).
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../packages/context"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../apps/orchestrator"))

SCENES = {
    "english": [
        ("PROFESSOR", "What did you do during the weekend?"),
        ("YOU", "I stay home... uhm..."),
        ("PROFESSOR", "Nice! And how do you say 'aproveitar' in English?"),
    ],
    "work": [
        ("ANA", "Talvez essa pipeline esteja demorando porque estamos fazendo um join muito grande no BigQuery."),
        ("YOU", "Faz sentido, quanto de bytes o job está escaneando?"),
        ("CARLOS", "E se a gente avaliar partition pruning e clustering?"),
    ],
}

def dry_run(mode):
    from orchestrator import Orchestrator
    orch = Orchestrator(mode="auto", cooldown_seconds=0)
    for sp, tx in SCENES[mode]:
        out = orch.handle(sp, tx)
        print(f"[{sp}] {tx}\n  -> {json.dumps(out, ensure_ascii=False)[:300]}")
        time.sleep(0.3)

def via_http(mode):
    for sp, tx in SCENES[mode]:
        req = urllib.request.Request("http://localhost:8000/ingest",
            data=json.dumps({"speaker": sp, "text": tx}).encode(),
            headers={"Content-Type": "application/json"})
        print(urllib.request.urlopen(req, timeout=5).read().decode()[:400])
        time.sleep(1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="english", choices=["english", "work"])
    ap.add_argument("--via", default="dry", choices=["dry", "http"])
    a = ap.parse_args()
    (dry_run if a.via == "dry" else via_http)(a.mode)
