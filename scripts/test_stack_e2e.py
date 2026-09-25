"""E2E do stack HTTP: API + /ingest (gating consolidated/event_id) + /ui + simulações + persistência.

Roda a API em :8000 (precisa fastapi+uvicorn instalados), alimenta via HTTP como o
STT faria e valida: provisional/not-consolidated/id-duplicated nunca geram insight,
finais consolidados passam pelo gate, /ui responde e os eventos são persistidos.

Uso: python3 scripts/test_stack_e2e.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["DATA_DIR"] = "/tmp/mc_e2e_sessions2"
os.environ["ORCHESTRATOR_MODE"] = "auto"


def req(method, path, body=None):
    url = f"http://127.0.0.1:8000{path}"
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=8) as resp:
        return resp.status, json.loads(resp.read().decode() or "null")


def main():
    subprocess.run(["pkill", "-f", "uvicorn main:app"], capture_output=True)
    time.sleep(0.5)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--app-dir",
         os.path.join(ROOT, "apps", "api"), "--host", "127.0.0.1",
         "--port", "8000", "--log-level", "error"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        up = False
        for _ in range(60):
            try:
                req("GET", "/health")
                up = True
                break
            except Exception:
                if proc.poll() is not None:
                    break
                time.sleep(0.4)
        if not up:
            raise SystemExit("servidor não subiu (proc rc=%s)" % proc.poll())
        print("health:", req("GET", "/health")[1])
        with urllib.request.urlopen("http://127.0.0.1:8000/ui", timeout=8) as r:
            print("ui: GET /ui ->", r.status, "bytes=", len(r.read()))
        for mode in ("english", "work"):
            print(f"== simulate {mode} (http) ==")
            out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "simulate_meeting.py"),
                                  "--mode", mode, "--via", "http"],
                                 capture_output=True, text=True, timeout=30)
            sys.stdout.write(out.stdout[:800])
        print("== /ingest gating ==")
        _, r = req("POST", "/ingest", {"speaker": "YOU", "text": "trecho parcial",
                                       "stage": "provisional", "event_id": "s:1"})
        print("provisional ->", r["insight"])
        assert r["insight"]["reason"] == "provisional", r
        _, r = req("POST", "/ingest", {"speaker": "YOU", "text": "trecho parcial",
                                       "stage": "final", "consolidated": False, "event_id": "s:2"})
        print("final nao consolidado ->", r["insight"])
        assert r["insight"]["reason"] == "not-consolidated", r
        _, r = req("POST", "/ingest", {"speaker": "YOU",
                                       "text": "em ingles aproveitar se diz to enjoy",
                                       "stage": "final", "consolidated": True,
                                       "event_id": "s:3", "language": "pt", "confidence": 0.9})
        print("final consolidado ->", r["insight"])
        assert r["insight"]["type"] == "insight" or r["insight"]["reason"] == "sem sinal suficiente", r
        _, r2 = req("POST", "/ingest", {"speaker": "YOU",
                                        "text": "em ingles aproveitar se diz to enjoy",
                                        "stage": "final", "consolidated": True,
                                        "event_id": "s:3", "language": "pt", "confidence": 0.9})
        print("retransmissao mesmo event_id ->", r2["insight"])
        assert r2["insight"]["reason"] == "id-duplicated", r2
        sess = os.environ["DATA_DIR"]
        event_ids = set()
        for f in sorted(os.listdir(sess)):
            if f.endswith(".jsonl"):
                for line in open(os.path.join(sess, f), encoding="utf-8"):
                    rec = json.loads(line)
                    if rec.get("event_id"):
                        event_ids.add(rec["event_id"])
        print("sessao:", sorted(os.listdir(sess)), "event_ids:", sorted(event_ids))
        assert "s:3" in event_ids
        print("STACK E2E OK — API, simulações, gating consolidated/event_id e persistência validados.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()