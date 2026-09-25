"""E2E do pipeline STT->API: ack, IDs, stage provisional/final, dedup e limiar de confiança.

Roda a API em :8011, envia eventos como o STT faria e verifica o comportamento.
Uso: python3 scripts/test_ws_e2e.py
"""
from __future__ import annotations
import asyncio
import json
import os
import subprocess
import sys
import time

os.environ["DATA_DIR"] = "/tmp/mc_e2e_sessions"
os.makedirs(os.environ["DATA_DIR"], exist_ok=True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    apidir = os.path.join(ROOT, "apps", "api")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
         "--port", "8011", "--log-level", "error"],
        cwd=apidir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2.5)
        import websockets
        acks = []

        async def recv2(ws):
            """ack + broadcast chegam como 2 mensagens (ordem: ack primeiro)."""
            ac_lst = []
            for _ in range(2):
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                if m.get("type") == "ack":
                    ac_lst.append(m)
                else:
                    ac = m
            return ac_lst[-1], ac

        async def run():
            async with websockets.connect("ws://127.0.0.1:8011/ws") as ws:
                # 1) provisório: transmite, NÃO gera insight
                await ws.send(json.dumps({"id": 1, "event_id": "run:1", "stage": "provisional",
                                          "speaker": "YOU", "text": "Ontem eu fui ao",
                                          "language": "pt", "confidence": 0.61,
                                          "low_confidence": False}))
                ack1, b1 = await recv2(ws)
                acks.append(ack1["id"])
                assert ack1["id"] == 1, ack1
                assert b1["transcript"]["stage"] == "provisional" and b1["transcript"]["id"] == 1, b1
                assert b1["insight"]["reason"] == "provisional", b1

                # 2) final consolidado (confidence alta): transmite + insight potencial
                await ws.send(json.dumps({"id": 2, "event_id": "run:2", "stage": "final",
                                          "consolidated": True, "speaker": "YOU",
                                          "text": "Ontem eu fui ao mercado", "language": "pt",
                                          "confidence": 0.93, "low_confidence": False}))
                ack2, b2 = await recv2(ws)
                acks.append(ack2["id"])
                assert ack2["id"] == 2 and b2["transcript"]["stage"] == "final", (ack2, b2)

                # 3) retransmissão do MESMO event_id -> dedup server-side, mas ack de novo
                await ws.send(json.dumps({"id": 2, "event_id": "run:2", "stage": "final",
                                          "consolidated": True, "speaker": "YOU",
                                          "text": "Ontem eu fui ao mercado", "language": "pt",
                                          "confidence": 0.93, "low_confidence": False}))
                ack3, b3 = await recv2(ws)
                acks.append(ack3["id"])
                assert ack3["id"] == 2 and b3["insight"]["reason"] == "id-duplicated", (ack3, b3)

                # 4) final NÃO consolidado -> noop (provisório de verdade não gera insight)
                await ws.send(json.dumps({"id": 5, "event_id": "run:5", "stage": "final",
                                          "consolidated": False, "speaker": "YOU",
                                          "text": "trecho parcial", "language": "pt",
                                          "confidence": 0.9, "low_confidence": False}))
                ack5a, b5a = await recv2(ws)
                acks.append(ack5a["id"])
                assert b5a["insight"]["reason"] == "not-consolidated", b5a

                # 5) baixa confiança no final -> noop low_confidence (mesmo sendo insightável)
                await ws.send(json.dumps({"id": 3, "event_id": "run:3", "stage": "final",
                                          "consolidated": True, "speaker": "OTHERS",
                                          "text": "Talvez essa pipeline esteja demorando por join no BigQuery",
                                          "language": "pt", "confidence": 0.31,
                                          "low_confidence": True}))
                ack4, b4 = await recv2(ws)
                acks.append(ack4["id"])
                assert ack4["id"] == 3 and b4["insight"]["reason"] == "low_confidence", (ack4, b4)

                # 6) final de boa confiança gera insight de trabalho
                await ws.send(json.dumps({"id": 4, "event_id": "run:4", "stage": "final",
                                          "consolidated": True, "speaker": "OTHERS",
                                          "text": "Talvez essa pipeline esteja demorando por join no BigQuery",
                                          "language": "pt", "confidence": 0.9,
                                          "low_confidence": False}))
                ack5, b5 = await recv2(ws)
                acks.append(ack5["id"])
                assert b5["insight"]["type"] == "insight" and b5["insight"]["kind"] == "work_copilot", b5

                # 7) métricas do STT -> broadcast sem ack (mensagem espelho)
                await ws.send(json.dumps({"type": "metrics", "acked": 7, "retries": 0,
                                          "drops": 0, "queue": 0}))
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                assert m["type"] == "metrics" and m["acked"] == 7, m

        asyncio.run(run())
        assert acks == [1, 2, 2, 5, 3, 4], acks
        print("acks:", acks)
        print("E2E OK — ack por id, event_id dedup, consolidated, limiar e métricas validados.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()