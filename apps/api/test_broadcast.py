"""Regressão: broadcast() não trava com cliente lento (head-of-line) — fix de produção.

Uso: python3 apps/api/test_broadcast.py
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import time

os.environ["SEND_TIMEOUT"] = "0.3"          # acelerar o teste
os.environ["DATA_DIR"] = "/tmp/mc_broadcast_test"
os.environ["ORCHESTRATOR_MODE"] = "auto"
os.makedirs(os.environ["DATA_DIR"], exist_ok=True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import main as api  # noqa: E402


class SlowClient:
    """Cliente que nunca drena o send — send_text fica pendente (simula buffer cheio)."""
    def __init__(self):
        self.closed = False
        self.received = []

    async def send_text(self, message: str):
        await asyncio.sleep(60)

    async def close(self):
        self.closed = True


class FastClient:
    def __init__(self):
        self.received: list[str] = []

    async def send_text(self, message: str):
        self.received.append(message)

    async def close(self):
        pass


def test_broadcast_slow_client_does_not_block():
    slow, fast = SlowClient(), FastClient()
    api.clients = {slow, fast}
    t0 = time.monotonic()
    asyncio.run(api.broadcast({"type": "ping", "seq": 1}))
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"broadcast travou {elapsed:.1f}s (cliente lento segurou o loop)"
    assert len(fast.received) == 1, fast.received
    payload = json.loads(fast.received[0])
    assert payload["type"] == "ping" and payload["seq"] == 1, payload
    assert slow not in api.clients, "cliente lento deveria ter sido descartado"
    assert fast in api.clients, "cliente saudável não pode ser descartado"
    assert slow.closed, "cliente lento deveria ser desconectado"
    print(f"ok broadcast não bloqueou ({elapsed:.2f}s), saudável recebeu, lento descartado+fechado")


def test_broadcast_no_clients_noop():
    api.clients = set()
    asyncio.run(api.broadcast({"type": "ping"}))  # não deve levantar
    print("ok broadcast sem clientes é noop")


def main():
    print("== apps/api/test_broadcast.py ==")
    for fn in (test_broadcast_slow_client_does_not_block, test_broadcast_no_clients_noop):
        fn()
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()