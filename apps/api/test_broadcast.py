"""Regressão: broadcast nunca segura o loop com cliente lento (head-of-line).

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


def test_broadcast_never_blocks_handler():
    slow, fast = SlowClient(), FastClient()
    api.clients = {slow, fast}
    api.BROADCAST_QUEUE = asyncio.Queue(maxsize=20000)

    async def scenario():
        worker = asyncio.create_task(api._broadcast_worker())
        try:
            t0 = time.monotonic()
            api.broadcast({"type": "ping", "seq": 1})
            enqueue_s = time.monotonic() - t0
            # broadcast() é só agendamento O(1): nunca bloqueia o handler
            assert enqueue_s < 0.05, f"broadcast levou {enqueue_s:.2f}s"

            while time.monotonic() - t0 < 3.0 and not fast.received:
                await asyncio.sleep(0.01)
            elapsed = time.monotonic() - t0
            assert fast.received, "cliente saudável não recebeu o evento"
            payload = json.loads(fast.received[0])
            assert payload == {"type": "ping", "seq": 1}, payload
            assert elapsed < 2.0, f"envio demorou {elapsed:.1f}s (cliente lento segurou)"
            assert slow not in api.clients, "cliente lento deveria ser descartado"
            assert fast in api.clients, "cliente saudável não pode ser descartado"
            assert slow.closed, "cliente lento deveria ser desconectado"
            print(f"ok: broadcast O(1) ({enqueue_s*1000:.0f}ms), saudável recebeu em {elapsed:.2f}s, lento descartado+fechado")
        finally:
            worker.cancel()
            with _suppress(asyncio.CancelledError):
                await worker

    asyncio.run(scenario())


def test_broadcast_no_clients_noop():
    api.clients = set()
    api.BROADCAST_QUEUE = asyncio.Queue(maxsize=20000)

    async def scenario():
        worker = asyncio.create_task(api._broadcast_worker())
        try:
            api.broadcast({"type": "ping"})  # não deve levantar
            await asyncio.sleep(0.05)
        finally:
            worker.cancel()
            with _suppress(asyncio.CancelledError):
                await worker

    asyncio.run(scenario())
    print("ok broadcast sem clientes é noop")


def _suppress(*excs):
    from contextlib import suppress
    return suppress(*excs)


def main():
    print("== apps/api/test_broadcast.py ==")
    for fn in (test_broadcast_never_blocks_handler, test_broadcast_no_clients_noop):
        fn()
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()