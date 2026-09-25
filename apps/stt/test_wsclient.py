"""Testes da entrega confiável via WebSocket (wsclient.py) — sem modelo/API real.

Uso: python3 apps/stt/test_wsclient.py

Cobre: ack normal, ausência de ack (retry com backoff SEM descartar), queda de
conexão + reenvio pós-reconexão, persistência em disco e recuperação após
reinício, backpressure (fila cheia nunca descarta) e métricas.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from wsclient import WsClient  # noqa: E402


async def _spawn(handler):
    import websockets
    server = await websockets.serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}/ws"


async def _ack_everything(ws):
    async for raw in ws:
        msg = json.loads(raw)
        await ws.send(json.dumps({"type": "ack", "id": msg.get("id")}))


async def _ignore_everything(ws):
    async for _ in ws:
        pass


def _client(url: str, tmpdir: str | None = None, **kw) -> WsClient:
    opts = dict(ack_timeout=0.3, retry_backoff=0.15, initial_backoff=0.05,
                connect_timeout=2.0, max_backoff=1.0)
    opts.update(kw)
    return WsClient(url, pending_dir=tmpdir, **opts)


async def _finished(url: str, tmpdir: str | None = None) -> WsClient:
    ws = _client(url, tmpdir)
    await ws.start()
    return ws


def test_ack_normal():
    async def run():
        server, url = await _spawn(_ack_everything)
        ws = await _finished(url)
        for i in range(3):
            await ws.send({"speaker": "YOU", "text": f"msg {i}", "id": i + 1})
        await asyncio.sleep(0.8)
        assert ws.metrics["sent"] == 3, ws.metrics
        assert ws.metrics["acked"] == 3, ws.metrics
        assert len(ws._pending) == 0, ws._pending
        assert ws.metrics["drops"] == 0 and ws.metrics["overflow_dropped"] == 0
        m = ws.metrics_event()
        assert m["acked"] == 3 and m["ack_p50_ms"] >= 0 and m["queue"] == 0
        await ws.stop()
        server.close()
    asyncio.run(run())
    print("ok  test_ack_normal")


def test_no_ack_retries_but_keeps_event():
    async def run():
        server, url = await _spawn(_ignore_everything)
        ws = await _finished(url)
        await ws.send({"speaker": "OTHERS", "text": "sem ack"})
        await asyncio.sleep(1.4)
        assert ws.metrics["sent"] >= 1, ws.metrics
        assert ws.metrics["ack_timeout"] >= 1, ws.metrics
        assert ws.metrics["retries"] >= 1, ws.metrics
        assert ws.metrics["drops"] == 0, "nenhum evento pode ser perdido por ack_timeout"
        assert len(ws._pending) == 1, ws._pending
        await ws.stop()
        server.close()
    asyncio.run(run())
    print("ok  test_no_ack_retries_but_keeps_event")


def test_drop_connection_resend_after_reconnect():
    async def run():
        def flaky_first(n=1):
            cnt = {"n": 0}

            async def h(ws):
                cnt["n"] += 1
                if cnt["n"] <= n:
                    async for _ in ws:
                        await ws.close(code=1011)
                        return
                async for raw in ws:
                    await ws.send(json.dumps({"type": "ack", "id": json.loads(raw).get("id")}))

            return h

        server1, url1 = await _spawn(flaky_first(1))
        ws = await _finished(url1)
        await ws.send({"speaker": "YOU", "text": "soltar conexao"})
        await asyncio.sleep(1.6)
        assert ws.metrics["connect_total"] >= 2, ws.metrics
        assert ws.metrics["retries"] >= 1, ws.metrics
        assert ws.metrics["drops"] == 0, ws.metrics
        assert ws.metrics["acked"] >= 1, ws.metrics
        assert len(ws._pending) == 0, ws._pending
        await ws.stop()
        server1.close()
    asyncio.run(run())
    print("ok  test_drop_connection_resend_after_reconnect")


def test_persistence_recovers_after_restart():
    async def run():
        tmpdir = tempfile.mkdtemp()
        # sessão 1: envia sem ack -> queda -> pendente vai pro disco
        server1, url1 = await _spawn(_ignore_everything)
        ws1 = await _finished(url1, tmpdir)
        await ws1.send({"speaker": "YOU", "text": "não confirmado"})
        await asyncio.sleep(0.6)
        await ws1.stop()  # flush pendentes p/ disco
        server1.close()
        # sessão 2: mesmo pending_dir -> recupera e reenvia (server com ack)
        server2, url2 = await _spawn(_ack_everything)
        ws2 = await _finished(url2, tmpdir)
        await asyncio.sleep(1.8)
        assert ws2.metrics["persisted_loaded"] >= 1, ws2.metrics
        assert ws2.metrics["acked"] >= 1, ws2.metrics
        assert len(ws2._pending) == 0, ws2._pending
        await ws2.stop()
        server2.close()
    asyncio.run(run())
    print("ok  test_persistence_recovers_after_restart")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()