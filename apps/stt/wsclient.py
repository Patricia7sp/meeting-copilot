"""WebSocket persistente do STT — entrega confiável (zero perda por ack_timeout).

Garantias:
- **Backpressure**: envio é `await queue.put` — fila cheia NUNCA descarta evento.
- **Idempotência por `event_id`** (`run_id` + sequencial): o servidor deduplica
  retransmissões seguramente.
- **Retries com backoff**: evento não-ack mantém-se em `_pending` e é reenviado
  periodicamente (attempts crescentes, backoff limitado).
- **Remoção SÓ após ack**: `_pending` só sai quando o servidor confirma; se não
  confirmar, fica em disco e volta após reconexão/reinício.
- **Persistência local**: fila de pendentes gravada em `pending_dir` (janela de
  ~0.5s + flush no stop) — queda do WS ou crash não perde transcrição.
- **Métricas publicáveis**: queue, retries, ack latency p50/p95, drops,
  reconexões (evento `type:"metrics"` + `report()` no log).

Uso:
    ws = WsClient("ws://host:8000/ws", max_queue=200, pending_dir="data/ws_pending")
    await ws.start()
    await ws.send({"speaker": "YOU", "text": "..."})   # atribui id + event_id
    await ws.send(ws.metrics_event())                  # métricas p/ a UI
    ws.report()
    await ws.stop()
"""
from __future__ import annotations
import asyncio
import contextlib
import itertools
import json
import logging
import math
import os
import socket
import time
from pathlib import Path


class WsClient:
    def __init__(self, url: str, max_queue: int = 200, ack_timeout: float = 8.0,
                 connect_timeout: float = 15.0, max_backoff: float = 30.0,
                 retry_backoff: float = 3.0, max_retries: int | None = None,
                 pending_dir: str | None = None, run_id: str | None = None,
                 persist_interval: float = 0.5, initial_backoff: float = 1.0):
        self.url = url
        self.max_queue = max_queue
        self.ack_timeout = ack_timeout
        self.connect_timeout = connect_timeout
        self.max_backoff = max_backoff
        self.retry_backoff = retry_backoff
        # None = ilimitado (meta: zero perda por timeout transitório)
        self.max_retries = max_retries
        self.pending_dir = pending_dir
        self.persist_interval = persist_interval
        self.initial_backoff = initial_backoff

        resume = self._load_resume(run_id)
        self.run_id = resume["run_id"] if resume else (run_id or f"{socket.gethostname()}-{int(time.time())}")
        self._resume_path = Path(pending_dir) / "resume.json" if pending_dir else None
        self._pending_file = Path(pending_dir) / f"pending_{self.run_id}.jsonl" if pending_dir else None

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._pending: dict[str, dict] = {}     # event_id -> {id, payload, ts, attempts, next_retry}
        self._acked: set[str] = set()
        self._seq = itertools.count(resume["seq"] if resume else 1)
        self._sweep_task: asyncio.Task | None = None
        self._run_task: asyncio.Task | None = None
        self._persist_task: asyncio.Task | None = None
        self._persist_dirty = False
        self._persist_event = asyncio.Event()
        self._log = logging.getLogger("stt.ws")
        self.metrics = {"sent": 0, "acked": 0, "retries": 0, "ack_timeout": 0,
                        "overflow_dropped": 0, "drops": 0, "connect_total": 0,
                        "reconnects": 0, "send_errors": 0, "recv_errors": 0,
                        "persisted_loaded": 0, "latencies_ms": []}

        if self._pending_file and self._pending_file.exists():
            self._restore_pending()

    # ---------- resume / persistência ----------

    def _load_resume(self, run_id: str | None) -> dict | None:
        """run_id estável p/ reuso entre reinícios: resumo gravado em pending_dir vence."""
        pd = self.pending_dir
        if pd:
            p = Path(pd) / "resume.json"
            if p.exists():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and data.get("run_id"):
                        return data
                except Exception:
                    pass
        if run_id:
            return {"run_id": run_id, "seq": 1}
        return None

    def _save_resume(self) -> None:
        if not self._resume_path:
            return
        try:
            self._resume_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(self._resume_path, json.dumps(
                {"run_id": self.run_id, "seq": next(self._seq)}))
        except Exception as e:
            self._log.warning("ws resume persist falhou: %s", e)

    def _restore_pending(self) -> None:
        """Recarrega pendentes de disco: voltam à fila e são reenviados (idempotente)."""
        if not self._pending_file:
            return
        try:
            for line in self._pending_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                ev_id = rec.get("event_id")
                if not ev_id or ev_id in self._acked:
                    continue
                self._pending[ev_id] = {
                    "id": rec.get("id"), "payload": rec.get("payload", {}),
                    "ts": 0.0, "attempts": 0, "next_retry": 0.0,
                }
                self.metrics["persisted_loaded"] += 1
        except Exception as e:
            self._log.warning("ws restore pendentes falhou: %s", e)
        if self._pending:
            self._log.info("ws recuperou %d eventos pendentes do disco", len(self._pending))

    def _mark_persist(self) -> None:
        self._persist_dirty = True
        self._persist_event.set()

    async def _persist_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._persist_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            self._persist_event.clear()
            if self._persist_dirty:
                self._persist_dirty = False
                self._write_pending()

    def _write_pending(self) -> None:
        if not self._pending_file:
            return
        try:
            self._pending_file.parent.mkdir(parents=True, exist_ok=True)
            lines = [json.dumps({"event_id": ev_id,
                                 "id": p["id"], "payload": p["payload"]},
                                ensure_ascii=False)
                     for ev_id, p in self._pending.items()]
            _atomic_write(self._pending_file, "\n".join(lines) + ("\n" if lines else ""))
        except Exception as e:
            self._log.warning("ws persist pendentes falhou: %s", e)

    def _flush_persist(self) -> None:
        if self.pending_dir:
            self._write_pending()
            self._save_resume()

    # ---------- vida ----------

    async def start(self) -> None:
        self._run_task = asyncio.create_task(self._run())
        self._sweep_task = asyncio.create_task(self._sweep())
        if self.pending_dir:
            self._persist_task = asyncio.create_task(self._persist_loop())
        self._log.info("ws start %s (fila max=%d run_id=%s)", self.url, self.max_queue, self.run_id)

    async def stop(self) -> None:
        for t in (self._sweep_task, self._persist_task, self._run_task):
            if t:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
        self._flush_persist()

    # ---------- envio ----------

    async def send(self, payload: dict) -> int | None:
        """Enfileira evento com backpressure (nunca descarta). Devolve o id."""
        msg = dict(payload)
        if "id" not in msg:
            msg["id"] = next(self._seq)
        if "event_id" not in msg:
            msg["event_id"] = f"{self.run_id}:{msg['id']}"
        msg["__ts_sent"] = time.monotonic()
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            # backpressure: espera o drain liberar espaço, NUNCA descarta.
            await self._queue.put(msg)
        return msg.get("id")

    # ---------- loop principal ----------

    async def _run(self) -> None:
        backoff = self.initial_backoff
        while True:
            try:
                import websockets  # type: ignore
            except ImportError:
                self._log.warning("sem websockets; eventos ficam na fila/pendentes")
                await asyncio.sleep(5.0)
                continue
            try:
                try:
                    ws = await websockets.connect(self.url, open_timeout=self.connect_timeout)
                except TypeError:  # websockets < 14 usa `timeout=`
                    ws = await websockets.connect(self.url, timeout=self.connect_timeout)
                async with ws:
                    self._log.info("ws conectado")
                    self.metrics["connect_total"] += 1
                    backoff = 1.0
                    sender = asyncio.create_task(self._drain(ws))
                    reader = asyncio.create_task(self._read_acks(ws))
                    try:
                        done, _pending = await asyncio.wait(
                            {sender, reader}, return_when=asyncio.FIRST_COMPLETED)
                        for t in done:
                            if t.cancelled():
                                continue
                            exc = t.exception()
                            if exc is not None:
                                raise exc
                    finally:
                        for t in (sender, reader):
                            if not t.done():
                                t.cancel()
                        for t in (sender, reader):
                            with contextlib.suppress(BaseException):
                                await t  # cancela e recupera exceções (não "never retrieved")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.metrics["recv_errors"] += 1
                self.metrics["reconnects"] += 1
                self._log.warning("ws erro: %s; reconectando em %.1fs", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)

    async def _drain(self, ws) -> None:
        while True:
            msg = await self._queue.get()
            msg.pop("__ts_sent", None)
            ev_id = msg.get("event_id")
            if ev_id and ev_id in self._acked:
                continue  # já confirmado numa retransmissão anterior
            try:
                await ws.send(json.dumps(msg, ensure_ascii=False))
            except Exception as e:
                self.metrics["send_errors"] += 1
                self._log.warning("ws send falhou: %s", e)
                await self._queue.put(msg)  # devolve pra fila (reenvio pós-reconexão)
                raise
            prev = self._pending.get(ev_id, {})
            self._pending[ev_id] = {
                "id": msg.get("id"), "payload": msg,
                "ts": time.monotonic(),
                "attempts": prev.get("attempts", 0) + 1,
                "next_retry": time.monotonic() + self.retry_backoff,
            }
            self.metrics["sent"] += 1
            self._mark_persist()

    async def _read_acks(self, ws) -> None:
        while True:
            raw = await ws.recv()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") != "ack":
                continue
            aid = msg.get("id")
            for ev_id, p in list(self._pending.items()):
                if p.get("id") == aid:
                    self._pending.pop(ev_id)
                    self._acked.add(ev_id)
                    self.metrics["acked"] += 1
                    lat_ms = (time.monotonic() - p["ts"]) * 1000.0
                    lats = self.metrics["latencies_ms"]
                    lats.append(lat_ms)
                    if len(lats) > 500:
                        del lats[: len(lats) - 500]
                    self._mark_persist()
                    break

    async def _sweep(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            for ev_id, p in list(self._pending.items()):
                if ev_id in self._acked:
                    continue
                expired = now - p["ts"] >= self.ack_timeout
                can_retry = now >= p.get("next_retry", 0)
                if not expired or not can_retry:
                    continue
                if self.max_retries is not None and p["attempts"] > self.max_retries:
                    self.metrics["drops"] += 1
                    self._log.warning("ws desistindo id=%s após %d tentativas", p["id"], p["attempts"])
                    self._pending.pop(ev_id)
                    self._mark_persist()
                    continue
                self.metrics["retries"] += 1
                self.metrics["ack_timeout"] += 1
                p["next_retry"] = now + self.retry_backoff
                self._log.warning("ws ack timeout: id=%s reenviando (attempt=%d)",
                                  p["id"], p["attempts"] + 1)
                try:
                    self._queue.put_nowait(dict(p["payload"]))
                except asyncio.QueueFull:
                    pass  # o drain vai pegar quando liberar espaço

    # ---------- métricas ----------

    def _percentile(self, q: float) -> float:
        lats = sorted(self.metrics["latencies_ms"])
        if not lats:
            return 0.0
        i = min(len(lats) - 1, math.ceil(len(lats) * q) - 1)
        return lats[i]

    def metrics_event(self) -> dict:
        return {
            "type": "metrics",
            "run_id": self.run_id,
            "queue": self._queue.qsize(),
            "max_queue": self.max_queue,
            "pending_ack": len(self._pending),
            "sent": self.metrics["sent"],
            "acked": self.metrics["acked"],
            "retries": self.metrics["retries"],
            "ack_timeout": self.metrics["ack_timeout"],
            "drops": self.metrics["drops"],
            "overflow_dropped": self.metrics["overflow_dropped"],
            "reconnects": self.metrics["reconnects"],
            "connect_total": self.metrics["connect_total"],
            "persisted_loaded": self.metrics["persisted_loaded"],
            "ack_p50_ms": round(self._percentile(0.5), 1),
            "ack_p95_ms": round(self._percentile(0.95), 1),
        }

    def report(self) -> str:
        return (f"ws: enviados={self.metrics['sent']} acked={self.metrics['acked']} "
                f"queue={self._queue.qsize()}/{self.max_queue} "
                f"pending_ack={len(self._pending)} "
                f"retries={self.metrics['retries']} ack_timeout={self.metrics['ack_timeout']} "
                f"drops={self.metrics['drops']} overflow={self.metrics['overflow_dropped']} "
                f"persisted={self.metrics['persisted_loaded']} "
                f"reconnects={self.metrics['reconnects']} "
                f"ack_ms p50={self._percentile(0.5):.1f} p95={self._percentile(0.95):.1f}")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)