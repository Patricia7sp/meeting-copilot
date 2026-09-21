"""API hub: recebe transcrição (STT ou simulador) e devolve insights via WS + HTTP."""
from __future__ import annotations
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../orchestrator"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../packages/context"))
from orchestrator import Orchestrator

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

MODE = os.getenv("ORCHESTRATOR_MODE", "auto")
COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "4"))  # menor no MVP p/ demo
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parents[2] / "data" / "sessions"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

orch = Orchestrator(mode=MODE if MODE in ("auto", "english", "work") else "auto",
                    cooldown_seconds=COOLDOWN)

SESSION_ID = datetime.now().strftime("%Y-%m-%d_%H%M")
_jsonl = open(DATA_DIR / f"{SESSION_ID}.jsonl", "a", encoding="utf-8")

def persist(speaker: str, text: str, out: dict):
    _jsonl.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                             "speaker": speaker, "text": text, "out": out},
                            ensure_ascii=False) + "\n")
    _jsonl.flush()

if HAS_FASTAPI:
    app = FastAPI(title="Meeting Copilot MVP")

    @app.get("/health")
    def health():
        return {"ok": True, "mode": orch.mode, "session": SESSION_ID}

    @app.post("/ingest")
    def ingest(payload: dict):
        speaker = payload.get("speaker", "OTHERS")
        text = payload.get("text", "")
        out = orch.handle(speaker, text, payload.get("lang", "unknown"))
        persist(speaker, text, out or {})
        return {"insight": out, "context": orch.ctx.window_text(6)}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    msg = {"text": raw}
                speaker = msg.get("speaker", "OTHERS")
                text = msg.get("text", "")
                out = orch.handle(speaker, text, msg.get("lang", "unknown"))
                persist(speaker, text, out or {})
                await websocket.send_text(json.dumps(
                    {"transcript": {"speaker": speaker, "text": text},
                     "insight": out,
                     "rec": True}, ensure_ascii=False))
        except WebSocketDisconnect:
            pass

    @app.get("/ui")
    def ui():
        html = Path(__file__).resolve().parent.parent / "ui" / "index.html"
        return HTMLResponse(html.read_text(encoding="utf-8"))
else:
    app = None
