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
    from fastapi.middleware.cors import CORSMiddleware
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

MODE = os.getenv("ORCHESTRATOR_MODE", "auto")
COOLDOWN = int(os.getenv("COOLDOWN_SECONDS", "4"))  # menor no MVP p/ demo
def _default_data_dir() -> Path:
    try:
        return Path(__file__).resolve().parents[2] / "data" / "sessions"  # repo local
    except IndexError:
        return Path("/app/data/sessions")  # dentro do container

DATA_DIR = Path(os.getenv("DATA_DIR", _default_data_dir()))
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
    # Acesso via Tailscale: UI roda no mesmo host/porta, mas libera CORS
    # para o caso do cliente abrir de outra origem (ex. file:// no Tauri).
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|100\.\d+\.\d+\.\d+)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        candidates = [
            Path(__file__).resolve().parent.parent / "ui" / "index.html",  # repo local
            Path(__file__).resolve().parent / "ui_index.html",  # dentro do container
        ]
        for html in candidates:
            if html.exists():
                return HTMLResponse(html.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>UI não encontrada</h1>", status_code=404)
else:
    app = None
