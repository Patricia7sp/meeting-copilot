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
clients: set["WebSocket"] = set()
_seen_ids: set = set()            # dedup server-side por event_id (idempotência)
_SEEN_MAX = 4000


def _note_seen(msg_id) -> bool:
    """Retorna True se o id já foi processado (dedup de retransmissão)."""
    if msg_id is None:
        return False
    if msg_id in _seen_ids:
        return True
    _seen_ids.add(msg_id)
    if len(_seen_ids) > _SEEN_MAX:
        _seen_ids.clear()
    return False


def persist(speaker: str, text: str, out: dict, lang: str = "unknown",
            confidence=None, low_confidence: bool = False, translation=None,
            msg_id=None, event_id=None, stage: str = "final",
            consolidated: bool = True):
    _jsonl.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                             "id": msg_id, "event_id": event_id, "stage": stage,
                             "consolidated": consolidated,
                             "speaker": speaker, "text": text, "language": lang,
                             "confidence": confidence, "low_confidence": low_confidence,
                             "translation": translation, "out": out},
                            ensure_ascii=False, default=str) + "\n")
    _jsonl.flush()


async def ack(websocket: "WebSocket", msg_id) -> None:
    if msg_id is None:
        return
    try:
        await websocket.send_text(json.dumps({"type": "ack", "id": msg_id}))
    except Exception:
        pass


async def broadcast(payload: dict) -> None:
    """Entrega eventos tanto ao STT quanto a todas as UIs conectadas."""
    message = json.dumps(payload, ensure_ascii=False)
    disconnected: list[WebSocket] = []
    for client in tuple(clients):
        try:
            await client.send_text(message)
        except (RuntimeError, WebSocketDisconnect):
            disconnected.append(client)
    for client in disconnected:
        clients.discard(client)

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
        lang = payload.get("language", payload.get("lang", "unknown"))
        confidence = payload.get("confidence")
        low = bool(payload.get("low_confidence", False))
        translation = payload.get("translation")
        stage = payload.get("stage", "final")
        consolidated = bool(payload.get("consolidated", True))
        msg_id = payload.get("id")
        event_id = payload.get("event_id")
        # insights só quando a fala está consolidada (final) e não repete id
        if stage == "final" and consolidated and not _note_seen(event_id or msg_id):
            out = orch.handle(speaker, text, lang, confidence=confidence, low_confidence=low)
        else:
            reason = "provisional" if stage != "final" else (
                "not-consolidated" if not consolidated else "id-duplicated")
            out = {"type": "noop", "reason": reason}
        persist(speaker, text, out or {}, lang=lang, confidence=confidence,
                low_confidence=low, translation=translation, msg_id=msg_id,
                event_id=event_id, stage=stage, consolidated=consolidated)
        return {"insight": out, "context": orch.ctx.window_text(6)}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        clients.add(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    msg = {"text": raw}
                msg_id = msg.get("id")
                await ack(websocket, msg_id)
                if msg.get("type") == "metrics":
                    await broadcast(msg)
                    continue
                speaker = msg.get("speaker", "OTHERS")
                text = msg.get("text", "")
                stage = msg.get("stage", "final")
                consolidated = bool(msg.get("consolidated", True))
                lang = msg.get("language", msg.get("lang", "unknown"))
                confidence = msg.get("confidence")
                low = bool(msg.get("low_confidence", False))
                translation = msg.get("translation")
                dedup_key = msg.get("event_id") or msg_id
                if stage != "final" or not consolidated or _note_seen(dedup_key):
                    reason = ("provisional" if stage != "final"
                              else ("not-consolidated" if not consolidated else "id-duplicated"))
                    out = {"type": "noop", "reason": reason}
                else:
                    out = orch.handle(speaker, text, lang,
                                      confidence=confidence, low_confidence=low)
                persist(speaker, text, out or {}, lang=lang, confidence=confidence,
                        low_confidence=low, translation=translation,
                        msg_id=msg_id, event_id=msg.get("event_id"),
                        stage=stage, consolidated=consolidated)
                await broadcast(
                    {"transcript": {"id": msg_id, "event_id": msg.get("event_id"),
                                    "utterance_id": msg.get("utterance_id"),
                                    "stage": stage, "consolidated": consolidated,
                                    "speaker": speaker, "text": text, "language": lang,
                                    "confidence": confidence, "low_confidence": low,
                                    "translation": translation,
                                    "provisional_ids": msg.get("provisional_ids", [])},
                     "insight": out,
                     "rec": True}
                )
        except WebSocketDisconnect:
            clients.discard(websocket)

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
