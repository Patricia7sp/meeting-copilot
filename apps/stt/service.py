"""STT service: VAD + faster-whisper. Modo mock por default (Entrega 1).

Real (Entrega 2):
  MOCK_MODE=false python service.py  -> captura PipeWire/WASAPI e envia p/ API_WS_URL
Mock (agora):
  python service.py --mock-once "What did you do during the weekend?"
"""
from __future__ import annotations
import asyncio
import json
import os

API_WS_URL = os.getenv("API_WS_URL", "ws://localhost:8000/ws")
MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
MOCK = os.getenv("MOCK_MODE", "true").lower() == "true"


async def send_mock(text: str, speaker: str = "OTHERS"):
    try:
        import websockets  # type: ignore
    except ImportError:
        print(f"[mock] (sem websockets instalado) {speaker}: {text}")
        print(f"[mock] rode: pip install websockets && python service.py --mock-once \"{text}\"")
        return
    async with websockets.connect(API_WS_URL) as ws:
        await ws.send(json.dumps({"speaker": speaker, "text": text}))
        print(await ws.recv())


def run_realtime():
    """Entrega 2: implementar loop soundcard -> Silero VAD -> faster-whisper."""
    print(f"[stt] modo realtime ainda não ativo nesta entrega (modelo={MODEL}).")
    print("[stt] use scripts/simulate_meeting.py para validar o pipeline.")


if __name__ == "__main__":
    import sys
    if "--mock-once" in sys.argv:
        idx = sys.argv.index("--mock-once")
        text = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "What did you do during the weekend?"
        asyncio.run(send_mock(text))
    elif MOCK:
        run_realtime()
    else:
        run_realtime()
