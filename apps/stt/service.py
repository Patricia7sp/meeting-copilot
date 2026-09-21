"""STT service: Mixer (Linux PipeWire / Mac BlackHole) -> VAD -> faster-whisper -> API WS.

Uso:
  python3 apps/audio/check_audio.py                 # diagnóstico (sem deps)
  MOCK_MODE=false python3 apps/stt/service.py       # realtime (precisa sounddevice + faster-whisper)
  python3 apps/stt/service.py --mock-once "texto"   # teste sem áudio
"""
from __future__ import annotations
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../audio"))
from capture import Mixer, detect_setup

API_WS_URL = os.getenv("API_WS_URL", "ws://localhost:8000/ws")
MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
MOCK = os.getenv("MOCK_MODE", "true").lower() == "true"
LOOPBACK_DEVICE = os.getenv("LOOPBACK_DEVICE")  # nome ou índice; None = default
MIC_DEVICE = os.getenv("MIC_DEVICE")


async def send_text(text: str, speaker: str = "OTHERS"):
    try:
        import websockets  # type: ignore
    except ImportError:
        print(f"[stt] {speaker}: {text}  (sem websockets; instale p/ enviar à API)")
        return
    async with websockets.connect(API_WS_URL) as ws:
        await ws.send(json.dumps({"speaker": speaker, "text": text}))
        print(await ws.recv())


def transcribe_chunk(pcm: bytes, model=None) -> str:
    """Transcreve 1 chunk 16kHz mono. Sem faster-whisper instalado: retorna '' (skip)."""
    if model is None:
        return ""
    import numpy as np  # type: ignore
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    # energia simples como VAD barato antes de chamar o modelo
    if (abs(audio).mean() < 0.005):
        return ""
    segments, _ = model.transcribe(audio, language=None, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def run_realtime():
    setup = detect_setup()
    print(f"[stt] OS={setup.os_name} loopback={setup.loopback_kind} ok={setup.loopback_available}")
    print(f"[stt] {setup.hint}")
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel(MODEL, device="cpu", compute_type="int8")
        print(f"[stt] faster-whisper {MODEL} carregado.")
    except ImportError:
        model = None
        print("[stt] faster-whisper NÃO instalado — rodando em modo escuta mock "
              "(instale: pip install -r apps/stt/requirements.txt). Chunks serão descartados.")
    mixer = Mixer(loopback_device=LOOPBACK_DEVICE, mic_device=MIC_DEVICE)
    print("[stt] capturando... Ctrl+C para parar.")
    try:
        while True:
            pcm, meta = mixer.read_chunk(1.0)
            if meta.get("mock") and model is None:
                continue
            text = transcribe_chunk(pcm, model)
            if text:
                speaker = "YOU" if meta.get("you_only") else "OTHERS"
                asyncio.run(send_text(text, speaker))
    except KeyboardInterrupt:
        print("\n[stt] parado.")


if __name__ == "__main__":
    if "--mock-once" in sys.argv:
        idx = sys.argv.index("--mock-once")
        text = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else "What did you do during the weekend?"
        asyncio.run(send_text(text))
    elif MOCK:
        print(f"[stt] MOCK_MODE=true (modelo={MODEL}). Defina MOCK_MODE=false p/ realtime.")
        print("[stt] use scripts/simulate_meeting.py para validar o pipeline.")
    else:
        run_realtime()
