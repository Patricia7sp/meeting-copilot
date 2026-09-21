"""Camada de áudio unificada Linux + macOS (stdlib + sounddevice opcional).

Filosofia: mesmo código, só muda a *fonte* do áudio do sistema.
- Linux (trabalho): monitor PipeWire/Pulse — já vem no sistema, zero driver.
- macOS (pessoal): BlackHole 2ch + Multi-Output Device — instala 1x via brew.
- Mic: igual nos dois (dispositivo default).
- Mixer: soma loopback (OTHERS) + mic (YOU) em 16kHz mono, chunks de ~1s p/ VAD/STT.

Nada aqui exige áudio real para importar/testar: tudo degrada para mock.
"""
from __future__ import annotations
import platform
import shutil
import subprocess
from dataclasses import dataclass


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 1.0


@dataclass
class AudioSetup:
    os_name: str          # Linux | Darwin
    loopback_kind: str    # pipewire-monitor | blackhole | unknown
    loopback_available: bool
    mic_available: bool
    hint: str


def detect_os() -> str:
    return platform.system()  # "Linux" | "Darwin" | "Windows"


def _cmd_ok(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, capture_output=True, timeout=5, check=False)
        return True
    except Exception:
        return False


def check_linux_monitor() -> tuple[bool, str]:
    """Detecta PipeWire/Pulse monitor sem precisar de lib de áudio."""
    if shutil.which("pactl"):
        try:
            out = subprocess.run(["pactl", "list", "short", "sources"],
                                 capture_output=True, text=True, timeout=5).stdout
            monitors = [l for l in out.splitlines() if ".monitor" in l]
            if monitors:
                return True, f"monitor encontrado: {monitors[0].split()[1]}"
            return False, "pactl ok mas nenhum source .monitor listado (PipeWire pode estar sem monitor exposto)"
        except Exception as e:
            return False, f"pactl falhou: {e}"
    if shutil.which("pw-cli"):
        return True, "pw-cli presente (PipeWire) — monitor provavelmente disponível"
    return False, "nem pactl nem pw-cli encontrados — instale pipewire-pulse ou pulseaudio-utils"


def check_mac_blackhole() -> tuple[bool, str]:
    """BlackHole aparece como dispositivo CoreAudio com 'BlackHole' no nome."""
    try:
        out = subprocess.run(["system_profiler", "SPAudioDataType"],
                             capture_output=True, text=True, timeout=10).stdout
        if "BlackHole" in out:
            return True, "BlackHole detectado no sistema"
    except Exception:
        pass
    # fallback: checa via brew
    if shutil.which("brew"):
        try:
            out = subprocess.run(["brew", "list", "--cask"], capture_output=True,
                                 text=True, timeout=10).stdout
            if "blackhole" in out.lower():
                return True, "BlackHole instalado via brew (confirme no Audio MIDI Setup)"
        except Exception:
            pass
    return False, ("BlackHole não detectado. Instale com: brew install --cask blackhole-2ch; "
                   "depois crie Multi-Output Device (BlackHole + alto-falantes) no Audio MIDI Setup")


def detect_setup() -> AudioSetup:
    os_name = detect_os()
    mic = False
    try:
        import sounddevice as sd  # type: ignore
        devs = sd.query_devices()
        mic = any(d.get("max_input_channels", 0) > 0 for d in devs)
    except Exception:
        mic = False  # sem sounddevice não dá p/ afirmar; check_audio.py avisa

    if os_name == "Linux":
        ok, hint = check_linux_monitor()
        return AudioSetup("Linux", "pipewire-monitor", ok, mic, hint)
    if os_name == "Darwin":
        ok, hint = check_mac_blackhole()
        return AudioSetup("Darwin", "blackhole", ok, mic, hint)
    return AudioSetup(os_name, "unknown", False, mic,
                      f"SO {os_name} ainda não coberto na Entrega 2 (foco Linux+Mac)")


def print_report() -> AudioSetup:
    s = detect_setup()
    print(f"[audio] OS={s.os_name} loopback={s.loopback_kind} "
          f"loopback_ok={s.loopback_available} mic={s.mic_available}")
    print(f"[audio] {s.hint}")
    if s.os_name == "Linux" and not s.loopback_available:
        print("[audio] dica Linux: rode `pactl list short sources | grep monitor` durante uma call "
              "p/ ver o monitor ativo.")
    if s.os_name == "Darwin" and not s.loopback_available:
        print("[audio] dica Mac: sem BlackHole o app captura SÓ seu mic (modo aula 1:1 ainda útil).")
    return s


class Mixer:
    """Soma loopback + mic em blocos 16kHz mono.

    Entrega 2: implementado em cima de `sounddevice` com callback.
    Sem sounddevice instalado: opera em modo mock (gera silêncio) p/ não quebrar CI/testes.
    """

    def __init__(self, loopback_device=None, mic_device=None,
                 samplerate: int = SAMPLE_RATE):
        self.loopback_device = loopback_device
        self.mic_device = mic_device
        self.samplerate = samplerate

    def read_chunk(self, seconds: float = CHUNK_SECONDS):
        """Retorna (pcm_bytes_16k_mono, meta). Mock = silêncio se sem backend."""
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np  # type: ignore
        except ImportError:
            n = int(self.samplerate * seconds)
            import array
            return array.array("h", [0] * n).tobytes(), {"mock": True}

        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore
        frames = int(self.samplerate * seconds)
        # Captura mic + loopback separadamente e soma; se um falhar, usa o outro.
        try:
            mic = sd.rec(frames, samplerate=self.samplerate, channels=1,
                         dtype="int16", device=self.mic_device)
        except Exception:
            mic = None
        try:
            lb = sd.rec(frames, samplerate=self.samplerate, channels=1,
                        dtype="int16", device=self.loopback_device)
        except Exception:
            lb = None
        import time
        time.sleep(seconds)
        parts = [a for a in (mic, lb) if a is not None]
        if not parts:
            return (np.zeros((frames, 1), dtype=np.int16).tobytes(), {"mock": True})
        mixed = parts[0].astype(np.int32)
        for p in parts[1:]:
            mixed = mixed + p.astype(np.int32)
        mixed = (mixed // len(parts)).astype(np.int16)
        you_only = mic is not None and lb is None
        return mixed.tobytes(), {"mock": False, "you_only": you_only}
