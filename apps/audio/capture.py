"""Camada de áudio unificada Linux + macOS (stdlib + sounddevice opcional).

Filosofia: mesmo código, só muda a *fonte* do áudio do sistema.
- Linux (trabalho): monitor PipeWire/Pulse — já vem no sistema, zero driver.
- macOS (pessoal): BlackHole 2ch + Multi-Output Device — instala 1x via brew.
- Mic: igual nos dois (dispositivo default).
- Mixer: soma loopback (OTHERS) + mic (YOU) em 16kHz mono, chunks de ~1s p/ VAD/STT.

Nada aqui exige áudio real para importar/testar: tudo degrada para mock.
"""
from __future__ import annotations
import math
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 1.0

# Locutor é decidido por energia (RMS), nunca por "dispositivo abriu".
DEFAULT_RMS_THRESHOLD = 0.004  # linear em escala 0..1 (int16 full scale)

# Ruído digital absoluto: abaixo disto é cauda digital (silêncio puro), não ruído de sala.
DIGITAL_FLOOR_DB = -70.0


@dataclass
class AudioSetup:
    os_name: str          # Linux | Darwin
    loopback_kind: str    # pipewire-monitor | blackhole | unknown
    loopback_available: bool
    mic_available: bool
    hint: str


@dataclass
class SourceAudio:
    """Um pedaço de áudio de uma única fonte (mic ou loopback), 16kHz mono int16."""
    source: str      # "mic" | "loopback"
    pcm: bytes       # int16 mono
    rms: float       # energia linear 0..1 (RMS sobre int16/full-scale)
    active: bool     # device abriu com sucesso nesta leitura (informação, não decisão)
    mock: bool = False  # sem backend de áudio (CI/teste): pcm é silêncio


def rms_of(pcm_or_arr) -> float:
    """RMS linear (0..1) do áudio 16kHz mono int16. Aceita bytes ou ndarray int16."""
    import math
    try:
        import numpy as np  # type: ignore
    except ImportError:
        return 0.0
    if isinstance(pcm_or_arr, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(pcm_or_arr, dtype=np.int16)
    else:
        arr = np.asarray(pcm_or_arr, dtype=np.int16).ravel()
    if arr.size == 0:
        return 0.0
    x = arr.astype(np.float32) / 32768.0
    return float(math.sqrt(float(np.mean(x * x))))


def rms_db(rms: float) -> float:
    """RMS em dBFS (silêncio perto de -inf, fala ~ -30..-6 dB)."""
    import math
    return 20.0 * math.log10(max(rms, 1e-9))


class NoiseFloorCalibrator:
    """Piso de ruído por fonte (calibração contínua).

    Mantém uma estimativa exponencial do RMS de "fundo" de UMA fonte. O piso só
    é aprendido com frames claramente abaixo do limiar de fala (não-speech), e a
    calibração é separada por fonte: o ruído ambiente do mic não afeta o loopback
    e vice-versa. Todo o resto da decisão (delta, histerese) fica no gate do STT.
    """

    def __init__(self, initial_floor_db: float = -46.0, alpha: float = 0.10,
                 min_learn_delta_db: float = 4.0):
        self.floor_db = initial_floor_db
        self.alpha = alpha
        self.min_learn_delta_db = min_learn_delta_db
        self.frames = 0

    def as_db(self, rms: float) -> float:
        return 20.0 * math.log10(max(rms, 1e-9))

    def observe(self, rms: float, *, is_speech: bool) -> float:
        """Alimenta com o RMS de um frame. is_speech=True não calibra (fala sobe o piso).

        Retorna o piso atual (dB). Frames de puro silêncio digital levam o piso de
        volta ao valor inicial (não ficam presos em -∞).
        """
        self.frames += 1
        if is_speech:
            return self.floor_db
        db = self.as_db(rms)
        if db <= DIGITAL_FLOOR_DB:
            return self.floor_db
        if self.floor_db - db >= self.min_learn_delta_db:
            return self.floor_db
        self.floor_db = self.alpha * db + (1.0 - self.alpha) * self.floor_db
        return self.floor_db


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

    def read_sources(self, seconds: float = CHUNK_SECONDS) -> dict[str, SourceAudio]:
        """Captura mic + loopback como fontes SEPARADAS (mesmo instante, ~mesma duração).

        Retorna {"mic": SourceAudio, "loopback": SourceAudio} com PCM, RMS e flag de
        dispositivo ativo. Nenhuma decisão de locutor é tomada aqui — quem decide é o
        STT, via energia/rms por fonte.
        Sem backend de áudio: ambas as fontes voltam como silêncio (mock) p/ não quebrar CI.
        """
        frames = int(self.samplerate * seconds)
        try:
            import numpy as np          # type: ignore
            import sounddevice as sd    # type: ignore
        except (ImportError, OSError):  # sem backend (ou PortAudio ausente) -> mock silencioso

            def _silent() -> bytes:
                return b"\x00\x00" * frames

            return {
                "mic": SourceAudio("mic", _silent(), 0.0, False, mock=True),
                "loopback": SourceAudio("loopback", _silent(), 0.0, False, mock=True),
            }

        def _silent() -> bytes:
            return np.zeros((frames, 1), dtype=np.int16).tobytes()

        def capture_device(name: str, device: int | str | None) -> SourceAudio:
            try:
                with sd.InputStream(
                    samplerate=self.samplerate,
                    channels=1,
                    dtype="int16",
                    device=device,
                ) as stream:
                    arr, _overflowed = stream.read(frames)
            except (sd.PortAudioError, ValueError) as error:
                print(f"[audio][{name}] falhou ao abrir device={device!r}: {error}")
                return SourceAudio(name, _silent(), 0.0, False)
            return SourceAudio(name, arr.tobytes(), rms_of(arr), True)

        devices = (("mic", self.mic_device), ("loopback", self.loopback_device))
        with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="audio-capture") as pool:
            futures = {
                name: pool.submit(capture_device, name, device)
                for name, device in devices
            }
            return {name: future.result() for name, future in futures.items()}

    def read_chunk(self, seconds: float = CHUNK_SECONDS):
        """Mantido p/ compatibilidade: soma mic + loopback em 16kHz mono.

        Retorna (pcm_bytes_16k_mono, meta) com meta.rms_mic/rms_loopback e
        meta.you_only (apenas diagnóstico; decisão real de locutor fica no STT).
        Mock = silêncio se sem backend.
        """
        srcs = self.read_sources(seconds)
        mic, lb = srcs["mic"], srcs["loopback"]
        import numpy as np  # type: ignore
        a = np.frombuffer(mic.pcm, dtype=np.int16)
        b = np.frombuffer(lb.pcm, dtype=np.int16)
        parts = [p for p, s in ((a, mic), (b, lb)) if s.active]
        if not parts:
            silent = b"\x00\x00" * max(len(a), len(b)) or b""
            return silent, {
                "mock": mic.mock and lb.mock, "you_only": False,
                "rms_mic": mic.rms, "rms_loopback": lb.rms,
            }
        mixed = parts[0].astype(np.int32)
        for p in parts[1:]:
            mixed = mixed + p.astype(np.int32)
        mixed = (mixed // len(parts)).astype(np.int16)
        return mixed.tobytes(), {
            "mock": mic.mock and lb.mock, "you_only": mic.active and not lb.active,
            "rms_mic": mic.rms, "rms_loopback": lb.rms,
        }

    async def stream_frame(self, frame_seconds: float = 0.25, max_iter: int | None = None):
        """Gera quadros contínuos por fonte (mic + loopback) pra segmentação em tempo real.

        Cada yield é uma leitura de `frame_seconds` (default 250ms) das DUAS fontes
        separadas. Quadros pequenos dão resolução fina para medir silêncio real
        (500-800ms) e troca de fonte sem depender de `RMS > limiar`.
        Mock/CI: gera quadros de silêncio (mock=True), sem precisar de áudio real.
        """
        if frame_seconds <= 0:
            frame_seconds = 0.25
        import asyncio
        n = 0
        while True:
            n += 1
            if max_iter and n > max_iter:
                return
            sources = await asyncio.to_thread(self.read_sources, frame_seconds)
            yield sources
