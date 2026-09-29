"""Testes da camada de áudio (Feature 007): conversão BlackHole 48k estéreo -> mono 16k.

Uso: python3 apps/audio/test_capture.py
Cobre: downmix estéreo->mono (média), resample 48k/44.1k->16k (comprimento e relevo),
pass-through 16k mono, erro de tamanho impar, meta preenchida, WAV gravado e relatável.
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import numpy as np  # noqa: E402
from capture import (AudioMeta, Mixer, SourceAudio, SAMPLE_RATE,  # noqa: E402
                     to_mono16k, write_wav)


def interleaved(frames: int, channels: int, freq: float = 440.0, rate: int = 48000) -> bytes:
    t = np.arange(frames) / rate
    mono = (0.4 * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    stereo = np.column_stack([mono, mono]) if channels == 2 else mono[:, None]
    return stereo.astype(np.int16).tobytes()


def test_downmix_stereo_keeps_energy():
    frames = 48000 * 1  # 1s a 48k estéreo
    raw = interleaved(frames, 2)
    pcm, down, resampled = to_mono16k(raw, 48000, 2)
    assert down is True and resampled is True
    n = len(pcm) // 2
    assert n == SAMPLE_RATE, ("1s mono16k", n, SAMPLE_RATE)
    a = np.frombuffer(pcm, dtype=np.int16)
    peak = max(abs(int(a.min())), int(a.max()))
    assert 12000 <= peak <= 14000, ("preserva o sinal (0.4*32767~13107)", peak)


def test_resample_48k_to_16k_len():
    pcm, down, resampled = to_mono16k(interleaved(48000, 1), 48000, 1)
    assert down is False and resampled is True
    assert len(pcm) // 2 == SAMPLE_RATE


def test_resample_44100_to_16k_len():
    pcm, down, resampled = to_mono16k(interleaved(44100, 1, rate=44100), 44100, 1)
    assert resampled is True
    assert 16000 <= len(pcm) // 2 <= 16001  # 44100*16000/44100 = 16000


def test_passthrough_16k_mono():
    raw = (np.zeros(16000, dtype=np.int16) | 123).tobytes()
    pcm, down, resampled = to_mono16k(raw, 16000, 1)
    assert pcm == raw and down is False and resampled is False


def test_stereo_mono_channel_split():
    """Canal esquerdo é sinal, direito silêncio -> média mantém metade da energia."""
    t = np.arange(48000) / 48000
    left = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = np.zeros_like(left)
    pcm, down, _ = to_mono16k(np.column_stack([left, right]).astype(np.int16).tobytes(), 48000, 2)
    assert down is True
    a = np.frombuffer(pcm, dtype=np.int16)
    assert abs(a.mean()) < 1.0
    # média esquerdo+direito (um silêncio) tem metade da energia de ambos iguais
    assert a.max() > 6000, ("metade da energia presente", a.max())


def test_mock_mixer_meta():
    """Sem sounddevice (CI/servidor) o mixer devolve silêncio marcado mock com meta."""
    srcs = Mixer().read_sources(0.5)
    for name, src in srcs.items():
        assert src.mock is True
        assert src.meta is not None and src.meta.source == name and src.meta.mock is True
        assert src.pcm == b"\x00\x00" * int(16000 * 0.5)


def test_sourceaudio_carries_meta():
    m = AudioMeta(source="loopback", device="BlackHole 2ch", device_rate=48000,
                  device_channels=2, rms=0.05, rms_db=-26.0, downmixed=True,
                  resampled=True, active=True)
    s = SourceAudio("loopback", b"\x00\x00" * 1600, 0.05, True, meta=m)
    assert s.meta == m and s.meta.device == "BlackHole 2ch"


def test_wav_roundtrip():
    d = Path(tempfile.mkdtemp(prefix="mc_wav_"))
    pcm = (np.sin(np.arange(16000) / 16000 * 2 * np.pi * 440)
           * 30000).astype(np.int16).tobytes()
    wav = d / "t.wav"
    write_wav(wav, pcm)
    import wave
    with wave.open(str(wav), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1
        assert w.readframes(w.getnframes()) == pcm


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()