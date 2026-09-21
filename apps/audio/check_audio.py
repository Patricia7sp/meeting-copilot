"""Diagnóstico de áudio cross-platform. Uso: python3 apps/audio/check_audio.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from capture import print_report, Mixer

if __name__ == "__main__":
    s = print_report()
    if "--test-mixer" in sys.argv:
        pcm, meta = Mixer().read_chunk(0.2)
        print(f"[audio] mixer chunk: {len(pcm)} bytes meta={meta}")
