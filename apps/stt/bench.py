"""Benchmark de modelos faster-whisper para escolha de equilíbrio (latência x qualidade).

Roda no Mac Intel (ou Linux). Uso:
  python3 apps/stt/bench.py --source mic --seconds 4 --models small,medium,large-v3-turbo
  python3 apps/stt/bench.py --source /caminho/audio.wav

Medianas de 3 passadas por modelo; imprime RTF (Real-Time Factor) e qualidade bruta
(texto). Decisão documentada em apps/stt/MODEL.md.
"""
from __future__ import annotations
import argparse
import time
import json


def acquire(source: str, seconds: int) -> bytes:
    """Retorna PCM 16kHz mono int16 (bytes). source=mic grava; senão lê wav."""
    if source == "mic":
        import sounddevice as sd  # type: ignore
        import numpy as np  # type: ignore
        frames = int(16000 * seconds)
        audio = sd.rec(frames, samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        return bytes(np.asarray(audio).tobytes())
    import wave
    with wave.open(source, "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        data = w.readframes(w.getnframes())
    import numpy as np  # type: ignore
    arr = np.frombuffer(data, dtype=np.int16).reshape(-1, ch).mean(axis=1).astype(np.int16)
    if rate != 16000:
        raise SystemExit(f"esperado 16kHz; wav tem {rate}Hz. Reencode antes do bench.")
    return arr.tobytes()


def run_model(model_name: str, pcm: bytes, passes: int = 2) -> dict:
    from faster_whisper import WhisperModel  # type: ignore
    import numpy as np  # type: ignore
    import statistics
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    t_load = time.perf_counter()
    m = WhisperModel(model_name, device="cpu", compute_type="int8")
    load_s = time.perf_counter() - t_load
    lat = []
    text = lang = None
    for _ in range(passes):
        t0 = time.perf_counter()
        segs, info = m.transcribe(audio, language=None, task="transcribe",
                                  condition_on_previous_text=False,
                                  no_speech_threshold=0.6, log_prob_threshold=-0.8,
                                  )
        text = " ".join(s.text.strip() for s in segs).strip()
        lang = info.language
        lat.append(time.perf_counter() - t0)
    n = len(audio) / 16000.0
    med = statistics.median(lat)
    return {"model": model_name, "lat_med_s": round(med, 2), "rtf": round(med / n, 2),
            "load_s": round(load_s, 1), "lang": lang, "text": text, "passes": passes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mic", help="mic ou caminho de wav 16k mono")
    ap.add_argument("--seconds", type=int, default=4)
    ap.add_argument("--models", default="small,medium,large-v3-turbo")
    ap.add_argument("--infer-passes", type=int, default=2)
    args = ap.parse_args()

    pcm = acquire(args.source, args.seconds)
    print(f"[bench] fonte={args.source} {args.seconds}s = {len(pcm)} bytes PCM 16k")
    results = []
    for name in (m.strip() for m in args.models.split(",") if m.strip()):
        try:
            r = run_model(name, pcm, passes=args.infer_passes)
        except Exception as e:
            r = {"model": name, "error": f"{e.__class__.__name__}: {e}"}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, indent=1))
    print("\n[bench] resumo (menor latência x qualidade percebida):")
    best = min((r for r in results if "error" not in r), key=lambda r: r["rtf"], default=None)
    print(json.dumps(best, ensure_ascii=False) if best else "nenhum modelo rodou")


if __name__ == "__main__":
    main()