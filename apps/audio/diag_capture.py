"""Diagnóstico de áudio — grava 30–60s do source em WAV para benchmark (Feature 007).

USO EXCLUSIVO DE VALIDAÇÃO/benchmark (o WAV fica LOCAL; nunca é enviado à API):
  python3 apps/audio/diag_capture.py --seconds 60 --source both --ref transcripts/aula.txt
  python3 apps/audio/diag_capture.py --seconds 60 --source loopback  # só o BlackHole

O que faz:
1. Captura cada fonte no rate/canais NATIVOS (BlackHole: estéreo 48kHz) e converte
   para o PCM 16kHz mono que o modelo consome (downmix + resample linear).
2. Grava `<out>/loopback_<ts>.wav` e/ou `<out>/mic_<ts>.wav` (16k mono) — a evidência.
3. Grava `<out>/meta_<ts>.json` com a configuração de áudio REAL por fonte
   (device, rate, canais, downmix/resample, duração, RMS/dB por segundo) — o
   `apps/stt/benchmark.py --wav ... --meta ...` consome este arquivo no relatório.

Depois rode o benchmark de modelos com o mesmo WAV:
  python3 apps/stt/benchmark.py --wav data/bench/capture/loopback_*.wav \
      --ref transcripts/aula.txt --models small,medium,large-v3-turbo,distil-large-v3 \
      --save apps/stt/BENCH_REPORT.md
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from capture import AudioMeta, Mixer, rms_db, write_wav  # noqa: E402


def _iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _profile(pcm16: bytes, seconds: int) -> list[float]:
    """RMS/dB por segundo (cada segundo = SAMPLE_RATE*2 bytes de PCM 16k mono)."""
    import numpy as np  # type: ignore
    a = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    per = 16000
    out: list[float] = []
    for i in range(seconds):
        seg = a[i * per:(i + 1) * per]
        if seg.size == 0:
            out.append(-99.0)
            continue
        rms = float((seg * seg).mean()) ** 0.5
        out.append(round(rms_db(rms), 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=int, default=60, help="duração da gravação (30-60s)")
    ap.add_argument("--source", choices=("loopback", "mic", "both"), default="both")
    ap.add_argument("--out", default="data/bench/capture",
                    help="diretório dos WAV/meta (benchmark only, fica local)")
    ap.add_argument("--ref", default=None,
                    help="(opcional) caminho do .txt com a transcrição oficial do trecho")
    args = ap.parse_args()

    if not 30 <= args.seconds <= 120:
        print(f"duração deve ficar em 30..120s (obtido {args.sources})")
        sys.exit(2)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mixer = Mixer()
    chosen = ("mic", "loopback") if args.source == "both" else (args.source,)

    bufs: dict[str, bytearray] = {name: bytearray() for name in chosen}
    metas: dict[str, AudioMeta] = {}
    rms_db_by_src: dict[str, list[float]] = {name: [] for name in chosen}
    print(f"[diag] gravando {args.seconds}s de {args.source} em {out}/ ...")
    for i in range(args.seconds):
        sources = mixer.read_sources(1.0)
        for name in chosen:
            src = sources[name]
            bufs[name].extend(src.pcm)
            rms_db_by_src[name].append(round(rms_db(src.rms), 1))
            if name not in metas and src.meta:
                metas[name] = src.meta
            if i % 10 == 0:
                m = src.meta
                print(f"  [{name}] t={i:02d}s device={m.device if m else '?'} "
                      f"rate={m.device_rate if m else '?'} ch={m.device_channels if m else '?'} "
                      f"downmix={m.downmixed if m else '?'} resample={m.resampled if m else '?'}")

    report = {
        "generated_at": _iso(),
        "duration_s": args.seconds,
        "note": "áudio de benchmark LOCAL — nunca enviado à API",
        "reference_transcript": str(Path(args.ref).resolve()) if args.ref else None,
        "sources": {},
    }
    for name in chosen:
        pcm16 = bytes(bufs[name])
        wav = out / f"{name}_{ts}.wav"
        write_wav(wav, pcm16)
        meta: AudioMeta = metas.get(name) or AudioMeta(source=name)
        rms_vals = rms_db_by_src[name]
        avg_db = round(sum(rms_vals) / len(rms_vals), 1) if rms_vals else -99.0
        # padroniza field do over-100s profile (só estatísticas, não o array inteiro)
        report["sources"][name] = {
            "wav": str(wav.resolve()),
            "device": meta.device,
            "device_rate": meta.device_rate,
            "device_channels": meta.device_channels,
            "out_rate": meta.out_rate,
            "out_channels": meta.out_channels,
            "chunk_seconds": meta.chunk_seconds,
            "n_samples": meta.n_samples,
            "rms_db_avg": avg_db,
            "rms_db_min": min(rms_vals) if rms_vals else -99.0,
            "rms_db_max": max(rms_vals) if rms_vals else -99.0,
            "dbfs_per_second": rms_vals,  # perfil p/ validar onde houve fala/silêncio
            "downmixed": meta.downmixed,
            "resampled": meta.resampled,
            "mock": meta.mock,
            "active": meta.active,
        }
        print(f"[diag] gravado {wav.resolve()} ({len(pcm16) // 32000}s de áudio, "
              f"rms_avg={avg_db}dB min={report['sources'][name]['rms_db_min']}dB "
              f"max={report['sources'][name]['rms_db_max']}dB)")

    meta_path = out / f"meta_{ts}.json"
    meta_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[diag] meta: {meta_path.resolve()}")
    print("[diag] benchmark: python3 apps/stt/benchmark.py --wav "
          + " ".join(report["sources"][n]["wav"] for n in report["sources"])
          + f" --ref {args.ref or '<transcricao-oficial.txt>'} --models small,medium,large-v3-turbo,distil-large-v3 --save apps/stt/BENCH_REPORT.md")


if __name__ == "__main__":
    main()