"""Benchmark de modelos faster-whisper em PT-BR e inglês.

Métricas por modelo: WER, COBERTURA (fração das palavras de referência capturadas),
acerto de IDIOMA, latência (p50/p95) e uso de CPU (ratio cpu/wall).

Modo speak (você grava e digita a referência):
  python3 apps/stt/benchmark.py --speak --seconds 5 --langs pt,en

Modo run (usar as gravações salvas em data/bench/<lang>/*.wav + *.json):
  python3 apps/stt/benchmark.py --run --models small,medium,large-v3-turbo
  python3 apps/stt/benchmark.py --run --save apps/stt/BENCH_RESULTS.md

Decisão documentada em apps/stt/MODEL.md.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

DATA_DIR = Path(os.getenv("BENCH_DATA", "data/bench"))


def norm(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s.lower())
    return re.sub(r"[^\w\u00C0-\u024F ]+", " ", s).split()


def levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    if not r:
        return float(bool(h))
    return levenshtein(r, h) / len(r)


def coverage(ref: str, hyp: str) -> float:
    r = norm(ref)
    if not r:
        return 0.0
    seen = set(norm(hyp)) if len(norm(hyp)) <= len(r) else set(norm(hyp))
    return sum(1 for w in r if w in seen) / len(r)


def acquire_speak(seconds: int) -> bytes:
    import sounddevice as sd  # type: ignore
    import numpy as np  # type: ignore
    frames = int(16000 * seconds)
    print(f"  [mic] gravando {seconds}s... fale agora.")
    a = sd.rec(frames, samplerate=16000, channels=1, dtype="int16")
    sd.wait()
    print("  [mic] gravado.")
    return bytes(np.asarray(a).tobytes())


def save_fixture(pcm: bytes, lang: str, ref: str, idx: int) -> None:
    d = DATA_DIR / lang
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{idx}.wav").write_bytes(pcm)
    (d / f"{idx}.json").write_text(json.dumps({"lang": lang, "text": ref}, ensure_ascii=False))


def read_fixtures(base: Path) -> list[dict]:
    out = []
    for lang_dir in sorted(base.iterdir()):
        if not lang_dir.is_dir():
            continue
        for wav in sorted(lang_dir.glob("*.wav")):
            meta = wav.with_suffix(".json")
            if not meta.exists():
                continue
            rec = json.loads(meta.read_text(encoding="utf-8"))
            out.append({"lang": lang_dir.name, "ref": rec.get("text", ""),
                        "wav": bytes(wav.read_bytes()), "name": str(wav)})
    return out


def run_fixture(model, pcm: bytes, passes: int) -> dict:
    import numpy as np  # type: ignore
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    lat, texts, langs = [], [], []
    cpu0 = time.process_time()
    for _ in range(passes):
        t0 = time.perf_counter()
        segs, info = model.transcribe(audio, language=None, task="transcribe",
                                      vad_filter=True, no_speech_threshold=0.6,
                                      log_prob_threshold=-0.8,
                                      condition_on_previous_text=False)
        texts.append(" ".join(s.text.strip() for s in segs).strip())
        langs.append(getattr(info, "language", "unknown"))
        lat.append(time.perf_counter() - t0)
    cpu = time.process_time() - cpu0
    wall = sum(lat)
    return {"lat": lat, "text": texts[-1], "lang": langs[-1], "cpu_s": cpu, "wall_s": wall}


def model_row(model_obj, name, fixtures, passes) -> dict:
    wer_list, cov_list, lang_ok, lats = [], [], 0, []
    cpu_s = wall_s = 0.0
    for f in fixtures:
        r = run_fixture(model_obj, f["wav"], passes)
        lats.extend(r["lat"])
        cpu_s += r["cpu_s"]
        wall_s += r["wall_s"]
        wer_list.append(wer(f["ref"], r["text"]))
        cov_list.append(coverage(f["ref"], r["text"]))
        if r["lang"] == f["lang"] or (f["lang"].startswith(r["lang"]) or r["lang"].startswith(f["lang"])):
            lang_ok += 1
    n = len(fixtures)
    p50 = statistics.median(lats)
    s = sorted(lats)
    p95 = s[max(0, (len(s) * 95) // 100 - 1)]
    return {
        "model": name,
        "wer": round(statistics.mean(wer_list), 4),
        "wer_p": f"{statistics.mean(wer_list):.1%}",
        "coverage": f"{statistics.mean(cov_list):.1%}",
        "lang_acc": f"{lang_ok}/{n}",
        "lat_p50_ms": round(p50 * 1000),
        "lat_p95_ms": round(p95 * 1000),
        "cpu_ratio": f"{cpu_s / wall_s:.2f}" if wall_s else "-",
    }


def markdown_table(rows: list[dict]) -> str:
    hdr = "| modelo | WER | cobertura | idioma | p50(ms) | p95(ms) |"
    sep = "|---|---|---|---|---|---|"
    lines = [hdr, sep]
    for r in rows:
        lines.append(f"| {r['model']} | {r['wer_p']} | {r['coverage']} | {r['lang_acc']} "
                     f"| {r['lat_p50_ms']} | {r['lat_p95_ms']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speak", action="store_true", help="grava e digita a referência dos fixtures")
    ap.add_argument("--run", action="store_true", help="roda o benchmark nos fixtures")
    ap.add_argument("--seconds", type=int, default=6)
    ap.add_argument("--langs", default="pt,en")
    ap.add_argument("--models", default="small,medium,large-v3-turbo")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    if args.speak:
        for lang in (l.strip() for l in args.langs.split(",") if l.strip()):
            for i in range(1, 3):
                input(f"\n--{lang.upper()} fixture {i}: Enter para gravar...")
                pcm = acquire_speak(args.seconds)
                ref = input("Digite o texto falado (referência): ").strip() or input("REFERÊNCIA (obrigatório): ").strip()
                save_fixture(pcm, lang, ref, i)
        print(f"Fixtures salvos em {DATA_DIR}/. Agora rode: python3 apps/stt/benchmark.py --run")
        return

    from faster_whisper import WhisperModel  # type: ignore
    fixtures = read_fixtures(DATA_DIR)
    if not fixtures:
        print(f"Sem fixtures em {DATA_DIR}. Comece por --speak.")
        return
    print(f"[bench] {len(fixtures)} fixtures: " + ", ".join(f"{f['lang']}({f['name']})" for f in fixtures))
    rows = []
    for name in (m.strip() for m in args.models.split(",") if m.strip()):
        print(f"[bench] carregando {name}...")
        t0 = time.perf_counter()
        m = WhisperModel(name, device="cpu", compute_type="int8")
        load_s = time.perf_counter() - t0
        r = model_row(m, name, fixtures, args.passes)
        r["load_s"] = round(load_s, 1)
        rows.append(r)
        print([(k, v) for k, v in r.items()])
    table = markdown_table(rows)
    print("\n" + table)
    if args.save:
        Path(args.save).write_text(
            f"# Resultados do benchmark (gerado por apps/stt/benchmark.py)\n\n{table}\n", encoding="utf-8")
        print(f"salvo em {args.save}")


if __name__ == "__main__":
    main()