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
import wave
from pathlib import Path

DATA_DIR = Path(os.getenv("BENCH_DATA", "data/bench"))

# inglês: distil-large-v3 entra no rodapé do benchmark (é English-only)
WAV_MODELS_DEFAULT = "small,medium,large-v3-turbo,distil-large-v3"


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
    if not base.exists():
        return []
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


# ---------- benchmark por WAV único (trecho de vídeo pelo BlackHole) ----------

def load_wav(path: str) -> tuple[bytes, int, int, float]:
    """Lê WAV; devolve (pcm16k_mono, rate_original, channels_original, duracao_s).

    Aceita qualquer rate/canais: converte para o PCM mono 16k consumido pelo modelo
    (mesma conversão do service: `capture.to_mono16k`).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../audio"))
    from capture import to_mono16k, SAMPLE_RATE  # noqa: E402

    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
        duration = w.getnframes() / max(rate, 1)
    pcm16, _down, _res = to_mono16k(raw, rate, ch)
    return pcm16, rate, ch, duration


def _transcribe_passes(model, audio, passes: int, lang: str | None = None) -> dict:
    lat, texts, langs = [], [], []
    for _ in range(passes):
        t0 = time.perf_counter()
        segs, info = model.transcribe(audio, language=lang, task="transcribe",
                                      vad_filter=True, no_speech_threshold=0.6,
                                      log_prob_threshold=-0.8,
                                      condition_on_previous_text=False)
        texts.append(" ".join(s.text.strip() for s in segs).strip())
        langs.append(getattr(info, "language", "unknown"))
        lat.append(time.perf_counter() - t0)
    return {"text": texts[-1], "lang": langs[-1], "lat": lat}


def run_wav_benchmark(model_names: list[str], wav: str, ref: str | None,
                      passes: int, meta_path: str | None = None) -> tuple[list[dict], dict]:
    """Roda vários modelos no MESMO WAV (trecho de vídeo 16k mono).

    Retorna (rows, context). rows: métricas por modelo (WER vs referência oficial,
    cobertura, idioma, latência p50/p95). context: dados de áudio/benchmark p/ o
    relatório final (WAV, referência, duração, config de áudio do sidecar do diag).
    """
    from faster_whisper import WhisperModel  # type: ignore
    import numpy as np  # type: ignore

    pcm16, rate, ch, duration = load_wav(wav)
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    ref_text = None
    if ref:
        ref_text = Path(ref).read_text(encoding="utf-8").strip()

    rows = []
    for name in model_names:
        print(f"[bench] carregando {name}...")
        t0 = time.perf_counter()
        m = WhisperModel(name, device="cpu", compute_type="int8")
        load_s = time.perf_counter() - t0
        r = _transcribe_passes(m, audio, passes)
        lats = sorted(r["lat"])
        p95 = lats[max(0, (len(lats) * 95) // 100 - 1)]
        wer_v = wer(ref_text, r["text"]) if ref_text is not None else float("nan")
        cov_v = coverage(ref_text, r["text"]) if ref_text is not None else float("nan")
        lang_ok = "—" if ref_text is None else ("ok" if r["lang"] == "en" else "!!")
        rows.append({
            "model": name,
            "wer_v": wer_v,
            "wer_p": f"{wer_v:.1%}",
            "coverage_v": cov_v,
            "coverage": f"{cov_v:.1%}",
            "lang_acc": lang_ok,
            "lang": r["lang"],
            "lat_p50_ms": round(statistics.median(lats) * 1000),
            "lat_p95_ms": round(p95 * 1000),
            "load_s": round(load_s, 1),
            "text": r["text"],
        })
        print(f"    WER={rows[-1]['wer_p']} cobertura={rows[-1]['coverage']} "
              f"lang={r['lang']} p50={rows[-1]['lat_p50_ms']}ms p95={rows[-1]['lat_p95_ms']}ms "
              f"load={load_s:.0f}s :: {r['text'][:90]}")

    context = {"wav": str(Path(wav).resolve()), "ref": ref, "ref_text": ref_text,
               "duration_s": round(duration, 2), "rate_original": rate,
               "channels_original": ch, "passes": passes, "audio_meta": None}
    if meta_path and Path(meta_path).exists():
        try:
            context["audio_meta"] = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        except Exception:
            context["audio_meta"] = None
    return rows, context


def recommend(rows: list[dict]) -> tuple[dict, str]:
    """Escolhe o modelo por EVIDÊNCIA: menor WER com cobertura alta; p50 desempata."""
    ok = [r for r in rows if r.get("wer_v") is not None and not (r["wer_v"] != r["wer_v"])]
    if not ok:
        return {}, "sem referência oficial — impossível comparar WER"
    good = [r for r in ok if r["wer_v"] <= 0.25]
    pool = good or ok
    pool = sorted(pool, key=lambda r: (r["wer_v"], -r.get("coverage_v", 0.0), r["lat_p50_ms"]))
    best = pool[0]
    why = ("menor WER entre modelos candidatos (≤ 25%), com cobertura e p50 como desempate"
           if good else "nenhum modelo ≤ 25% de WER — menor erro absoluto")
    return best, why


def wav_report(rows: list[dict], ctx: dict, chosen: dict, why: str) -> str:
    lines = [
        "# Bench STT — trecho de vídeo (BlackHole/loopback)",
        "",
        "- **Data:** {0}".format(_now()),
        "- **Áudio (benchmark only, local):** `{0}`".format(ctx["wav"]),
    ]
    if ctx["ref"]:
        lines.append("- **Referência (transcrição oficial):** `{0}`".format(ctx["ref"]))
    lines += [
        "- **Duração:** {0}s | **Passes/modelo:** {1}".format(ctx["duration_s"], ctx["passes"]),
    ]
    am = ctx.get("audio_meta") or {}
    if am:
        for name, cfg in (am.get("sources") or {}).items():
            lines.append(
                "- **{0}:** {1} | rate nativo {2}Hz | {3}ch | mono 16k | "
                "downmix {4} | resample {5} | RMS médio {6}dB (min {7} / máx {8})".format(
                    name, cfg.get("device", "?"), cfg.get("device_rate"),
                    cfg.get("device_channels"), "sim" if cfg.get("downmixed") else "não",
                    "sim" if cfg.get("resampled") else "não",
                    cfg.get("rms_db_avg", "?"), cfg.get("rms_db_min"),
                    cfg.get("rms_db_max")))
        lines += ["", "### Configuração de áudio (evidência)", "", "```json",
                  json.dumps(am, ensure_ascii=False, indent=1), "```"]
    m = "modelo | WER | cobertura | idioma | p50(ms) | p95(ms) |"
    lines += ["", m, "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['wer_p']} | {r['coverage']} | {r['lang_acc']} "
                     f"| {r['lat_p50_ms']} | {r['lat_p95_ms']} |")
    if chosen:
        lines += [
            "",
            "## Escolha (por evidência no Mac)",
            "",
            "- **Modelo:** `{0}` — {1}".format(chosen["model"], why),
            "- WER {0} | cobertura {1} | p50 {2}ms | p95 {3}ms | idioma {4}".format(
                chosen["wer_p"], chosen["coverage"], chosen["lat_p50_ms"],
                chosen["lat_p95_ms"], chosen.get("lang", "?")),
        ]
    if ctx.get("ref_text"):
        lines += ["", "### Transcrição do melhor candidato", "",
                  ctx["ref_text"][:600], ""]
    return "\n".join(lines) + "\n"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mkdir", action="store_true",
                    help="cria data/bench/{pt,en} (fixtures sintéticas primeiro; áudio real ⏳ Mac)")
    ap.add_argument("--speak", action="store_true", help="grava e digita a referência dos fixtures")
    ap.add_argument("--run", action="store_true", help="roda o benchmark nos fixtures")
    ap.add_argument("--seconds", type=int, default=6)
    ap.add_argument("--langs", default="pt,en")
    ap.add_argument("--models", default="small,medium,large-v3-turbo")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--save", default="")
    ap.add_argument("--wav", default="",
                    help="benchmark no MESMO WAV (trecho de vídeo pelo BlackHole) em vários modelos")
    ap.add_argument("--ref", default="", help="arquivo .txt com a transcrição oficial do trecho")
    ap.add_argument("--meta", default="",
                    help="meta_<ts>.json do diag_capture (config de áudio de evidência)")
    args = ap.parse_args()

    if args.wav:
        models = [m.strip() for m in (args.models or WAV_MODELS_DEFAULT).split(",") if m.strip()]
        rows, ctx = run_wav_benchmark(models, args.wav,
                                      args.ref or None, args.passes,
                                      args.meta or None)
        chosen, why = recommend(rows)
        table = markdown_table(rows)
        print("\n" + table)
        print(f"\n[bench] recomendado: {chosen.get('model', '?')} — {why}")
        report = wav_report(rows, ctx, chosen, why)
        out = args.save or str(DATA_DIR / "BENCH_REPORT.md")
        Path(out).write_text(report, encoding="utf-8")
        print(f"[bench] relatório salvo em {out}")
        return

    if args.mkdir:
        for lang in (l.strip() for l in args.langs.split(",") if l.strip()):
            (DATA_DIR / lang).mkdir(parents=True, exist_ok=True)
            print(f"[bench] {DATA_DIR / lang}/ criado. "
                  f"Grave o áudio real no Mac com --speak e rode --run depois.")
        return

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