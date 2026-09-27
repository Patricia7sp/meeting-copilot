"""Testes do benchmark de modelos (Feature 005): WER, cobertura, agregados e fixtures.

Uso: python3 apps/stt/test_benchmarks.py
"""
from __future__ import annotations
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import benchmark  # noqa: E402


@dataclass
class FakeModel:
    text: str = ""
    lang: str = "pt"
    latency: float = 0.05
    calls: int = field(default=0)

    def transcribe(self, audio, **kw):
        import time
        self.calls += 1
        time.sleep(self.latency)
        return self._gen(), type("I", (), {"language": self.lang})()

    def _gen(self):
        class Seg:
            def __init__(self, t):
                self.text = t
        return iter([Seg(self.text)])


def test_wer_zero_and_errors():
    assert benchmark.wer("casa verde", "casa verde") == 0.0
    assert benchmark.wer("casa verde", "casa azul") == 0.5
    assert benchmark.wer("", "a") == 1.0
    assert benchmark.wer("casa", "") == 1.0


def test_coverage():
    assert benchmark.coverage("casa verde", "casa verde") == 1.0
    assert benchmark.coverage("casa verde", "casa") == 0.5
    assert benchmark.coverage("", "x") == 0.0


def test_model_row_aggregates_latency():
    m = FakeModel(text="casa verde", lang="pt", latency=0.05)
    fixtures = [
        {"lang": "pt", "ref": "casa verde", "wav": b"\x00" * 3200, "name": "pt/1.wav"},
        {"lang": "pt", "ref": "casa azul", "wav": b"\x00" * 3200, "name": "pt/2.wav"},
    ]
    row = benchmark.model_row(m, "fake-small", fixtures, passes=2)
    assert m.calls == 2 * 2, m.calls               # modelo chamado p/ cada fixture x passes
    assert row["model"] == "fake-small"
    assert row["lang_acc"].startswith("2/2")
    assert row["lat_p50_ms"] == 50 and row["lat_p95_ms"] == 50
    assert row["wer_p"] != "" and row["coverage"] != ""


def test_no_fixtures_returns_empty():
    assert benchmark.read_fixtures(Path("/tmp/nao-existe-benchmark-xyz")) == []


def test_fixture_roundtrip():
    base = Path(tempfile.mkdtemp(prefix="mc_bench_"))
    benchmark.DATA_DIR = base
    benchmark.save_fixture(b"\x00" * 3200, "pt", "bom dia", 1)
    fixtures = benchmark.read_fixtures(base)
    assert len(fixtures) == 1
    assert fixtures[0]["ref"] == "bom dia" and fixtures[0]["lang"] == "pt"


def test_norm_punctuation_insensitive():
    assert benchmark.norm("Casa, verde!") == benchmark.norm("casa verde")
    assert benchmark.norm("big-query") == benchmark.norm("big query")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("TODOS OS TESTES PASSARAM.")


if __name__ == "__main__":
    main()