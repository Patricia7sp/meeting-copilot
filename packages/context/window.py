"""Context window + rolling summary (stdlib only, sem dependências)."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class Turn:
    speaker: str  # YOU | OTHERS | PROFESSOR...
    text: str
    lang: str = "unknown"
    ts: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class ContextWindow:
    def __init__(self, max_turns: int = 12):
        self.max_turns = max_turns
        self.turns: list[Turn] = []
        self.rolling_summary: str = ""

    def add(self, speaker: str, text: str, lang: str = "unknown") -> None:
        text = text.strip()
        if not text:
            return
        self.turns.append(Turn(speaker=speaker, text=text, lang=lang))
        # compacta: quando estoura, resume os mais antigos de forma extrativa simples
        if len(self.turns) > self.max_turns * 2:
            overflow = self.turns[: len(self.turns) - self.max_turns]
            self.rolling_summary = self._fold_summary(self.rolling_summary, overflow)
            self.turns = self.turns[-self.max_turns :]

    def window(self, n: int | None = None) -> list[Turn]:
        n = n or self.max_turns
        return self.turns[-n:]

    def window_text(self, n: int | None = None) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.window(n))

    def _fold_summary(self, prev: str, overflow: list[Turn]) -> str:
        # Resumo extrativo barato: pega frases com palavras-chave (decisão, pergunta, verbo técnico)
        # Na V1 isso vira chamada ao modelo pequeno local.
        keywords = re.compile(
            r"(decid|todo|action|problema|erro|join|bigquery|pipeline|weekend|verb|tense|pronunc)",
            re.IGNORECASE,
        )
        picks = [f"{t.speaker}: {t.text}" for t in overflow if keywords.search(t.text)]
        if not picks:
            picks = [f"{t.speaker}: {t.text}" for t in overflow[:3]]
        chunk = " | ".join(picks)[:800]
        base = (prev + " || " if prev else "")
        return (base + chunk)[:1500]

    def to_prompt_block(self) -> str:
        parts = []
        if self.rolling_summary:
            parts.append(f"[HISTÓRICO RESUMIDO]\n{self.rolling_summary}")
        parts.append(f"[JANELA RECENTE]\n{self.window_text()}")
        return "\n\n".join(parts)
