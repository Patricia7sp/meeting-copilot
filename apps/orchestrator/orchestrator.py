"""Orchestrator: router proativo + Language Coach + Work Copilot.

Entrega 1: 100% regras locais + templates. Sem chamada LLM obrigatória.
Entrega 3: pluga LLM real (Gemini/OpenAI/Groq/Ollama) atrás de `generate_with_llm()`.
"""
from __future__ import annotations
import re
import sys
import os
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../packages/context"))
from window import ContextWindow


FILLERS = {"uhm", "ahn", "hmm", "uh", "ah", "é", "tipo assim", "tipo", "huh", "mm"}

QUESTION_RE = re.compile(r"\?\s*$")
ENGLISH_QUESTION_RE = re.compile(
    r"^\s*(what|how|why|when|where|who|do you|did you|have you|can you|tell me|what did|how about)\b",
    re.IGNORECASE,
)
TECH_RE = re.compile(
    r"(bigquery|join|pipeline|partition|cluster|materialized view|bytes|cardinalidade|"
    r"docker|kubernetes|python|sql|api|latency|erro|bug|deploy|arquitetura|etl)",
    re.IGNORECASE,
)
ENGLISH_LEARNING_RE = re.compile(
    r"(weekend|what did you|how do you say|how to say|grammar|vocabulary|pronunciation|"
    r"idiom|phrasal verb|past tense|present perfect)",
    re.IGNORECASE,
)


@dataclass
class Decision:
    action: str  # ignore | lang_coach | work_copilot
    reason: str
    confidence: float


class Orchestrator:
    def __init__(self, mode: str = "auto", cooldown_seconds: int = 10, max_turns: int = 12):
        assert mode in ("auto", "english", "work")
        self.mode = mode
        self.cooldown = cooldown_seconds
        self.ctx = ContextWindow(max_turns=max_turns)
        self._last_insight_ts = 0.0

    # ---------- entrada ----------
    def observe(self, speaker: str, text: str, lang: str = "unknown") -> None:
        self.ctx.add(speaker, text, lang)

    # ---------- decisão proativa ----------
    def route(self, speaker: str, text: str) -> Decision:
        t = text.strip()
        words = re.findall(r"[\w']+", t, re.UNICODE)

        if len(words) < 4:
            return Decision("ignore", "muito curto", 0.95)
        if t.lower().strip(" .") in FILLERS:
            return Decision("ignore", "filler", 0.95)

        forced = None
        if self.mode == "english":
            forced = "lang_coach"
        elif self.mode == "work":
            forced = "work_copilot"

        tech = bool(TECH_RE.search(t))
        learn = bool(ENGLISH_LEARNING_RE.search(t))
        question = bool(QUESTION_RE.search(t) or ENGLISH_QUESTION_RE.search(t))

        if self.mode == "auto":
            if tech and not learn:
                return Decision("work_copilot", "sinal técnico", 0.85)
            if learn or (question and self._looks_like_class()):
                return Decision("lang_coach", "sinal de aula/inglês", 0.8)
            if question:
                # pergunta genérica: decide pelo histórico
                if self._looks_like_class():
                    return Decision("lang_coach", "pergunta em contexto de aula", 0.6)
                return Decision("work_copilot", "pergunta em contexto de trabalho", 0.6)
            return Decision("ignore", "sem sinal suficiente", 0.7)

        # modo forçado ainda respeita anti-spam de obviedade
        if forced == "lang_coach" and tech and not learn and not question:
            return Decision("ignore", "técnico puro em modo english", 0.6)
        if forced == "work_copilot" and learn and not tech:
            # aula de inglês no meio do trabalho? ainda gera, mas com confiança menor
            return Decision("lang_coach", "sinal de inglês mesmo em modo work", 0.55)
        action = forced or "ignore"
        return Decision(action, f"modo fixo {self.mode}", 0.9)

    def _looks_like_class(self) -> bool:
        blob = self.ctx.window_text(6).lower()
        hits = sum(1 for pat in ["weekend", "verb", "tense", "pronunciation", "teacher",
                                 "professor", "vocabulary", "idiom", "homework"] if pat in blob)
        return hits >= 1

    def should_emit(self) -> bool:
        return (time.time() - self._last_insight_ts) >= self.cooldown

    # ---------- geração ----------
    def handle(self, speaker: str, text: str, lang: str = "unknown") -> dict | None:
        self.observe(speaker, text, lang)
        dec = self.route(speaker, text)
        if dec.action == "ignore":
            return {"type": "noop", "reason": dec.reason}
        if not self.should_emit():
            return {"type": "noop", "reason": "cooldown"}
        self._last_insight_ts = time.time()
        if dec.action == "lang_coach":
            return {"type": "insight", "kind": "lang_coach",
                    "decision": dec.reason, "card": self.lang_coach_card(text)}
        return {"type": "insight", "kind": "work_copilot",
                "decision": dec.reason, "card": self.work_copilot_card(text)}

    # ----- templates locais (funcionam offline) -----
    def lang_coach_card(self, last_text: str) -> dict:
        low = last_text.lower()
        card = {"say_this": [], "vocab": [], "follow_up": "", "grammar_tip": ""}
        if "weekend" in low or "what did you do" in low:
            card["say_this"] = [
                "I stayed home most of the weekend, but I went out on Sunday.",
                "It was pretty quiet — I caught up on some rest and met a friend for coffee.",
                "Not much, just recharged. How about you?",
            ]
            card["vocab"] = ["stay home", "go out", "spend time", "catch up on (rest/series)"]
            card["follow_up"] = "How about you? Did you do anything interesting?"
            card["grammar_tip"] = "Past simple: I stayed / I went (go → went)."
        elif "how do you say" in low or "how to say" in low:
            card["say_this"] = ["How do you say '___' in English?", "What's the natural way to say '___'?"]
            card["vocab"] = ["say", "mean", "natural way"]
            card["follow_up"] = "Could you give me an example with that word?"
            card["grammar_tip"] = "Use 'How do you say X?' (não 'How to say')."
        else:
            card["say_this"] = [
                "That's interesting — could you give me an example?",
                "I see. Could you say that in a different way?",
            ]
            card["vocab"] = ["That's interesting", "I see", "Could you...?"]
            card["follow_up"] = "What do you mean by that, exactly?"
            card["grammar_tip"] = "Para ganhar tempo: 'Let me think how to put this...'"
        return card

    def work_copilot_card(self, last_text: str) -> dict:
        low = last_text.lower()
        card = {"summary": "", "to_check": [], "to_ask": [], "risks": []}
        if "join" in low and ("bigquery" in low or "pipeline" in low):
            card["summary"] = "Hipótese: join grande sem partition pruning causando lentidão."
            card["to_check"] = [
                "Cardinalidade do join (fan-out?) + bytes processados no job",
                "Partition pruning ativo? Filtro na coluna particionada dos dois lados?",
                "Clustering das tabelas nas chaves do join",
                "Comparar com materialized view / tabela agregada",
            ]
            card["to_ask"] = [
                "Quantos bytes o job está escaneando vs. retornando?",
                "O filtro de data chega até o scan ou só após o join?",
            ]
            card["risks"] = ["Custo BigQuery explode com full scan", "Skew em chave quente"]
        else:
            card["summary"] = last_text[:120]
            card["to_check"] = ["Reproduzir com exemplo mínimo", "Checar logs/métricas do trecho citado"]
            card["to_ask"] = ["Qual o comportamento esperado vs. observado?", "Quem é o owner desse trecho?"]
            card["risks"] = ["Decisão sem dono vira retrabalho"]
        return card


def generate_with_llm(prompt: str) -> str | None:
    """Placeholder Entrega 3: lê LLM_PROVIDER env e chama Gemini/OpenAI/Groq/Ollama.
    Entrega 1 retorna None -> usa templates locais."""
    return None
