"""Orchestrator: classificador/roteador + Language Coach + Work Copilot.

Pipeline de decisão (prioridade: fidelidade antes de insights):
1. Só falas CONSOLIDADAS (stage=final) geram insight; provisórios ficam de fora.
2. Limiar de confiança: resumo/insight só se confidence >= INSIGHT_MIN_CONFIDENCE
   (e não-low).
3. Deduplicação: mesma fala normalizada não gera insight repetido.
4. Contexto CONSOLIDADO (mesmo locutor é juntado; baixa confiança sai da janela).
5. Classificação/roteamento via modelo OpenRouter (Jev, env ROUTER_MODEL) quando
   disponível; sem chave, regras locais. LLM GERADOR entra só na sugestão final.
"""
from __future__ import annotations
import os
import re
import sys
import time
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../packages/context"))
from window import ContextWindow  # noqa: E402


FILLERS = {"uhm", "ahn", "hmm", "uh", "ah", "é", "tipo assim", "tipo", "huh", "mm"}
INSIGHT_MIN_CONFIDENCE = float(os.getenv("INSIGHT_MIN_CONFIDENCE", "0.5"))
MAX_RECENT_FPS = int(os.getenv("DEDUP_WINDOW_SIZE", "20"))

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
    priority: float = 0.5   # 0..1: urgência/importância do insight
    needs_insight: bool = True


class Orchestrator:
    def __init__(self, mode: str = "auto", cooldown_seconds: int = 10, max_turns: int = 12):
        assert mode in ("auto", "english", "work")
        self.mode = mode
        self.cooldown = cooldown_seconds
        self.ctx = ContextWindow(max_turns=max_turns)
        self._last_insight_ts = 0.0
        self._recent_fps: list[str] = []
        self.classifier = None
        self._load_classifier()

    # ---------- classificador (OpenRouter "Jev") ou regras locais ----------
    def _load_classifier(self):
        try:
            from classifier import classifier_for  # type: ignore
            self.classifier = classifier_for()
        except Exception as e:
            self.classifier = None
            print(f"[orchestrator] classificador indisponível: {e} (usa regras locais)")

    # ---------- entrada ----------
    def observe(self, speaker: str, text: str, lang: str = "unknown",
                confidence: float | None = None, low: bool = False) -> None:
        self.ctx.add(speaker, text, lang, confidence=confidence, low=low)

    # ---------- decisão proativa ----------
    def route(self, speaker: str, text: str) -> Decision:
        t = text.strip()
        words = re.findall(r"[\w']+", t, re.UNICODE)

        if len(words) < 4:
            return Decision("ignore", "muito curto", 0.95, priority=0.1)
        if t.lower().strip(" .") in FILLERS:
            return Decision("ignore", "filler", 0.95, priority=0.1)

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
                return Decision("work_copilot", "sinal técnico", 0.85, priority=0.7)
            if learn or (question and self._looks_like_class()):
                return Decision("lang_coach", "sinal de aula/inglês", 0.8, priority=0.7)
            if question:
                if self._looks_like_class():
                    return Decision("lang_coach", "pergunta em contexto de aula", 0.6, priority=0.5)
                return Decision("work_copilot", "pergunta em contexto de trabalho", 0.6, priority=0.5)
            return Decision("ignore", "sem sinal suficiente", 0.7, priority=0.2)

        if forced == "lang_coach" and tech and not learn and not question:
            return Decision("ignore", "técnico puro em modo english", 0.6, priority=0.3)
        if forced == "work_copilot" and learn and not tech:
            return Decision("lang_coach", "sinal de inglês mesmo em modo work", 0.55, priority=0.5)
        action = forced or "ignore"
        return Decision(action, f"modo fixo {self.mode}", 0.9, priority=0.6)

    def decide(self, text: str) -> Decision:
        """Roteia via classificador (OpenRouter) se disponível; senão, regras locais."""
        if self.classifier is not None:
            try:
                dec = self.classifier(self.ctx.window_text(6), text)
                if dec is not None:
                    return dec
            except Exception as e:
                print(f"[orchestrator] classificador falhou: {e} (usa regras locais)")
        last_sp = self.ctx.turns[-1].speaker if self.ctx.turns else "OTHERS"
        return self.route(last_sp, text)

    def _looks_like_class(self) -> bool:
        blob = self.ctx.window_text(6).lower()
        hits = sum(1 for pat in ["weekend", "verb", "tense", "pronunciation", "teacher",
                                 "professor", "vocabulary", "idiom", "homework"] if pat in blob)
        return hits >= 1

    def should_emit(self) -> bool:
        return (time.time() - self._last_insight_ts) >= self.cooldown

    # ---------- deduplicação ----------
    def _fingerprint(self, text: str) -> str:
        return re.sub(r"[^\w\u00C0-\u024F ]+", "", text.lower()).strip()

    def _dedup(self, text: str) -> bool:
        fp = self._fingerprint(text)
        if fp in self._recent_fps:
            return True
        self._recent_fps.append(fp)
        if len(self._recent_fps) > MAX_RECENT_FPS:
            del self._recent_fps[: len(self._recent_fps) - MAX_RECENT_FPS]
        return False

    # ---------- geração ----------
    def handle(self, speaker: str, text: str, lang: str = "unknown",
               confidence: float | None = None, low_confidence: bool = False) -> dict:
        self.observe(speaker, text, lang, confidence=confidence, low=low_confidence)
        # fidelidade primeiro: fala de baixa confiança não gera insight
        if confidence is not None and (low_confidence or confidence < INSIGHT_MIN_CONFIDENCE):
            return {"type": "noop", "reason": "low_confidence"}
        if self._dedup(text):
            return {"type": "noop", "reason": "dedup"}
        dec = self.decide(text)
        if dec.action == "ignore" or not dec.needs_insight:
            return {"type": "noop", "reason": dec.reason}
        if dec.priority < 0.2:
            return {"type": "noop", "reason": f"low_priority ({dec.priority:.2f})"}
        if not self.should_emit():
            return {"type": "noop", "reason": "cooldown"}
        self._last_insight_ts = time.time()
        kind = dec.action  # lang_coach | work_copilot
        template = self.lang_coach_card(text) if kind == "lang_coach" else self.work_copilot_card(text)
        refined, used = self._try_refine(kind, text)
        card = refined or template
        out: dict = {"type": "insight", "kind": kind, "decision": dec.reason,
                     "priority": round(dec.priority, 2), "card": card}
        if refined:
            out["llm_model"] = used
        return out

    def _try_refine(self, kind: str, last_text: str) -> tuple[dict | None, str]:
        """Se LLM_PROVIDER=openrouter + chave presente, refina o template. Senão (None, motivo)."""
        import os as _os
        if _os.getenv("LLM_PROVIDER", "none").lower() != "openrouter":
            return None, "llm-off"
        try:
            from llm_openrouter import enhance_card, configured
        except ImportError:
            try:
                from apps.orchestrator.llm_openrouter import enhance_card, configured  # type: ignore
            except ImportError:
                return None, "no-module"
        if not configured():
            return None, "no-key"
        try:
            return enhance_card(kind, self.ctx.to_prompt_block(), last_text)
        except Exception as e:
            return None, f"llm-error: {e}"

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
