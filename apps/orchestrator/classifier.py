"""Classificador/roteador da conversa via OpenRouter (o "Jev").

Separação de responsabilidades:
- CLASSIFICAÇÃO (este módulo): dado o contexto consolidado + última fala, decide
  a ação (ignore/lang_coach/work_copilot), prioridade (0..1) e se o insight é
  necessário. Modelo configurável por ROUTER_MODEL (default: cadeia free).
- GERAÇÃO (llm_openrouter.enhance_card): só a SUGESTÃO final usa o LLM gerador.

Sem chave/erro/parse: retorna None -> o Orchestrator cai para a rota local.
Avaliação: python3 apps/orchestrator/router_eval.py --live  (requer chave).
"""
from __future__ import annotations
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from session import SESSION_TYPES, SessionResult  # type: ignore
from llm_openrouter import BASE, _headers, chat, configured  # type: ignore

if TYPE_CHECKING:
    from orchestrator import Decision  # noqa: F401


JEV_MODEL = os.getenv("JEV_MODEL", "typesafe/jev-1.13")
ROUTER_MODEL = (os.getenv("ROUTER_MODEL", "") or None) or JEV_MODEL


@dataclass
class _Raw:
    action: str | None = None
    priority: float = 0.5
    needs_insight: bool = True
    reason: str = ""


CLASSIFY_SYSTEM = (
    "Você é um roteador de uma reunião/aula em tempo real (PT-BR e EN). "
    "Analise o contexto e a ÚLTIMA fala e responda SOMENTE JSON, sem markdown: "
    '{"action": "ignore"|"lang_coach"|"work_copilot", '
    '"priority": 0.0, "need": true, "reason": "curta justificativa em PT"}. '
    "ignore = fala curta, filler ou sem valor; lang_coach = contexto de aula/inglês; "
    "work_copilot = tema técnico de trabalho. priority 0..1 (urgencia de insight). "
    "need=false nunca gera insight mesmo se o texto for curto."
)


def parse_classify(text: str) -> _Raw | None:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").split("\n", 1)[1] if "\n" in t else t.strip("`")
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    try:
        data = json.loads(t)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    act = data.get("action") if data.get("action") in ("ignore", "lang_coach", "work_copilot") else None
    try:
        prio = min(max(float(data.get("priority", 0.5)), 0.0), 1.0)
    except Exception:
        prio = 0.5
    return _Raw(action=act, priority=prio,
                needs_insight=bool(data.get("need", True)), reason=str(data.get("reason", ""))[:80])


def _num(value, default: float = 0.5) -> float:
    try:
        v = float(value)
        return min(max(v, 0.0), 1.0)
    except (TypeError, ValueError):
        return default


def _bool(value, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    try:
        t = str(value).strip().lower()
        if t in ("true", "yes", "1", "sim"):
            return True
        if t in ("false", "no", "0", "não"):
            return False
    except Exception:
        pass
    return default


def _jev_classify(context_block: str, last_text: str) -> tuple[Decision | None, SessionResult | None, str]:
    """Executa o Jev pelo endpoint de decisões estruturadas.

    Uma unica chamada com 5 perguntas: action, priority, need, session_type,
    session_confidence. Retorna (Decision, SessionResult, modelo) ou (None, None,
    motivo) p/ fallback local. No maximo UMA tentativa extra em erro de rede.
    """
    for attempt in range(2):
        record = f"Contexto: {context_block}\nÚltima fala: {last_text}"
        payload = {
            "model": ROUTER_MODEL,
            "state": {
                "description": "Uma fala e seu contexto em reunião ou aula.",
                "records": [{"id": "turn", "record": record}],
            },
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "Classifique o registro conforme o objetivo do copiloto.",
                    "criteria": {
                        "ignore": "Fala curta, filler ou sem contexto suficiente para insight.",
                        "lang_coach": "Aula ou prática de inglês que merece sugestão de resposta ou correção.",
                        "work_copilot": "Discussão de trabalho técnico que merece um insight acionável.",
                    },
                },
                "priority": {
                    "type": "number",
                    "instructions": "Urgência do insight 0..1 (0 = nenhuma, 1 = agora).",
                    "min": 0.0,
                    "max": 1.0,
                },
                "need": {
                    "type": "boolean",
                    "instructions": "true se esta fala merece um insight mesmo sem LLM de geração.",
                },
                "session_type": {
                    "type": "choice",
                    "instructions": "Tipo da sessao inteira; english_lesson = aula/pratica de ingles; "
                                    "work_meeting = reunião de trabalho técnico; general_conversation = "
                                    "conversa comum; unknown = ainda sem evidência suficiente.",
                    "criteria": {
                        "english_lesson": "Predomina prática/aula de inglês (estudante/professor).",
                        "work_meeting": "Predomina trabalho técnico (projeto, código, dados, decisões).",
                        "general_conversation": "Conversa geral sem foco claro.",
                        "unknown": "Pouca evidência; ainda não dá para afirmar.",
                    },
                },
                "session_confidence": {
                    "type": "number",
                    "instructions": "Confiança (0..1) do modelo no tipo de sessão escolhido.",
                    "min": 0.0,
                    "max": 1.0,
                },
            },
        }
        request = urllib.request.Request(
            "https://openrouter.ai/api/alpha/decisions",
            data=json.dumps(payload).encode(),
            headers=_headers(),
        )
        try:
            with urllib.request.urlopen(request, timeout=int(os.getenv("OPENROUTER_TIMEOUT", "12"))) as response:
                data = json.loads(response.read().decode())
            answers = data["answers"]
            a = answers.get("action") or {}
            action = a.get("choice")
            a_conf = _num(a.get("confidence"), 0.0)
            priority = _num(answers.get("priority"), a_conf)
            need = _bool(answers.get("need"), True)
            s = answers.get("session_type") or {}
            stype = s.get("choice")
            s_conf = _num(s.get("confidence") if s.get("confidence") is not None else s.get("value"), 0.0)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError,
                KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            if attempt == 0 and not isinstance(error, (TypeError, ValueError, json.JSONDecodeError)):
                continue
            return None, None, f"jev-error: {error}"
        if action not in ("ignore", "lang_coach", "work_copilot"):
            return None, None, "jev-invalid-action"
        if stype not in SESSION_TYPES or not (0.0 <= s_conf <= 1.0):
            return None, None, "jev-session-invalid"
        from orchestrator import Decision  # type: ignore[import-not-found]
        dec = Decision(action=action, reason="jev", confidence=a_conf,
                       priority=priority, needs_insight=need)
        sess = SessionResult(type=stype, confidence=s_conf, action=action,
                             priority=priority, needs_insight=need,
                             source="jev", reason="jev-decisions")
        return dec, sess, "typesafe/jev-1.13"
    return None, None, "jev-retry-exhausted"


def classify_full(context_block: str, last_text: str) -> tuple[Decision | None, SessionResult | None, str]:
    """Retorna (Decision, SessionResult, modelo|motivo). None => rota/fallback local."""
    if not configured():
        return None, None, "no-key"
    if ROUTER_MODEL and ROUTER_MODEL.startswith("typesafe/jev"):
        return _jev_classify(context_block, last_text)
    user = (f"[CONTEXTO CONSOLIDADO]\n{context_block}\n\n"
            f"[ÚLTIMA FALA]\n{last_text}\n\nResponda o JSON de classificação.")
    text, used = chat([{"role": "system", "content": CLASSIFY_SYSTEM},
                       {"role": "user", "content": user}], max_tokens=140,
                      temperature=0.0, model=ROUTER_MODEL)
    if not text:
        return None, None, used
    raw = parse_classify(text)
    if raw is None:
        return None, None, f"{used} (parse-fail)"
    from orchestrator import Decision  # type: ignore[import-not-found]
    dec = Decision(action=raw.action or "ignore", reason=raw.reason or "classifier",
                   confidence=raw.priority, priority=raw.priority,
                   needs_insight=raw.needs_insight)
    return dec, None, used


def classify(context_block: str, last_text: str) -> tuple[Decision | None, str]:
    """Compat: (Decision|None, modelo|motivo). Session fica em classify_full."""
    dec, _sess, used = classify_full(context_block, last_text)
    return dec, used


def classifier_for():
    """Retorna callable (context, last_text)->(Decision|None) ou None (usa regras locais)."""
    mode = os.getenv("ROUTER_CLASSIFIER", "auto").lower()
    if mode == "local":
        return None
    if mode == "openrouter" or configured():
        return lambda ctx, last: classify(ctx, last)[0]
    return None
