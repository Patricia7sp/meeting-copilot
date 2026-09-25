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
from llm_openrouter import BASE, _headers, chat, configured  # type: ignore

if TYPE_CHECKING:
    from orchestrator import Decision  # noqa: F401


ROUTER_MODEL = (os.getenv("ROUTER_MODEL", "") or None)


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


def _jev_classify(context_block: str, last_text: str) -> tuple[Decision | None, str]:
    """Executa o Jev pelo endpoint de decisões estruturadas do OpenRouter."""
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
            }
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
        answer = data["answers"]["action"]
        action = answer["choice"]
        confidence = float(answer["confidence"])
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return None, f"jev-error: {error}"
    if action not in ("ignore", "lang_coach", "work_copilot"):
        return None, "jev-invalid-action"
    from orchestrator import Decision  # type: ignore[import-not-found]
    return Decision(
        action=action,
        reason="jev",
        confidence=confidence,
        priority=confidence,
        needs_insight=action != "ignore",
    ), "typesafe/jev-1.13"


def classify(context_block: str, last_text: str) -> tuple[Decision | None, str]:
    """Retorna (Decision|None, modelo|motivo). None => usa rota local."""
    if not configured():
        return None, "no-key"
    if ROUTER_MODEL and ROUTER_MODEL.startswith("typesafe/jev"):
        return _jev_classify(context_block, last_text)
    user = (f"[CONTEXTO CONSOLIDADO]\n{context_block}\n\n"
            f"[ÚLTIMA FALA]\n{last_text}\n\nResponda o JSON de classificação.")
    text, used = chat([{"role": "system", "content": CLASSIFY_SYSTEM},
                       {"role": "user", "content": user}], max_tokens=140,
                      temperature=0.0, model=ROUTER_MODEL)
    if not text:
        return None, used
    raw = parse_classify(text)
    if raw is None:
        return None, f"{used} (parse-fail)"
    from orchestrator import Decision  # type: ignore[import-not-found]
    dec = Decision(action=raw.action or "ignore", reason=raw.reason or "classifier",
                   confidence=raw.priority, priority=raw.priority,
                   needs_insight=raw.needs_insight)
    return dec, used


def classifier_for():
    """Retorna callable (context, last_text)->(Decision|None) ou None (usa regras locais)."""
    mode = os.getenv("ROUTER_CLASSIFIER", "auto").lower()
    if mode == "local":
        return None
    if mode == "openrouter" or configured():
        return lambda ctx, last: classify(ctx, last)[0]
    return None
