"""Cliente OpenRouter (stdlib only): chat completions + cadeia de fallbacks free.

Env:
  OPENROUTER_API_KEY   obrigatória p/ chamadas reais
  OPENROUTER_MODEL     modelo primário (default: um free rápido)
  OPENROUTER_FALLBACKS lista separada por vírgula (default: outros frees)
  OPENROUTER_TIMEOUT   segundos (default 12)
  OPENROUTER_SITE_URL / OPENROUTER_APP_NAME (headers recomendados)

Sem chave: todas as funções retornam None -> orchestrator usa templates locais.
"""
from __future__ import annotations
import json
import os
import urllib.request

BASE = "https://openrouter.ai/api/v1"

# Cadeia default de modelos gratuitos (sufixo :free). A lista free gira;
# o teste live tenta em ordem até um responder 200. Ajuste via env.
DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
DEFAULT_FALLBACKS = [
    "google/gemma-3-27b-it:free",
    "qwen/qwen3-32b:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
    "deepseek/deepseek-chat-v3-0324:free",
]


def configured() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def model_chain() -> list[str]:
    primary = os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL).strip()
    raw = os.getenv("OPENROUTER_FALLBACKS", ",".join(DEFAULT_FALLBACKS))
    fallbacks = [m.strip() for m in raw.split(",") if m.strip()]
    chain = [primary] + [m for m in fallbacks if m != primary]
    return chain


def _headers() -> dict:
    h = {"Content-Type": "application/json",
         "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}"}
    if os.getenv("OPENROUTER_SITE_URL"):
        h["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "")
    h["X-Title"] = os.getenv("OPENROUTER_APP_NAME", "meeting-copilot")
    return h


def chat(messages: list[dict], max_tokens: int = 400, temperature: float = 0.4,
         timeout: int | None = None, model: str | None = None) -> tuple[str | None, str]:
    """Tenta cada modelo da cadeia. Retorna (texto|None, modelo_usado_ou_erro).

    `model` explícito sobrepõe o primário (ex.: classificador Jev via ROUTER_MODEL).
    """
    if not configured():
        return None, "no-key"
    timeout = timeout or int(os.getenv("OPENROUTER_TIMEOUT", "12"))
    chain = [model] + [m for m in model_chain() if m != model] if model else model_chain()
    last_err = "unknown"
    for mdl in chain:
        body = json.dumps({"model": mdl, "messages": messages,
                           "max_tokens": max_tokens, "temperature": temperature}).encode()
        req = urllib.request.Request(f"{BASE}/chat/completions", data=body, headers=_headers())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode())
            text = data["choices"][0]["message"]["content"].strip()
            return text, mdl
        except Exception as e:
            last_err = f"{mdl}: {e}"
            continue
    return None, last_err


COACH_SYSTEM = ("Você é um Language Coach pt->en em tempo real. Responda SOMENTE JSON válido, "
                "sem markdown, no formato: {\"say_this\":[3 frases curtas],\"vocab\":[4 itens],"
                "\"follow_up\":\"1 pergunta\",\"grammar_tip\":\"1 linha\"}. Frases prontas para falar.")
WORK_SYSTEM = ("Você é um copiloto técnico em reunião. Responda SOMENTE JSON válido, sem markdown, "
               "no formato: {\"summary\":\"1 linha\",\"to_check\":[4 itens],\"to_ask\":[2 perguntas],"
               "\"risks\":[2 riscos]}. Prático, sem enrolação.")


def enhance_card(kind: str, context_block: str, last_text: str) -> tuple[dict | None, str]:
    """Refina o card-template via OpenRouter. Retorna (card|None, modelo_ou_motivo)."""
    system = COACH_SYSTEM if kind == "lang_coach" else WORK_SYSTEM
    user = f"{context_block}\n\nÚltima fala: {last_text}\n\nGere o card JSON."
    text, used = chat([{"role": "system", "content": system},
                       {"role": "user", "content": user}], max_tokens=450)
    if not text:
        return None, used
    try:
        # tolera cercas de código caso o modelo desobedeça
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`").split("\n", 1)[1] if "\n" in t else t.strip("`")
            if t.lstrip().startswith("json"):
                t = t.lstrip()[4:]
        card = json.loads(t)
        assert isinstance(card, dict)
        return card, used
    except Exception as e:
        return None, f"{used} (parse-fail: {e})"


def list_free_models() -> list[str]:
    """Lista modelos gratuitos disponíveis (requer chave)."""
    if not configured():
        return []
    req = urllib.request.Request(f"{BASE}/models", headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
        return sorted(m["id"] for m in data.get("data", [])
                      if m.get("id", "").endswith(":free"))
    except Exception:
        return []
