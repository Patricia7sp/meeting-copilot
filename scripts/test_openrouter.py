"""Teste OpenRouter. Sem chave: valida fallback offline. Com chave: faz 1 chamada real.
Uso:
  python3 scripts/test_openrouter.py            # dry (sem custo)
  OPENROUTER_API_KEY=sk-or-... python3 scripts/test_openrouter.py --live
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../apps/orchestrator"))
import llm_openrouter as o

print(f"[openrouter] configured={o.configured()} chain={o.model_chain()}")
if "--live" in sys.argv:
    assert o.configured(), "defina OPENROUTER_API_KEY para --live"
    card, used = o.enhance_card("lang_coach", "[JANELA RECENTE]\nPROFESSOR: What did you do during the weekend?",
                                "What did you do during the weekend?")
    print(f"[openrouter] used={used}\ncard={card}")
    assert card and "say_this" in card, "resposta sem say_this"
    print("OK live: card EN refinado via OpenRouter.")
else:
    text, used = o.chat([{"role": "user", "content": "hi"}])
    assert text is None and used == "no-key", f"sem chave deveria retornar (None,'no-key'), veio {used}"
    print("OK dry: sem chave -> fallback para templates locais.")
