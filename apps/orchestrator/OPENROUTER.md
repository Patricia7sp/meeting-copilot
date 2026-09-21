# OpenRouter — como usar (foco em frees)

1. Crie a conta em https://openrouter.ai e gere a chave em Keys.
2. Exporte no shell (nunca commite a chave):
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   export OPENROUTER_MODEL="meta-llama/llama-3.3-70b-instruct:free"
   # opcional: cadeia de fallback
   export OPENROUTER_FALLBACKS="google/gemma-3-27b-it:free,qwen/qwen3-32b:free,mistralai/mistral-small-3.1-24b-instruct:free"
   ```
3. Teste:
   ```bash
   python3 scripts/test_openrouter.py --live
   ```
4. Rode a API com LLM ativo:
   ```bash
   LLM_PROVIDER=openrouter ORCHESTRATOR_MODE=auto ./scripts/run_local.sh
   ```

## Comportamento

- **Sem chave:** tudo funciona offline com templates locais (zero custo, zero rede).
- **Com chave:** cada insight tenta refinar o card-template via OpenRouter (timeout ~12s, mas na prática 1-3s nos frees rápidos). Falha/parsing ruim → mantém template local, sem quebrar a UI.
- **Cadeia free:** se o primário der 429/lotado, tenta o próximo automaticamente. A lista free gira — veja a atual em https://openrouter.ai/models?max_price=0 ou rode `list_free_models()`.
- **Custo:** modelos `:free` não cobram; se um dia usar um pago, o limite vem da sua conta OpenRouter, não do app.

## Modelos free sugeridos (cadeia default)

1. `meta-llama/llama-3.3-70b-instruct:free` — bom equilíbrio raciocínio/latência
2. `google/gemma-3-27b-it:free` — rápido, bom p/ Coach
3. `qwen/qwen3-32b:free` — bom multilíngue pt/en
4. `mistralai/mistral-small-3.1-24b-instruct:free` — rápido, técnico ok
5. `deepseek/deepseek-chat-v3-0324:free` — raciocínio mais forte, às vezes mais lento
