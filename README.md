# Meeting Copilot — MVP (Entrega 1)

Copiloto realtime para aulas de inglês e reuniões de trabalho. Local-first, open source, baixo custo.

## Status: Entrega 2d — fidelidade da transcrição (segmentação consolidada + entrega WS sem perda) ✅

Em cima da Entrega 2c (duas etapas, WS com ack, benchmark PT/EN e router Jev):

- **Segmentação contínua por fonte** (`apps/stt/stream.py` → `Segmenter`):
  - gate por fonte com **noise-floor calibrado + delta + histerese** (`NoiseFloorGate`,
    `apps/audio/capture.py`): mic→`YOU`/loopback→`OTHERS`; o ruído ambiente do mic é
    aprendido no piso e não vira `YOU`.
  - `provisional` ao vivo: janelas sobrepostas de ~1.6s sobre o enunciado a cada
    ~0.8s, fronteira deduplicada (`dedup_delta`), `condition_on_previous_text=False`.
  - **UM `final` consolidado por enunciado**: só sai com silêncio real ≥
    `PAUSE_SECONDS=0.7s` (ou cap de memória de 120s). `MAX_UTTERANCE_SECONDS=10s`
    força refresh do provisório sem jamais finalizar → 60s de fala contínua chegam
    em UM final sem lacunas. O final re-transcreve o áudio inteiro e é o que gera insight.
- **WebSocket com entrega confiável** (`apps/stt/wsclient.py`):
  fila com backpressure (nunca descarta), `event_id` (run_id+seq) idempotente no
  servidor, ack só remove o pendente; **retries ilimitados com backoff**; persistência
  em `data/ws_pending/` recupera não-acknados após reinício; `metrics_event()`
  espelhada na UI. Relatório `sent/acked/ack_timeout/drops/reconnects/lat p50/p95`.
- **Insights só em `stage=final` && `consolidated=true` && evento não visto**
  (`apps/api/main.py`): provisório e retransmissão nunca duplicam; janela usa só finais.
- **Modelo default `small`** (WER não dependo do tamanho: a fidelidade está no pipeline).
- **Benchmark PT-BR/EN** (`apps/stt/benchmark.py`): WER próprio (Levenshtein),
  cobertura, acurácia de idioma, latência p50/p95 e uso de CPU, com modo
  `--speak` (grava + digita a referência).

Quickstart client (Mac):
```bash
source .venv/bin/activate          # ~/meeting-copilot-client/client
export API_WS_URL="ws://100.87.25.101:8000/ws"
export WHISPER_MODEL=small        # default já é small; medium/large só com ganho no bench
export MIC_DEVICE="Microfone (MacBook Pro)"
export TRANSLATE_TARGET="en"       # opcional — tradução separada (só na etapa final)
export ROUTER_MODEL="openrouter/je...:latest"  # opcional — classificador Jev (roda sem chave via regras locais)
MOCK_MODE=false python3 apps/stt/service.py
```

Primeira carga do modelo: 30–90s; depois tempo real. A UI mostra o texto
**provisório ao vivo** por enunciado; o `final` da fala consolida no MESMO card
(sem cards duplicados), com badges de idioma, confiança, baixa confiança e a
linha de métricas do WS.

Validação:
```bash
python3 scripts/test_pipeline.py          # pipeline puro
python3 apps/stt/test_logic.py            # lógica 2b + gate noise-floor/histerese
python3 apps/stt/test_segment.py          # segmentação contínua (silêncio, 60s, fontes, borda)
python3 apps/stt/test_wsclient.py         # WS: ack, retry, reconexão, persistência
python3 apps/orchestrator/test_orchestrator.py
python3 scripts/test_ws_e2e.py            # ack por id, event_id dedup, consolidated, métricas
python3 scripts/test_stack_e2e.py         # API+simulação via HTTP: gating consolidated/event_id, /ui, persistência
```

Decisão de modelo e medição: `apps/stt/MODEL.md` + `python3 apps/stt/benchmark.py`.

## Estrutura

```bash
cd /home/paty7sp/projetos/meeting-copilot
# 1. Testar pipeline puro (sem dependências externas)
python3 scripts/test_pipeline.py

# 2. Simular reunião completa (precisa: pip install fastapi uvicorn)
./scripts/run_local.sh
# abre http://localhost:8000/ui em outro terminal: python3 scripts/simulate_meeting.py --mode english
```

## Quickstart (Docker — quando quiser portátil)

```bash
docker compose up --build
# api em :8000, ollama em :11434, qdrant em :6333
```

## Estrutura

```text
apps/api          FastAPI WS hub
apps/stt          captura + VAD + faster-whisper
apps/orchestrator router + coach + work + context
apps/ui           overlay web (vira Tauri na V1)
packages/context  janela deslizante + resumo rolante
config/           settings.yaml (modo, idioma, LLM)
data/sessions/    transcrições persistidas (.md + .jsonl) — permitido por você
scripts/          simuladores e runners
```

## Modos

- `english` — Language Coach: frases prontas, vocab, follow-up, correção curta
- `work` — Work Copilot: resumo 1 linha, riscos, perguntas, checklist técnico
- `auto` — router decide por mensagem (default no MVP)

## Privacidade (liberado por você p/ trabalho + aulas)

- Áudio cru: só em RAM, nunca salvo por default
- Texto: salvo em `data/sessions/` para memória/RAG futuro
- Indicador `● TRANSCREVENDO` obrigatório na UI
- `Local Only` toggle: se ON, não chama LLM externo (usa templates locais)

## Roadmap

- [x] Entrega 1: esqueleto + pipeline simulado
- [x] Entrega 2: STT real loopback Linux/Mac (captura unificada)
- [x] Entrega 2b: STT multilíngue + locutor por energia + anti-alucinação
- [x] Entrega 2c: duas etapas (provisório/final) + WS com ack/métricas + benchmark PT/EN + router Jev
- [x] Entrega 2d: fidelidade — segmentação consolidada (UM final por fala) + gate noise-floor + WS sem perda (retry/persistência/event_id)
- [ ] Entrega 3: router com modelo local (Ollama qwen 1.5B) + LLM real streaming
- [ ] Entrega 4: RAG Qdrant + search + memória entre reuniões
