# Meeting Copilot — MVP (Entrega 1)

Copiloto realtime para aulas de inglês e reuniões de trabalho. Local-first, open source, baixo custo.

## Status: Entrega 1 — esqueleto rodável sem hardware de áudio

O que funciona nesta entrega:
- `apps/api` — FastAPI + WebSocket `/ws` que recebe segmentos de transcrição e devolve `transcript` + `insight`
- `apps/orchestrator` — Context window + Router local (regras, sem LLM) + Language Coach / Work Copilot (templates + LLM opcional via API)
- `apps/stt` — serviço faster-whisper + Silero VAD (roda real se tiver mic/loopback, senão roda em modo mock)
- `apps/ui` — overlay web mínimo (preview do que será o Tauri always-on-top)
- `scripts/simulate_meeting.py` — simula uma aula/reunião sem precisar de áudio real, prova o pipeline end-to-end

## Quickstart (sem Docker, 2 min)

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
- [ ] Entrega 2: STT real loopback Linux/Windows + UI Tauri
- [ ] Entrega 3: router com modelo local (Ollama qwen 1.5B) + LLM real streaming
- [ ] Entrega 4: RAG Qdrant + search + memória entre reuniões
