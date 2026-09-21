# meeting-copilot — AGENTS.md

Este projeto usa **Reversa** (modo greenfield + forward) para conduzir da ideia ao código.

## Comandos Reversa disponíveis

- `reversa-new` — refinar ideia em PRD/specs (já temos `_reversa_sdd/` inicial via Entrega 1)
- `reversa-forward` — evoluir uma feature por vez a partir das specs (`_reversa_forward/`)
- `reversa-clarify` / `reversa-plan` / `reversa-to-do` / `reversa-coding` / `reversa-sync` — pipeline forward

Skills instaladas em `.agents/skills/` (espelho do framework Reversa).
Estado em `.reversa/state.json`. Specs em `_reversa_sdd/`, features em `_reversa_forward/`.

## Convenções do projeto

- Backend Python 3.11, FastAPI + WS; STT local faster-whisper; UI Tauri (preview web em `apps/ui/`)
- Pipeline: audio -> VAD -> STT -> context window -> router -> coach/work -> UI
- Privacidade: áudio só em RAM; transcrição em `data/sessions/` (permitido pelo dono p/ trabalho + aulas)
- Commits: `feat:`, `fix:`, `docs:`; MVP por entregas discutidas com o dono
