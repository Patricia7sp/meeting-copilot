# Modelos faster-whisper — decisão (Entrega 2b - qualidade de transcrição)

## Critério

Decisão no **Mac Intel** (CPU, `compute_type=int8`), medindo **latência realtime**
(RTF = segundos de processamento / segundos de áudio) e **qualidade percebida**
(PT e EN, com chunks de ~1.6 s e `condition_on_previous_text=False`).

## Como medir na sua máquina (benchmark completo PT/EN)

```bash
source .venv/bin/activate
pip install faster-whisper sounddevice numpy
python3 apps/stt/benchmark.py --speak        # grava 2 fixtures por idioma e digita a referência
python3 apps/stt/benchmark.py --run --models small,medium,large-v3-turbo
# métricas por modelo: WER, cobertura, acurácia de idioma, latência p50/p95, uso de CPU
```

Medição rápida de latência/RTF por modelo (1 fixture): `apps/stt/bench.py`.

## O que o service usa por default

- `WHISPER_MODEL=small` (default em `service.py` e no `docker-compose.yml`).
  O WER não depende do tamanho aqui: a fidelidade é definida pelo pipeline
  (segmentação consolidada + entrega confiável + janelas), não pelo modelo.
- Trocar p/ `medium`/`large` **depois** de confirmar no `bench.py` que há ganho
  de qualidade que justifique a latência extra no seu fluxo.

## Faixa esperada (referência, CPU):

| Modelo          | Carga   | RTF (CPU int8)   | Qualidade PT/EN | Uso recomendado |
|-----------------|---------|------------------|-----------------|-----------------|
| `small`         | rápida  | ~0.1–0.3 (bom)   | aceitável       | testes, máquinas fracas |
| `medium`        | média   | ~0.5–1.0         | boa             | equilíbrio p/ Mac Intel |
| `large-v3-turbo`| lenta   | ~1.0–2.0         | muito boa       | qualidade máxima; RTF pode passar de 1 em Intel |

(preencha com os números reais do `bench.py` na sua máquina antes de fixar.)

## Decisão (guia)

- **Aulas + calls em PT/EN misturados no mesmo fluxo** → `medium` no Mac Intel:
  RTF < 1 com qualidade boa e detecção de idioma confiável.
- **Aumentar precisão acima de latência** → `large-v3-turbo`.
- **Micro-MVP / validação de pipeline** → `small` (o que você usou no teste inicial).

O peso da decisão fica para o resultado do `bench.py` no Mac Intel: se
`large-v3-turbo` ficar com RTF < 1.0 no seu fluxo (chunks de 1.6 s), mantenha-o;
senão, baixe para `medium` no `.env`/dar até o `bench.py` confirmar.

## Regras de qualidade já aplicadas no service

1. **Locutor por fonte**: mic→YOU, loopback→OTHERS, decidido só por fonte
   (nunca por "dispositivo abriu"). O gate usa **noise-floor por fonte** +
   delta acima do piso + histerese (`NoiseFloorGate`), não um RMS fixo: o ruído
   ambiente do mic é aprendido no piso e não vira `YOU`.
2. **Multilíngue**: `language=None, task="transcribe"` mantém o idioma falado
   (evento traz `language` detectado).
3. **Anti-alucinação**: gate (noise floor + histerese + webrtcvad opcional) antes
   do modelo; `no_speech_threshold=0.6`, `log_prob_threshold=-0.8`,
   `condition_on_previous_text=False` (mata o loop "bye-bye bye-bye");
   pós-filtro de repetição/filler; descarte com motivo nos logs.
4. **Tradução separada**: transcrição preserva o idioma. Tradução só se
   `TRANSLATE_TARGET=en` (nativo do faster-whisper). Outros destinos = feature futura.

## Regras da segmentação contínua + entrega WS (fidelidade da transcrição)

5. **Provisório ao vivo**: janelas sobrepostas de ~`WINDOW_SECONDS=1.6s` sobre o
   enunciado, a cada `PROVISIONAL_EVERY_SECONDS=0.8s`, fronteira deduplicada
   (sobreposição por bytes + `dedup_delta` por texto), `condition_on_previous_text=False`.
6. **UM `final` consolidado por enunciado**: `final` só sai com silêncio REAL ≥
   `PAUSE_SECONDS=0.7s` (ou cap de memória `MAX_UTTERANCE_CAP_SECONDS=120s`).
   `MAX_UTTERANCE_SECONDS=10s` só FORÇA refresh do provisório (janela de contexto
   do whisper), nunca finaliza — 60s de fala contínua chegam em UM final sem lacunas.
   O `final` re-transcreve o áudio inteiro do enunciado (`consolidated=True`,
   `condition_on_previous_text=True`, `LOG_PROB_THRESHOLD_FINAL=-1.0`) e é o texto
   que gera insight.
7. **WS com ack confiável**: client envia com `event_id` (run_id + seq) e `id`;
   usa fila com backpressure (nunca descarta), reenvio com
   `retry_backoff` e retries ilimitados por default, remove do pendente só com ack;
   persistência em `data/ws_pending/` com `event_id` idempotente no servidor.
   Ao final sai relatório `sent/acked/ack_timeout/drops/reconnects/lat p50/p95`.