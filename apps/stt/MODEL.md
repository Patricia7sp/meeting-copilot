# Modelos faster-whisper — decisão (Entrega 2b + Feature 005 - evidência)

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

[!] O parâmetro `--models` aceita também `distil-large-v3` (English-only).
Medição rápida de latência/RTF por modelo (1 fixture): `apps/stt/bench.py`.

## Benchmark por WAV do MESMO trecho de vídeo (YouTube → BlackHole) — Feature 007

O critério de aceite (Feature 007) é com áudio REAL do YouTube saindo no BlackHole:

```bash
python3 apps/audio/diag_capture.py --seconds 60 --source loopback --ref transcricao_oficial.txt
# grava data/bench/capture/loopback_<ts>.wav (mono 16k) + meta_<ts>.json +
# salva a transcrição oficial em --ref (ou cole depois)
python3 apps/stt/benchmark.py --wav data/bench/capture/loopback_<ts>.wav \
    --ref transcricao_oficial.txt --meta data/bench/capture/meta_<ts>.json [--save data/bench/BENCH_REPORT.md]
# roda small, medium, large-v3-turbo, distil-large-v3 e escolhe por WER/cobertura/p50/p95;
# o relatório embute a config de áudio (rate/canais/downmix/resample/RMS) de evidência.
```

Meta do `0.` critério: **WER ≤ 15%** e **cobertura de fala > 85%** no trecho.
Depois de escolher, fixe `WHISPER_MODEL=` no `.env`/`docker-compose` (STT roda no Mac).

## O que o service usa por default

- `WHISPER_MODEL=small` (default em `service.py` e no `docker-compose.yml`).
  O WER não depende do tamanho aqui: a fidelidade é definida pelo pipeline
  (segmentação consolidada + entrega confiável + janelas), não pelo modelo.
- Trocar p/ `medium`/`large`/`distil-large-v3` **depois** de o benchmark 007 com o
  WAV do BlackHole eleger o modelo por evidência (WER/cobertura/p50/p95 no Mac).
- Finais contínuos: `MAX_FINAL_SECONDS=12` (janela 8–15 s) com
  `FINAL_OVERLAP_SECONDS=1` de sobreposição p/ continuar a aula; pausa real ≥ 0.7 s
  continua encerrando o enunciado.
- Finais abaixo de `MIN_FINAL_CONFIDENCE` (servidor, env `MIN_FINAL_CONFIDENCE=0.35`)
  não entram no Jev/classificação/recomendações (só o resumo oficial filtra pelo `CONF_FLOOR`).

## Faixa esperada (referência, CPU):

| Modelo          | Carga   | RTF (CPU int8)   | Qualidade PT/EN | Uso recomendado |
|-----------------|---------|------------------|-----------------|-----------------|
| `small`         | rápida  | ~0.1–0.3 (bom)   | aceitável       | testes, máquinas fracas |
| `medium`        | média   | ~0.5–1.0         | boa             | equilíbrio p/ Mac Intel |
| `large-v3-turbo`| lenta   | ~1.0–2.0         | muito boa       | qualidade máxima; RTF pode passar de 1 em Intel |

(preencha com os números reais do `bench.py` na sua máquina antes de fixar.)

## Promoção de modelo SOMENTE com evidência (Feature 005)

`small` é o default até o benchmark com áudio real PT-BR/EN provar o contrário.
**Não promova `medium`/`large-v3-turbo` por "sentimento"**: rode o benchmark e
compare. Meta de qualidade para trocar de modelo: **cobertura ≥ 95%**,
**provisório ≤ 3s** e **final ≤ 5s** — senão, mantém `small`. As gravações reais
(PT-BR e EN com referência textual) exigem Mac + `faster-whisper`/`sounddevice`
(`apps/stt/benchmark.py --speak`); no servidor isso está ⏳ pendente. O
`benchmark.py --mkdir` já prepara `data/bench/{pt,en}/` p/ quando houver áudio.

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
4b. **Piso por fonte**: `INITIAL_FLOOR_DB_MIC=-46` (mic) e `INITIAL_FLOOR_DB_LOOPBACK=-58`
   (loopback/BlackHole — sinal mais limpo que o mic; um único piso para as duas fontes
   faria o loopback nunca chegar a falar). O `NoiseFloorGate` escolhe o piso inicial
   pela fonte (`mic` vs `loopback`).

## Regras da segmentação contínua + entrega WS (fidelidade da transcrição)

5. **Provisório ao vivo**: janelas sobrepostas de ~`WINDOW_SECONDS=1.6s` sobre o
   enunciado, a cada `PROVISIONAL_EVERY_SECONDS=0.8s`, fronteira deduplicada
   (sobreposição por bytes + `dedup_delta` por texto), `condition_on_previous_text=False`.
6. **UM `final` consolidado por enunciado**: `final` sai com silêncio REAL ≥
   `PAUSE_SECONDS=0.7s` (encerra todo o enunciado), ou como **final contínuo
   (rolling)** quando uma fala passa de `MAX_FINAL_SECONDS=12s` (janela 8–15 s):
   finaliza a janela e planta o próximo enunciado com `FINAL_OVERLAP_SECONDS` de
   cauda p/ continuidade — p/ aula de inglês via BlackHole não se perde fala e nunca
   se consolida frase longa de fragmentos ruins. `MAX_UTTERANCE_SECONDS=10s` só
   FORÇA refresh do provisório (janela de contexto do whisper), nunca finaliza;
   `MAX_UTTERANCE_CAP_SECONDS=120s` é o cap de memória.
   O `final` re-transcreve o áudio do enunciado/janela (`consolidated=True`,
   `condition_on_previous_text=True`, `LOG_PROB_THRESHOLD_FINAL=-1.0`) e é o texto
   que gera insight. Cada final traz `rolling`/`overlap_seconds` para a UI.
7. **WS com ack confiável**: client envia com `event_id` (run_id + seq) e `id`;
   usa fila com backpressure (nunca descarta), reenvio com
   `retry_backoff` e retries ilimitados por default, remove do pendente só com ack;
   persistência em `data/ws_pending/` com `event_id` idempotente no servidor.
   Ao final sai relatório `sent/acked/ack_timeout/drops/reconnects/lat p50/p95`.
8. **Atribuição por origem (não energia) no servidor também**: `source` mic→YOU,
   loopback→OTHERS; o servidor deriva o locutor do `source` mesmo se o client
   mandar `speaker` ocupado.
9. **Gate de confiança do servidor (Feature 007)**: finais com `low_confidence` ou
   `confidence < MIN_FINAL_CONFIDENCE` viram `status=em_revisao` e NÃO chamam o
   Jev/classificação/recomendações (orchestrator já noopa < 0.5); o resumo oficial
   segue filtrando pelo `CONF_FLOOR` do report.
10. **UI 3 estados (Feature 007)**: `capturado` (provisório, card tracejado),
    `em_revisao` (final abaixo do limiar, vermelho tracejado), `confirmado`
    (final confiável, card sólido). Provisório nunca parece final.