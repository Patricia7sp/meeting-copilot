# Como o Jev decide o tipo de sessão

Sessão persiste num perfil `{type, confidence, source, updated_at}`. O modelo
`typesafe/jev-1.13` (via OpenRouter, endpoint estruturado `decisions`) decide o tipo
**a partir do contexto consolidado**, nunca de uma frase isolada. Quando não há chave,
timeout, resposta inválida ou schema fora de faixa, o mesmo contrato é atendido pelo
**fallback local** (`source="local"`).

## Tipos

| Tipo | Significado | Sinal dominante |
|---|---|---|
| `work_meeting` | Reunião de trabalho (dados, engenharia de dados, arquitetura, banco, APIs, pipelines, deploy, produto, decisões técnicas) | Vocabulário técnico: join, pipeline, partition, cluster, api, deploy, custo, arquitetura… |
| `english_lesson` | Aula/prática/conversa de inglês (estudante-professor, exercícios) | Sinal de aprendizagem: how do you say, grammar, vocabulary, pronunciation, teacher, weekend… |
| `general_conversation` | Conversa comum sem foco claro | Nenhum dos dois sinais domina |
| `unknown` | Ainda sem evidência suficiente | Menos de 3 falas confiáveis / ~30s de conteúdo útil |

## Gatilhos (quando chamar o modelo)

A classificação começa após **contexto suficiente** e é **reavaliada periodicamente**:

- ≥ `SESSION_MIN_FINALS=3` falas finais confiáveis, **ou**
- ≥ `SESSION_CONTENT_SECONDS=30` de conteúdo útil, **ou**
- mudança material de assunto (2 falas finais consecutivas com um tipo novo divergente do atual).

Só falas `final` consolidadas, `confidence >= 0.5`, `low_confidence=false` e com conteúdo
(≥3 palavras) alimentam a decisão. Provisórios nunca entram.

## Uma chamada, 5 perguntas

O Jev responde num único request (`apps/orchestrator/classifier.py::_jev_classify`) com
`session_type`, `session_confidence`, `action`, `priority`, `needs_insight`. O contexto
vai numa **janela deslizante limitada** (até `SESSION_WINDOW=8` falas + resumo curto),
nunca histórico ilimitado.

## Estabilidade (anti-oscilação)

O perfil **não** troca por uma frase ambígua. Uma nova classificação só vira o tipo atual
quando:

- confiança nova > confiança atual + `JEV_TRUST_DELTA=0.05` (resposta Jev), **ou**
- o MESMO tipo novo aparece em 2 avaliações consecutivas (mudança sustentada).

Isso dá os comportamentos dos testes 4 e 5 do dono: tema isolado não muda a aula/reunião;
mudança sustentada atualiza.

## Falha do Jev

Qualquer erro (sem chave, timeout, retry 1 único de rede, JSON inválido, enum/schema fora
de faixa) cai no **fallback local** (`session.py::_local_result`): contagem de sinais
técnicos vs. de aprendizado sobre a janela, com confiabilidade heurística
(`english_lesson/work_meeting` 0.7, `general_conversation` 0.5, `unknown` 0.3) e
`source="local"`. A transcrição nunca é bloqueada.

## Classificação → roteamento → resumo

- A classificação **não dispara insight** por si só; apenas orienta o roteador
  (`work_copilot`/`lang_coach`) e a UI (`Aula de inglês · 87%`).
- No encerramento (`/end_session`), o **resumo em Markdown** (`report.py`) usa o tipo
  final para escolher os capítulos (decisões/pendências para reunião;
  vocabulário/correções/frases/pontos de aprendizado para aula) a partir das falas
  consolidadas — o Jev não gera texto longo.