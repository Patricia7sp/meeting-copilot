"""Resumo em Markdown por sessão encerrada (Feature 006).

O Jev `typesafe/jev-1.13` é usado SOMENTE para classificação (Feature 005),
nunca para gerar texto. A produção do resumo é uma extração local determinística
a partir das falas finais consolidadas e confiáveis, com seções por tipo:

- work_meeting          -> Decisões + Pendências e próximos passos
- english_lesson        -> Vocabulário + Correções + Frases sugeridas + Pontos de aprendizado
- general_conversation  -> resumo mínimo (sem capítulos)
- unknown               -> idem, sem transcrição quando não há conteúdo

Saída: `data/sessions/<sessao>_<tipo>.md` (escrita atômica, UTF-8 sem BOM).

Segurança: o módulo nunca recebe chave/token; a entrada já é filtrada (final,
consolidated, conf >= 0.5, !low, sem repetição) — e aqui é re-validada.
"""
from __future__ import annotations
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path


SESSION_TYPES = ("english_lesson", "work_meeting", "general_conversation", "unknown")
CONF_FLOOR = float(os.getenv("SESSION_MIN_CONFIDENCE", "0.5"))

TYPES_PT = {
    "work_meeting": "Reunião de trabalho",
    "english_lesson": "Aula de inglês",
    "general_conversation": "Conversa geral",
    "unknown": "Análise em curso",
}

_DECISION = re.compile(
    r"(decidimos|decidiu|decidido|definimos|combinamos|aprovamos|vamos (adotar|usar|fazer|migrar|criar)|"
    r"we decided|we agreed|we'll use|approved|let's go with|vai (usar|adotar|ficar|ser))", re.I)
_PENDING = re.compile(
    r"(pendencia|pendente|proximos passos|proximo passo|precisamos|need to|next step|next steps|"
    r"follow-up|vou verificar|vou resolver|aberto a|abertas?|vamos revisar|antes do|owner|dono|"
    r"deadline|prazo)", re.I)
_CORRECT = re.compile(
    r"(nao se (diz|fala)|o certo (e|eh)|em vez de|ao inves de|rather than|instead of|you mean|"
    r"you should (say|use)|nao e )", re.I)
_PHRASE = re.compile(
    r"(how (do|would) you say|how to say|natural way|diga|dizer|say this|frase)", re.I)
_GRAMMAR = re.compile(
    r"(grammar|past tense|present perfect|pronunciation|vocabulary|vocabulario|idiom|phrasal verb|"
    r"conjugacao)", re.I)
_VOCAB_LEAD = re.compile(
    r"(?:how (?:do|would) you (?:say|use)|what's the|you mean|means|say it as|in english)\s+"
    r"([A-Za-z][\w'-]*)", re.I)
_TOPIC_TRIM = re.compile(r"[^\w\u00C0-\u024F ]+", re.UNICODE)

STOPWORDS = {
    "about", "again", "all", "also", "and", "any", "are", "because", "been", "but", "can",
    "could", "did", "does", "for", "from", "had", "has", "have", "how", "into", "its",
    "just", "know", "like", "make", "more", "most", "not", "now", "only", "our", "out",
    "over", "said", "say", "some", "such", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "very", "was", "were", "what",
    "when", "where", "which", "while", "who", "will", "with", "would", "you", "your", "yes",
    "no", "so", "okay", "entao", "ai", "ne", "vamos", "todo", "toda", "mas", "que", "para",
    "com", "por", "uma", "um", "dos", "das", "nao", "mais", "muito", "sempre", "tambem",
    "depois", "quando", "porque", "esse", "essa", "este", "esta", "isso", "isto", "gente",
    "coisa", "coisas", "aqui", "onde", "meu", "minha", "nosso", "sobre", "entre",
}

SPEAKER_TYPES = {"YOU": "Você", "OTHERS": "Outros"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def valid_entry(entry: dict) -> bool:
    """Re-valida: só falas finais, consolidadas e confiáveis entram no relatório."""
    t = (entry.get("text") or "").strip()
    if not t:
        return False
    if entry.get("stage") and str(entry["stage"]) != "final":
        return False
    if entry.get("consolidated") is False:
        return False
    if entry.get("low_confidence"):
        return False
    conf = entry.get("confidence")
    if conf is not None:
        try:
            if float(conf) < CONF_FLOOR:
                return False
        except (TypeError, ValueError):
            return False
    return True


def clean_entries(raw: list[dict]) -> list[dict]:
    """Filtra por confiabilidade + remove repetição consecutiva (mesmo texto)."""
    out: list[dict] = []
    last = None
    for e in raw:
        if not valid_entry(e):
            continue
        text = " ".join(str(e.get("text", "")).split())
        if text == last:
            continue
        last = text
        out.append({"speaker": str(e.get("speaker") or "OTHERS"),
                    "text": text,
                    "confidence": e.get("confidence"),
                    "duration_ms": int(e.get("duration_ms") or 0)})
    return out


def _top_keywords(entries: list[dict], n: int = 5) -> list[str]:
    counts: dict[str, int] = {}
    for e in entries:
        for w in re.findall(r"[\w\u00C0-\u024F]{4,}",
                            _TOPIC_TRIM.sub(" ", e["text"]).lower()):
            if w in STOPWORDS:
                continue
            counts[w] = counts.get(w, 0) + 1
    order = sorted(counts, key=lambda w: (-counts[w], w))
    return order[:n]


def extract_all(entries: list[dict]) -> dict[str, list[str]]:
    """Extrai itens por seção, sem duplicar a mesma fala em mais de uma seção."""
    used: set[str] = set()
    sections: dict[str, list[str]] = {
        "decisions": [], "pending": [], "corrections": [],
        "phrases": [], "learning": [],
    }

    def keep(kind: str, text: str) -> None:
        if text in used:
            return
        used.add(text)
        sections[kind].append(text)

    for e in entries:
        t = e["text"]
        if _DECISION.search(t):
            keep("decisions", t)
        if _PENDING.search(t):
            keep("pending", t)
        if _CORRECT.search(t):
            keep("corrections", t)
        if _PHRASE.search(t):
            keep("phrases", t)
        if _GRAMMAR.search(t):
            keep("learning", t)
    return sections


def _vocab(entries: list[dict], limit: int = 12) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for e in entries:
        for m in re.finditer(r'["\']([^"\']{1,40})["\']', e["text"]):
            w = m.group(1).strip().lower()
            if w and w not in seen:
                seen.add(w)
                out.append(m.group(1).strip())
        m = _VOCAB_LEAD.search(e["text"])
        if m:
            w = m.group(1).strip(".,!?").lower()
            if w and w not in seen and w not in STOPWORDS:
                seen.add(w)
                out.append(m.group(1).strip(".,!?"))
        if len(out) >= limit:
            break
    return out[:limit]


def _executive_summary(profile: dict, entries: list[dict],
                       sections: dict[str, list[str]]) -> str:
    stype = profile.get("type", "unknown")
    label = TYPES_PT.get(stype, stype)
    n = len(entries)
    if n == 0:
        return ("Sessão classificada como {0}, mas sem falas finais consolidadas e "
                "confiáveis registradas até o encerramento.").format(label)
    kws = _top_keywords(entries)
    topics = ", ".join(kws) if kws else "sem tópicos dominantes"
    parts = [
        "Sessão classificada como {0} (confiança {1}% via {2}).".format(
            label, int(round((profile.get("confidence") or 0) * 100)),
            "Jev" if profile.get("source") == "jev" else "sinais locais"),
        "{0} fala(s) final(is) consolidada(s) considerada(s).".format(n),
        "Tópicos centrais: {0}.".format(topics),
    ]
    if stype == "work_meeting":
        parts.append("Decisões detectadas: {0}; pendências/próximos passos: {1}.".format(
            len(sections["decisions"]), len(sections["pending"])))
    elif stype == "english_lesson":
        parts.append("Vocabulário: {0} itens; correções: {1}; frases sugeridas: {2}; "
                     "pontos de aprendizado: {3}.".format(
            len(_vocab(entries)), len(sections["corrections"]),
            len(sections["phrases"]), len(sections["learning"])))
    return " ".join(parts)


def _bullets(items: list[str]) -> str:
    if not items:
        return ""
    return "\n".join("- " + re.sub(r"\s+", " ", it)[:220] for it in items)


def render_markdown(*, session_id: str, started_at: str, ended_at: str,
                    profile: dict, entries: list[dict]) -> str:
    stype = profile.get("type", "unknown")
    if stype not in SESSION_TYPES:
        stype = "unknown"
    tlabel = TYPES_PT.get(stype, stype)
    conf = int(round((profile.get("confidence") or 0) * 100))
    source_pt = "Jev" if profile.get("source") == "jev" else "heurística local"
    duration_s = float(sum(e.get("duration_ms", 0) for e in entries)) / 1000.0
    minutes = int(round(duration_s / 60.0)) if duration_s else 0
    sections = extract_all(entries)

    lines = [
        "# Resumo da Sessão",
        "",
        "- **Sessão:** {0}".format(session_id),
        "- **Início:** {0}".format(started_at),
        "- **Encerramento:** {0}".format(ended_at),
        "- **Tipo de sessão:** {0} (`{1}`)".format(tlabel, stype),
        "- **Confiança da classificação:** {0}% ({1})".format(conf, source_pt),
        "- **Duração estimada:** {0} min".format(minutes) if minutes else
        "- **Duração estimada:** < 1 min",
        "- **Falas consolidadas consideradas:** {0}".format(len(entries)),
        "",
        "## Resumo executivo",
        "",
        _executive_summary(profile, entries, sections),
        "",
    ]

    if entries:
        lines += ["## Pontos principais", "", _bullets([e["text"] for e in entries]), ""]

    if stype == "work_meeting":
        lines += ["## Decisões", ""]
        lines += [_bullets(sections["decisions"])
                  or "_Nenhuma decisão explícita detectada._", ""]
        lines += ["## Pendências e próximos passos", ""]
        lines += [_bullets(sections["pending"])
                  or "_Nenhuma pendência detectada._", ""]
    elif stype == "english_lesson":
        vocab = _vocab(entries)
        lines += ["## Vocabulário", ""]
        lines += [_bullets(vocab) if vocab else "_Nenhum item de vocabulário detectado._", ""]
        lines += ["## Correções", ""]
        lines += [_bullets(sections["corrections"])
                  or "_Nenhuma correção detectada._", ""]
        lines += ["## Frases sugeridas", ""]
        lines += [_bullets(sections["phrases"])
                  or "_Nenhuma sugestão de frase registrada._", ""]
        lines += ["## Pontos de aprendizado", ""]
        lines += [_bullets(sections["learning"]) or "_—._", ""]

    if entries and stype in SESSION_TYPES and stype != "unknown":
        lines += ["## Transcrição consolidada", ""]
        lines += [_bullets(["{0}: {1}".format(
            SPEAKER_TYPES.get(e["speaker"], e["speaker"]), e["text"]) for e in entries]), ""]

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def build_report(*, session_id: str, profile: dict,
                 entries: list[dict]) -> tuple[str, str]:
    """Conveniência: `(markdown, ended_at)` com timestamps padrão."""
    now = _now_iso()
    started_at = profile.get("started_at") or now
    md = render_markdown(session_id=session_id, started_at=started_at,
                         ended_at=now, profile=profile,
                         entries=entries if isinstance(entries, list) else [])
    return md, now


def save_report(directory: Path | str, session_id: str, profile: dict,
                md: str) -> Path:
    """Grava `<sessao>_<tipo>.md` com escrita atômica (tmp + os.replace)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stype = profile.get("type", "unknown")
    if stype not in SESSION_TYPES:
        stype = "unknown"
    safe = re.sub(r"[^0-9A-Za-z_\-]+", "_", session_id).strip("_") or "session"
    name = "{0}_{1}.md".format(safe, stype.replace("_", "-"))
    target = directory / name
    tmp = directory / ("." + name + ".tmp" + str(int(time.time() * 1000)))
    tmp.write_text(md, encoding="utf-8")
    os.replace(tmp, target)
    return target


def display_path(path: Path) -> str:
    """Caminho amigável: relativo ao repo quando possível."""
    try:
        return str(path.resolve().relative_to(Path(os.getcwd()).resolve()))
    except ValueError:
        try:
            repo = Path(__file__).resolve().parents[2]
            return str(path.resolve().relative_to(repo))
        except ValueError:
            return str(path)


def load_entries(jsonl: Path | str) -> list[dict]:
    """Lê `<session>.jsonl` e devolve as falas finais consolidadas confiáveis."""
    path = Path(jsonl)
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out.append({
                    "speaker": rec.get("speaker", "OTHERS"),
                    "text": rec.get("text", ""),
                    "confidence": rec.get("confidence"),
                    "duration_ms": rec.get("duration_ms"),
                    "stage": rec.get("stage"),
                    "consolidated": rec.get("consolidated",
                                             True if "consolidated" not in rec else rec["consolidated"]),
                    "low_confidence": rec.get("low_confidence"),
                })
    except OSError as e:
        print("[report] falha ao ler transcrição: {0}".format(e))
    return clean_entries(out)