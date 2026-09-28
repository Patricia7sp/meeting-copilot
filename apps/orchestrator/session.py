"""Perfil persistente de sessão e candidatura de classificação (Jev/local).

Responsabilidades (Requirements 005, FR-3):
- Não classificar provisórios; só finais consolidados confiáveis (conf >= limiar, não-low).
- Gatilhos: >= SESSION_MIN_FINALS falas finais qualificadas OU >= SESSION_CONTENT_SECONDS de
  conteúdo útil OU mudança sustentada de assunto (2 finais consecutivos com tipo local
  divergente do atual).
- Janela limitada: maximo SESSION_WINDOW finais consolidados + resumo curto dos antigos.
- Estabilidade (FR-3.8): troca de tipo só com confiança nova > atual (+ JEV_TRUST_DELTA) OU
  mesmo tipo novo em 2 avaliações consecutivas.
- Dedup de chamada Jev (FR-6.3) por fingerprint da janela; limite JEV_MAX_CALLS e intervalo
  mínimo JEV_MIN_INTERVAL (FR-6.4).
- Fallback local derivado por sinais das falas (`source="local"`); nunca bloqueia o pipeline.
- Persistência atômica em `<session_dir>/<session_id>_session.json`.
"""
from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SESSION_TYPES = ("english_lesson", "work_meeting", "general_conversation", "unknown")
SESSION_MIN_FINALS = int(os.getenv("SESSION_MIN_FINALS", "3"))
SESSION_WINDOW = int(os.getenv("SESSION_WINDOW", "8"))
SESSION_CONTENT_SECONDS = float(os.getenv("SESSION_CONTENT_SECONDS", "30.0"))
JEV_MAX_CALLS = int(os.getenv("JEV_MAX_CALLS", "200"))
JEV_MIN_INTERVAL = float(os.getenv("JEV_MIN_INTERVAL", "4.0"))
JEV_TRUST_DELTA = float(os.getenv("JEV_TRUST_DELTA", "0.05"))
CONF_THRESHOLD = float(os.getenv("SESSION_MIN_CONFIDENCE", "0.5"))
LOCAL_CONF = {"english_lesson": 0.7, "work_meeting": 0.7,
              "general_conversation": 0.5, "unknown": 0.3}

_TECH = re.compile(
    r"(bigquery|join|pipeline|partition|cluster|materialized view|bytes|cardinalidade|"
    r"docker|kubernetes|python|sql|api|latency|erro|bug|deploy|arquitetura|etl|scanne|"
    r"cache|query|job|data|tabela|banco)", re.IGNORECASE)
_LEARN = re.compile(
    r"(weekend|what did you|how do you say|how to say|grammar|vocabulary|pronunciation|"
    r"idiom|phrasal verb|past tense|present perfect|teacher|professor|lesson|exercise|"
    r"phrases|meaning|mean in english)", re.IGNORECASE)


@dataclass
class SessionResult:
    """Resultado estruturado de uma avaliação de sessão (Jev ou uma fonte externa)."""
    type: str = "unknown"
    confidence: float = 0.0
    action: str = "ignore"
    priority: float = 0.5
    needs_insight: bool = True
    source: str = "jev"          # jev | local
    reason: str = ""

    def to_dict(self) -> dict:
        return {"type": self.type, "confidence": round(self.confidence, 3),
                "source": self.source, "reason": self.reason}


def _local_type_of(text: str) -> str:
    if _LEARN.search(text) and not _TECH.search(text):
        return "english_lesson"
    if _TECH.search(text):
        return "work_meeting"
    return "general_conversation"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_session_dir() -> Path:
    env = os.getenv("DATA_DIR")
    if env:
        d = Path(env)
        if "sessions" not in str(d).lower():
            d = d / "sessions"
        return d
    return Path(__file__).resolve().parents[2] / "data" / "sessions"


class SessionTracker:
    """Mantém o perfil de sessão. Não depende do Orchestrator nem do classifier."""

    def __init__(self, *, session_id: str | None = None, session_dir: Path | str | None = None):
        self.session_id = session_id or os.getenv("SESSION_ID", "session")
        env_dir = os.getenv("SESSION_DIR", "")
        self.dir = Path(session_dir) if session_dir else (Path(env_dir) if env_dir else _default_session_dir())
        self.directory = self.dir
        self.window: list[dict] = []        # finais consolidados qualificados (max SESSION_WINDOW)
        self._summary: list[str] = []       # resumo extrativo dos antigos
        self._n_finals = 0
        self._content_seconds = 0.0
        self._n_calls = 0
        self._last_call = 0.0
        self._last_fp: str | None = None
        self._trend = 0                     # finais consecutivos com tipo local divergente
        self._changed = False               # sinal p/ session_event

        stype = os.getenv("SESSION_INITIAL_TYPE", "")
        self.profile = {
            "type": stype if stype in SESSION_TYPES else "unknown",
            "confidence": LOCAL_CONF["unknown"],
            "source": "local",
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "n_finals": 0,
            "classifications": 0,
        }
        self._candidate: dict | None = None   # {"type","confidence","source","seen":int}
        self._pending_changed = False
        self._load()

    # ---------- entrada a partir de um final qualificado ----------

    def observe_final(self, text: str, *, speaker: str = "OTHERS",
                      confidence: float | None = None, duration_ms: int | None = None,
                      classification: SessionResult | None = None) -> dict:
        """Adiciona um final confiável à janela e pode reavaliar/confirmar o perfil."""
        self._changed = False
        conf = confidence if confidence is not None else CONF_THRESHOLD
        entry = {"speaker": speaker, "text": text, "confidence": conf,
                 "duration_ms": duration_ms or 0}
        self._n_finals += 1
        self._content_seconds += entry["duration_ms"] / 1000.0
        self._append(entry)
        # primeiro gatilho substantivo ou periódico ou mudança de assunto
        due = self._due()
        if due:
            self._evaluate(entry["text"], classification)
        else:
            self._track_trend(_local_type_of(entry["text"]))
        self._persist()
        return {"changed": self._changed, "profile": self.to_dict()}

    def should_call(self, ctx_fingerprint: str | None = None) -> bool:
        """Política de custo/dedup p/ chamar o Jev (FR-6.3/6.4)."""
        if not self._due():
            return False
        if self._n_calls >= JEV_MAX_CALLS:
            return False
        if time.monotonic() - self._last_call < JEV_MIN_INTERVAL:
            return False
        if ctx_fingerprint is not None and ctx_fingerprint == self._last_fp:
            return False
        return True

    def on_call_done(self, ctx_fingerprint: str | None) -> None:
        self._n_calls += 1          # tenta contar, sucesso ou falha (limite FR-6.4)
        self._last_call = time.monotonic()
        if ctx_fingerprint is not None:
            self._last_fp = ctx_fingerprint

    # ---------- avaliação ----------

    def _due(self) -> bool:
        return (self._n_finals >= SESSION_MIN_FINALS
                or self._content_seconds >= SESSION_CONTENT_SECONDS
                or self._trend >= 2)

    def _local_result(self) -> SessionResult:
        counts: dict[str, int] = {}
        for entry in self.window:
            t = _local_type_of(entry["text"])
            counts[t] = counts.get(t, 0) + 1
        most = max(counts, key=lambda k: counts[k]) if counts else "unknown"
        if most == "unknown" and not counts:
            most = "general_conversation"
        # Mudança material (Q4/Q5): os 2 últimos finais convergem para um tipo diferente
        # do atual -> candidato com esse tipo, sem esperar a maioria da janela virar.
        tail = [_local_type_of(e["text"]) for e in self.window[-2:]]
        if (len(tail) == 2 and tail[0] == tail[1] and tail[0] != self.profile["type"]
                and tail[0] != most):
            return SessionResult(type=tail[0], confidence=LOCAL_CONF.get(tail[0], 0.7),
                                 action="ignore", source="local",
                                 reason="material-change")
        conf = LOCAL_CONF.get(most, 0.5) * min(1.0, 0.5 + 0.12 * len(self.window))
        return SessionResult(type=most, confidence=round(min(conf, 0.95), 3),
                             action="ignore", source="local", reason="local-signals")

    def _evaluate(self, last_text: str, classification: SessionResult | None) -> None:
        cand = classification if classification is not None else self._local_result()
        cur = self.profile
        if cand.type == cur["type"]:
            self._candidate = None
            self._trend = 0
            return
        if self._candidate and self._candidate["type"] == cand.type:
            self._candidate["seen"] += 1
            self._candidate["confidence"] = max(self._candidate["confidence"], cand.confidence)
        else:
            self._candidate = {"type": cand.type, "confidence": cand.confidence,
                               "source": cand.source, "seen": 1}
        if cand.confidence > cur["confidence"] + JEV_TRUST_DELTA or self._candidate["seen"] >= 2:
            self._commit(cand)
            self._candidate = None
            self._trend = 0

    def _track_trend(self, local_type: str) -> None:
        if local_type != self.profile["type"] and self.profile["type"] != "unknown":
            self._trend += 1
        else:
            self._trend = 0

    def _commit(self, cand: SessionResult) -> None:
        self.profile = {
            "type": cand.type,
            "confidence": round(min(max(cand.confidence, 0.0), 1.0), 3),
            "source": cand.source,
            "started_at": self.profile.get("started_at") or _now_iso(),
            "updated_at": _now_iso(),
            "n_finals": self._n_finals,
            "classifications": self._n_calls,
        }
        self._changed = True

    def _append(self, entry: dict) -> None:
        self.window.append(entry)
        if len(self.window) > SESSION_WINDOW:
            dropped = self.window[: len(self.window) - SESSION_WINDOW]
            self.window = self.window[-SESSION_WINDOW:]
            for d in dropped:
                words = [w for w in _tokens(d["text"]) if len(w) > 3]
                if words:
                    self._summary.append(" ".join(words[:8]))
            if len(self._summary) > 4:
                self._summary = self._summary[-4:]

    # ---------- fingerprint / dedup ----------

    def context_block(self) -> str:
        parts = [f"{e['speaker']}: {e['text']}" for e in self.window]
        if self._summary:
            parts.append("(resumo: " + " ; ".join(self._summary) + ")")
        return "\n".join(parts)

    @staticmethod
    def _fp(text: str) -> str:
        import hashlib
        return hashlib.sha256(re.sub(r"\s+", " ", text).encode()).hexdigest()[:16]

    def ctx_fingerprint(self) -> str:
        return self._fp(self.context_block())

    # ---------- perfil ----------

    def to_dict(self) -> dict:
        return {"type": self.profile["type"],
                "confidence": round(self.profile["confidence"], 3),
                "source": self.profile["source"],
                "updated_at": self.profile["updated_at"]}

    @property
    def changed_signal(self) -> bool:
        return self._changed

    def consume_changed(self) -> bool:
        was = self._changed
        self._changed = False
        return was

    # ---------- persistência ----------

    def _persist(self) -> None:
        payload = dict(self.profile)
        payload["window_n"] = len(self.window)
        payload["classifications"] = self._n_calls
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            tmp = self.directory / f"{self.session_id}_session.json.tmp"
            target = self.directory / f"{self.session_id}_session.json"
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, target)
        except OSError as e:
            print(f"[session] persistência falhou: {e}")

    def _load(self) -> None:
        try:
            target = self.directory / f"{self.session_id}_session.json"
            if target.exists():
                data = json.loads(target.read_text(encoding="utf-8"))
                if data.get("type") in SESSION_TYPES:
                    self.profile.update({k: data[k] for k in ("type", "confidence", "source")
                                         if k in data and data[k] is not None})
                    self.profile["started_at"] = data.get("started_at") or _now_iso()
                    self.profile["updated_at"] = data.get("updated_at", _now_iso())
        except (OSError, ValueError):
            pass

    def mark_ended(self) -> None:
        """Marca o encerramento da sessão no perfil e persiste."""
        self.profile["ended_at"] = _now_iso()
        self._persist()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w\u00C0-\u024F']+", text.lower())