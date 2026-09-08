"""The dictation record format, and adapters into it.

The assignment says the reviewing corpus will contain raw ASR output, the
LLM-formatted output, and ordinary log metadata. That is exactly the shape below.
`adapt()` maps a foreign record into it, so importing another corpus is a
documented mapping rather than a code change.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field, field_validator

# Field names an incoming record might use for each of ours. Checked in order.
ALIASES: dict[str, tuple[str, ...]] = {
    "external_id": ("external_id", "id", "record_id", "dictation_id", "uuid"),
    "ts": ("ts", "timestamp", "created_at", "recorded_at", "time", "date"),
    "app": ("app", "application", "target_app", "destination", "context_app"),
    "raw_asr": ("raw_asr", "raw", "asr", "asr_text", "transcript", "raw_text"),
    "formatted": (
        "formatted", "formatted_text", "llm_output", "llm_formatted",
        "output", "final_text", "text",
    ),
    "style_id": ("style_id", "style", "style_name"),
    "duration_ms": ("duration_ms", "duration", "length_ms", "audio_ms"),
    "recipient": ("recipient", "to", "addressed_to", "sent_to", "counterparty"),
}

# Anything not mapped above is preserved here rather than dropped, so a corpus
# with extra columns loses nothing on import.
_KNOWN = {name for names in ALIASES.values() for name in names} | {"meta", "metadata"}


class RecordError(ValueError):
    """A record could not be parsed. Carries the line number for reporting."""

    def __init__(self, message: str, line: int | None = None) -> None:
        super().__init__(message)
        self.line = line


class EpisodeRecord(BaseModel):
    """One dictation, as it arrives from a corpus."""

    external_id: str | None = None
    ts: dt.datetime
    app: str | None = None
    raw_asr: str
    formatted: str
    style_id: str | None = None
    duration_ms: int | None = None
    recipient: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ts")
    @classmethod
    def _ensure_aware(cls, value: dt.datetime) -> dt.datetime:
        """Keep the offset the dictation was spoken in.

        Deliberately not converted to UTC here: the local wall-clock hour is what
        a person means by "around 5 PM", and normalising on the way in would
        throw it away. A naive timestamp is assumed to be UTC.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=dt.timezone.utc)
        return value

    @property
    def ts_utc(self) -> dt.datetime:
        return self.ts.astimezone(dt.timezone.utc)

    @property
    def tz_offset_min(self) -> int:
        offset = self.ts.utcoffset() or dt.timedelta(0)
        return int(offset.total_seconds() // 60)

    @property
    def local_hour(self) -> int:
        return self.ts.hour

    @property
    def local_date(self) -> str:
        return self.ts.date().isoformat()

    @property
    def recipient_norm(self) -> str | None:
        """Matching key for a recipient, tolerant of how the name was typed."""
        if not self.recipient:
            return None
        return " ".join(self.recipient.strip().lower().split()) or None

    @field_validator("app", "style_id")
    @classmethod
    def _normalise_label(cls, value: str | None) -> str | None:
        return value.strip().lower() or None if value else None

    @property
    def content_hash(self) -> str:
        """Dedupe identity: same text at the same instant is the same dictation."""
        # Normalised to UTC so the same instant expressed in two offsets dedupes.
        basis = f"{self.ts_utc.isoformat()}|{self.app or ''}|{self.raw_asr}|{self.formatted}"
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()

    @property
    def episode_id(self) -> str:
        """Deterministic id, so re-importing a corpus is idempotent."""
        basis = self.external_id or self.content_hash
        return "ep_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def _first_present(data: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in data and data[name] not in (None, ""):
            return data[name]
    return None


def adapt(data: dict[str, Any]) -> EpisodeRecord:
    """Map an arbitrary record onto EpisodeRecord using the alias table.

    Tolerant on purpose: a corpus that calls the field `transcript` instead of
    `raw_asr` should import without anyone editing code. Anything genuinely
    ambiguous raises rather than being guessed at.
    """
    mapped: dict[str, Any] = {
        field: _first_present(data, names) for field, names in ALIASES.items()
    }

    # A corpus may carry only one text field. Formatted output is what the user
    # saw, so treat a lone text field as formatted and mirror it into raw_asr
    # rather than inventing a transcript.
    if mapped["raw_asr"] is None and mapped["formatted"] is not None:
        mapped["raw_asr"] = mapped["formatted"]
    if mapped["formatted"] is None and mapped["raw_asr"] is not None:
        mapped["formatted"] = mapped["raw_asr"]

    if not mapped["raw_asr"]:
        raise RecordError(
            "record has no recognisable text field; expected one of "
            f"{ALIASES['raw_asr'] + ALIASES['formatted']}"
        )
    if mapped["ts"] is None:
        raise RecordError(
            f"record has no timestamp; expected one of {ALIASES['ts']}"
        )

    meta = dict(data.get("meta") or data.get("metadata") or {})
    for key, value in data.items():
        if key not in _KNOWN and key not in meta:
            meta[key] = value
    mapped["meta"] = meta

    if isinstance(mapped["duration_ms"], float):
        mapped["duration_ms"] = int(mapped["duration_ms"])

    return EpisodeRecord(**mapped)


def load_jsonl(path: Path | str) -> Iterator[EpisodeRecord]:
    """Stream a JSONL corpus. Blank lines are skipped; bad lines raise with a number."""
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RecordError(f"invalid JSON: {exc}", line=line_no) from exc
            try:
                yield adapt(data)
            except RecordError as exc:
                raise RecordError(str(exc), line=line_no) from exc
            except Exception as exc:
                raise RecordError(str(exc), line=line_no) from exc
