"""Episode ingest. No LLM involved.

Storing a dictation is not a judgement call, so nothing here decides anything:
every record becomes an episode, losslessly. Deciding what is worth *remembering*
is Phase 6's job, and it reads from this table rather than replacing it.

Ingest is idempotent. Episode ids are derived from the record's external id, or
from a content hash when it has none, so re-running an import - which a reviewing
agent will do - inserts nothing new instead of duplicating the corpus.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from kivi.records import EpisodeRecord, RecordError, load_jsonl

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    inserted: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.inserted + self.duplicates

    def summary(self) -> str:
        parts = [f"{self.inserted} inserted"]
        if self.duplicates:
            parts.append(f"{self.duplicates} already present")
        if self.errors:
            parts.append(f"{len(self.errors)} failed")
        return ", ".join(parts)


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def ingest_records(
    conn: sqlite3.Connection,
    records: Iterable[EpisodeRecord],
    *,
    source: str = "import",
    batch_size: int = 500,
) -> IngestReport:
    """Insert episodes, skipping any already present."""
    report = IngestReport()
    now = _utcnow()
    batch: list[tuple] = []

    def _count() -> int:
        return int(conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0])

    def flush() -> None:
        if not batch:
            return
        # INSERT OR IGNORE lets the UNIQUE constraints on external_id and
        # content_hash do the deduplication, rather than a SELECT per row.
        #
        # Count rows directly rather than via total_changes: the FTS triggers
        # fire per insert and total_changes counts trigger writes too, which
        # would over-report insertions and drive the duplicate count negative.
        before = _count()
        conn.executemany(
            "INSERT OR IGNORE INTO episodes"
            " (id, external_id, content_hash, ts, ts_epoch, app, raw_asr,"
            "  formatted, style_id, duration_ms, meta_json, source, created_at,"
            "  tz_offset_min, local_hour, local_date, recipient, recipient_norm)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        inserted = _count() - before
        report.inserted += inserted
        report.duplicates += len(batch) - inserted
        batch.clear()

    for record in records:
        batch.append(
            (
                record.episode_id,
                record.external_id,
                record.content_hash,
                record.ts_utc.isoformat(),
                int(record.ts.timestamp()),
                record.app,
                record.raw_asr,
                record.formatted,
                record.style_id,
                record.duration_ms,
                json.dumps(record.meta, ensure_ascii=False, default=str),
                source,
                now,
                record.tz_offset_min,
                record.local_hour,
                record.local_date,
                record.recipient,
                record.recipient_norm,
            )
        )
        if len(batch) >= batch_size:
            flush()

    flush()
    conn.commit()
    return report


def ingest_file(
    conn: sqlite3.Connection,
    path: Path | str,
    *,
    source: str = "import",
    strict: bool = False,
) -> IngestReport:
    """Ingest a JSONL corpus.

    By default a malformed line is reported and skipped so one bad record cannot
    stop a 500-record import. `strict=True` re-raises instead.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"corpus not found: {path}")

    records: list[EpisodeRecord] = []
    errors: list[str] = []
    stream = load_jsonl(path)

    while True:
        try:
            records.append(next(stream))
        except StopIteration:
            break
        except RecordError as exc:
            if strict:
                raise
            where = f"line {exc.line}" if exc.line else "record"
            errors.append(f"{path.name}:{where}: {exc}")
            log.warning("skipping %s: %s", where, exc)
            # load_jsonl's generator is finished once it raises, so a lenient
            # import needs to resume from a fresh one past the failed line.
            stream = _resume_after(path, exc.line)

    report = ingest_records(conn, records, source=source)
    report.errors.extend(errors)
    return report


def _resume_after(path: Path, line: int | None):
    """Continue a lenient import after a bad line."""
    if line is None:
        return iter(())
    remaining = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            if line_no > line:
                remaining.append((line_no, raw))

    def generator():
        import json as _json

        from kivi.records import adapt

        for line_no, raw in remaining:
            text = raw.strip()
            if not text or text.startswith("//"):
                continue
            try:
                yield adapt(_json.loads(text))
            except Exception as exc:
                raise RecordError(str(exc), line=line_no) from exc

    return generator()


# ---------------------------------------------------------------------------
# Inspection and reset
# ---------------------------------------------------------------------------

# Cleared by `kivi reset`. Ordered so foreign keys never block a delete.
DOMAIN_TABLES = [
    "answers",
    "clarifications",
    "turns",
    "sessions",
    "traces",
    "edges",
    "memory_sources",
    "memory_vectors",
    "memories",
    "candidates",
    "episode_vectors",
    "episodes",
]

# Kept by `kivi reset`, cleared only by `--include-cache`. Resetting the system
# should not re-bill the corpus.
CACHE_TABLES = ["llm_cache", "embed_cache", "api_call_log"]


def reset(conn: sqlite3.Connection, *, include_cache: bool = False) -> dict[str, int]:
    """Empty the system. Returns rows deleted per table."""
    deleted: dict[str, int] = {}
    tables = DOMAIN_TABLES + (CACHE_TABLES if include_cache else [])
    for table in tables:
        # cursor.rowcount maps to sqlite3_changes(), which excludes rows removed
        # by triggers and cascades - the count we actually want per table.
        deleted[table] = max(conn.execute(f"DELETE FROM {table}").rowcount, 0)
    conn.commit()
    # FTS external-content indexes need an explicit rebuild after a bulk delete.
    for fts in ("episodes_fts", "memories_fts"):
        conn.execute(f"INSERT INTO {fts}({fts}) VALUES('rebuild')")

    # The vec0 index is a virtual table and is NOT in DOMAIN_TABLES, so a reset
    # left its rows behind. Re-importing then reuses the same episode rowids and
    # embedding dies on a UNIQUE constraint - which made `reset` followed by
    # `embed` fail every time.
    for virtual in ("episode_vec", "memory_vec"):
        try:
            conn.execute(f"DELETE FROM {virtual}")
        except sqlite3.OperationalError:
            pass  # not created yet, or sqlite-vec unavailable
    conn.commit()
    conn.execute("VACUUM")
    return deleted


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Row counts for every domain table, for `kivi stats` and the eval report."""
    counts: dict[str, int] = {}
    for table in DOMAIN_TABLES + CACHE_TABLES:
        try:
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        except sqlite3.OperationalError:
            continue
        counts[table] = int(row["n"])
    return counts
