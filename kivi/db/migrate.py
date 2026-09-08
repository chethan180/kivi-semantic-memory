"""Migration runner.

Numbered `.sql` files applied in order, each in a transaction, recorded in
`schema_migrations`. Deliberately small: the assignment asks for a schema and
migrations that a reviewing agent can run, not a migration framework.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
_NAME_RE = re.compile(r"^(\d+)_(.+)\.sql$")

BOOTSTRAP = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    checksum   TEXT NOT NULL,
    applied_at TEXT NOT NULL
);
"""


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _split_statements(sql: str) -> list[str]:
    """Split a migration into individually executable statements.

    Uses `sqlite3.complete_statement`, which is SQLite's own parser and already
    understands that the semicolons inside a `CREATE TRIGGER ... BEGIN ... END;`
    block do not end the statement. A hand-rolled splitter got this wrong in a
    way worth recording: it looked for a line *starting* with BEGIN, but a
    trigger header ends with it (`... ON episodes BEGIN`), so every trigger body
    swallowed the rest of the file.
    """
    statements: list[str] = []
    buffer = ""

    for line in sql.splitlines(keepends=True):
        if not buffer and not line.strip():
            continue
        # A standalone comment line before any statement is not part of one.
        if not buffer and line.strip().startswith("--"):
            continue
        buffer += line
        if sqlite3.complete_statement(buffer):
            statements.append(buffer.strip())
            buffer = ""

    if buffer.strip():
        statements.append(buffer.strip())
    return statements


def discover() -> list[tuple[int, str, Path]]:
    """All migration files on disk, ordered by version."""
    found: list[tuple[int, str, Path]] = []
    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        match = _NAME_RE.match(path.name)
        if not match:
            raise ValueError(
                f"migration {path.name} must be named <version>_<name>.sql"
            )
        found.append((int(match.group(1)), match.group(2), path))
    found.sort(key=lambda item: item[0])
    return found


def applied(conn: sqlite3.Connection) -> dict[int, sqlite3.Row]:
    conn.executescript(BOOTSTRAP)
    rows = conn.execute(
        "SELECT version, name, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    return {row["version"]: row for row in rows}


def current_version(conn: sqlite3.Connection) -> int:
    done = applied(conn)
    return max(done) if done else 0


def pending_migrations(conn: sqlite3.Connection) -> list[tuple[int, str, Path]]:
    done = applied(conn)
    return [item for item in discover() if item[0] not in done]


def migrate(conn: sqlite3.Connection, *, verbose: bool = False) -> list[str]:
    """Apply pending migrations. Returns the names applied, in order."""
    done = applied(conn)
    applied_now: list[str] = []

    for version, name, path in discover():
        sql = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]

        if version in done:
            if done[version]["checksum"] != checksum:
                # An edited migration means the DB no longer matches the code.
                # Say so loudly rather than pretending the schema is current.
                log.warning(
                    "migration %03d_%s changed since it was applied; "
                    "recreate the database with `kivi reset --hard` to pick it up",
                    version, name,
                )
            continue

        if verbose:
            log.info("applying migration %03d_%s", version, name)
        try:
            # Statement-at-a-time inside one transaction, NOT executescript.
            #
            # executescript issues an implicit COMMIT before it runs, so a
            # migration that fails halfway leaves everything before the failure
            # permanently applied and unrecorded - the database ends up in a
            # state no migration describes, and re-running cannot fix it because
            # the early statements now conflict. This happened: a syntax error in
            # a trigger left a table already rebuilt, and the retry failed
            # against the new shape. SQLite DDL is transactional, so doing it
            # this way means a failed migration leaves no trace.
            conn.execute("BEGIN")
            for statement in _split_statements(sql):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_migrations (version, name, checksum, applied_at)"
                " VALUES (?, ?, ?, ?)",
                (version, name, checksum, _utcnow()),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        applied_now.append(f"{version:03d}_{name}")

    return applied_now
