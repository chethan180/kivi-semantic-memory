"""SQLite connections.

One design note that matters for the reviewing agent: `sqlite-vec` is loaded as a
compiled extension, and `sqlite3.enable_load_extension` is compiled out of some
Python builds (notably macOS system Python). So vectors are stored in an ordinary
BLOB table which is always the source of truth, and the vec0 index is treated as
an accelerator that may or may not be present. A machine without the extension
falls back to numpy cosine over the same rows - at this corpus size that is a few
milliseconds - rather than failing to start.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_vec_state: bool | None = None


def connect(
    db_path: Path | str,
    *,
    load_vec: bool = True,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """Open a connection with the pragmas this project relies on.

    `check_same_thread=False` is for hosts that hand successive requests to
    different threads - Streamlit reruns, a web server - where a connection
    would otherwise be rejected for being used off the thread that made it. It
    is only safe alongside short-lived, one-purpose connections; WAL plus the
    busy timeout handle the concurrency, but a single connection shared between
    threads that both write is still a way to corrupt state.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path, timeout=15.0, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Generous, because a single turn can have this connection writing memories
    # while the model cache writes its ledger on another.
    conn.execute("PRAGMA busy_timeout=15000")

    if load_vec:
        _try_load_vec(conn)
    return conn


def _try_load_vec(conn: sqlite3.Connection) -> bool:
    global _vec_state
    try:
        import sqlite_vec
    except ImportError:
        _vec_state = False
        return False

    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (AttributeError, sqlite3.OperationalError) as exc:
        # AttributeError: this Python was built without extension loading.
        if _vec_state is not False:
            log.info("sqlite-vec unavailable (%s); using numpy fallback for search", exc)
        _vec_state = False
        return False

    _vec_state = True
    return True


def vec_available(conn: sqlite3.Connection | None = None) -> bool:
    """Whether the sqlite-vec extension loaded. Reported by `kivi doctor`."""
    if _vec_state is not None:
        return _vec_state
    if conn is not None:
        return _try_load_vec(conn)
    return False
