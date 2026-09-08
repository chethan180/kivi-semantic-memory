"""Regression test for the connection-per-thread failure.

Streamlit runs each rerun on whichever thread is free. A sqlite3 connection is
bound to the thread that opened it, so a connection cached across reruns raises

    ProgrammingError: SQLite objects created in a thread can only be used in
    that same thread

as soon as Streamlit picks a different one. That happened in ui/app.py the first
time the page was opened. These tests pin both halves of the fix.
"""

from __future__ import annotations

import concurrent.futures
import sqlite3

import pytest

from kivi.db import connect, migrate


def _make_db(tmp_path):
    path = tmp_path / "threads.db"
    conn = connect(path)
    migrate(conn)
    conn.execute(
        "INSERT INTO episodes (id, content_hash, ts, ts_epoch, raw_asr, formatted,"
        " meta_json, source, created_at)"
        " VALUES ('ep_1', 'h', '2026-09-05T10:00:00+00:00', 1757066400, 'a', 'A',"
        " '{}', 'test', '2026-09-05T10:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    return path


def test_default_connection_is_thread_bound(tmp_path):
    """Confirms the failure mode is real, so the fix is not cargo cult."""
    path = _make_db(tmp_path)
    conn = connect(path)  # check_same_thread=True, the default

    def query():
        return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(sqlite3.ProgrammingError):
            pool.submit(query).result()
    conn.close()


def test_check_same_thread_false_crosses_threads(tmp_path):
    """The escape hatch the UI needs for a connection it opens and hands off."""
    path = _make_db(tmp_path)
    conn = connect(path, check_same_thread=False)

    def query():
        return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(query).result() == 1
    conn.close()


def test_fresh_connection_per_thread_is_the_actual_pattern(tmp_path):
    """What ui/app.py does now: open per run, never share.

    Several threads each opening their own connection must all succeed, which is
    the behaviour a Streamlit rerun depends on.
    """
    path = _make_db(tmp_path)

    def worker(_):
        conn = connect(path, check_same_thread=False)
        try:
            return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        finally:
            conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(worker, range(24))) == [1] * 24


def test_migrations_are_idempotent_across_connections(tmp_path):
    """The UI migrates once per process but opens many connections after."""
    path = _make_db(tmp_path)
    for _ in range(3):
        conn = connect(path, check_same_thread=False)
        assert migrate(conn) == []  # nothing left to apply
        conn.close()
