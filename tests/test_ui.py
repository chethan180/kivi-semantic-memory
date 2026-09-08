"""Render tests for the inspection UI.

Two runtime errors reached the browser before this file existed - a connection
shared across threads, and a Hit object rendered by code written for a
sqlite3.Row. Both were invisible to the query-level tests because those never
executed the page. AppTest actually runs the script, so a raised exception in any
tab fails here instead of in front of someone.

Skipped when the database has no data, since these assert against real content.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP = Path(__file__).resolve().parent.parent / "ui" / "app.py"


def _has_data() -> bool:
    from kivi.config import get_settings

    path = get_settings().resolved_db_path
    if not path.exists():
        return False
    try:
        conn = sqlite3.connect(path)
        n = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        conn.close()
        return n > 0
    except sqlite3.Error:
        return False


pytestmark = pytest.mark.skipif(
    not _has_data(), reason="needs an ingested database; run `kivi import` first"
)


def _run() -> "AppTest":
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.run()
    return app


def _assert_clean(app) -> None:
    if app.exception:
        raise AssertionError(
            "page raised: " + "\n".join(str(e.value) for e in app.exception)
        )


def test_page_loads_without_exception():
    """The regression both reported bugs would have been caught by."""
    _assert_clean(_run())


def test_every_tab_rendered():
    app = _run()
    _assert_clean(app)
    assert len(app.tabs) == 6


def test_dictation_search_renders_hits():
    """Hit objects and sqlite3.Rows go through the same renderer.

    This is the exact path that raised "'Hit' object is not subscriptable".
    """
    app = _run()
    _assert_clean(app)
    app.text_input[0].set_value("steering meeting").run()
    _assert_clean(app)


def test_recall_renders_with_graph_expansion():
    app = _run()
    _assert_clean(app)
    # The recall box is the second text input on the page.
    app.text_input[2].set_value("who is working on DSPM").run()
    _assert_clean(app)


def test_memory_type_filter_switches_cleanly():
    app = _run()
    _assert_clean(app)
    for value in ("entity", "preference", "commitment", "all"):
        app.radio[0].set_value(value).run()
        _assert_clean(app)


def test_candidate_status_filter_switches_cleanly():
    app = _run()
    _assert_clean(app)
    for value in ("rejected", "pending", "promoted"):
        app.radio[1].set_value(value).run()
        _assert_clean(app)


def test_trace_outcome_filter_switches_cleanly():
    app = _run()
    _assert_clean(app)
    for value in ("learned", "nothing learned", "skipped", "all"):
        app.radio[2].set_value(value).run()
        _assert_clean(app)
