"""Kivi - inspection UI.

Scope note: this is a TEST HARNESS, not the product surface. It exists so the
pipeline can be driven and inspected by hand while the backend is built. The
product interface is a separate design problem - the assignment grades
"interface, interactions, language, visual character" alongside the backend, and
a tabbed inspector is not an answer to that.

Run: streamlit run ui/app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kivi import db as kivi_db  # noqa: E402
from kivi.config import get_settings  # noqa: E402
from kivi.llm.gemini import GeminiClient  # noqa: E402
from kivi.memory import policy  # noqa: E402
from kivi.memory.search import recall as run_recall  # noqa: E402
from kivi.retrieval.search import Filters, search as run_search  # noqa: E402

st.set_page_config(page_title="Kivi memory", page_icon="•", layout="wide")

st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      .block-container {padding-top: 2.2rem; max-width: 1150px;}
      .ep {border-left: 2px solid #d9d4c4; padding: .35rem 0 .35rem .8rem; margin: .5rem 0;}
      .meta {color: #8a8574; font-size: .78rem; letter-spacing: .02em;}
      .quote {color: #55503f; font-size: .82rem; font-style: italic;}
      .pill {display:inline-block; padding:.05rem .5rem; border-radius:10px;
             font-size:.72rem; background:#efece1; color:#5b5646; margin-right:.3rem;}
      .bad {background:#f6dcd8; color:#8a3227;}
      .ok {background:#dfe9db; color:#3d5c33;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def _migrated() -> int:
    """Run migrations once per process, not once per rerun."""
    settings = get_settings()
    conn = kivi_db.connect(settings.resolved_db_path, check_same_thread=False)
    try:
        kivi_db.migrate(conn)
        return kivi_db.current_version(conn)
    finally:
        conn.close()


def get_conn():
    """A fresh connection per script run.

    Deliberately NOT cached. Streamlit executes each rerun on whichever thread is
    free, and a sqlite3 connection belongs to the thread that opened it - a
    cached one raises ProgrammingError as soon as the thread changes. Opening a
    connection costs well under a millisecond, so caching it buys nothing and
    costs correctness.
    """
    settings = get_settings()
    return kivi_db.connect(settings.resolved_db_path, check_same_thread=False)


@st.cache_resource
def get_client():
    """Safe to cache: httpx is thread-safe, and the LLM cache opens its own
    short-lived connection per call rather than holding one."""
    return GeminiClient(get_settings())


_migrated()
conn = get_conn()
client = get_client()


def pill(text: str, kind: str = "") -> str:
    return f'<span class="pill {kind}">{text}</span>'


def _field(row, attr: str, key: str, default=None):
    """Read a field from either a dataclass or a sqlite3.Row.

    Both flow into this view: `search()` returns Hit objects with attributes,
    while direct queries return Rows with keys, and they name the episode
    differently (`episode_id` vs `id`). Normalising here keeps one renderer
    instead of two that drift apart.
    """
    value = getattr(row, attr, None)
    if value is not None:
        return value
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def episode_block(row, quote: str | None = None) -> None:
    ts = _field(row, "ts", "ts", "") or ""
    app = _field(row, "app", "app") or "—"
    ident = _field(row, "episode_id", "id", "") or ""
    text = _field(row, "formatted", "formatted", "") or ""
    st.markdown(
        f'<div class="ep"><div class="meta">{ts[:16].replace("T", " ")} · '
        f'{app} · {ident}</div>{text}'
        + (f'<div class="quote">“{quote}”</div>' if quote else "")
        + "</div>",
        unsafe_allow_html=True,
    )


st.title("Kivi")
st.caption(
    "Inspection harness for the memory backend — episodes, what was learned, "
    "what was refused, and why."
)

tabs = st.tabs(
    ["Dictations", "What Kivi knows", "What it refused", "Recall", "Why", "Policy"]
)

# --------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Search your dictations")
    col1, col2, col3, col4 = st.columns([4, 1.3, 1.3, 1])
    query = col1.text_input("Search", "", placeholder="what did I say about…",
                            label_visibility="collapsed")
    apps = [r["app"] for r in conn.execute(
        "SELECT DISTINCT app FROM episodes WHERE app IS NOT NULL ORDER BY app"
    ).fetchall()]
    app_choice = col2.selectbox("App", ["any", *apps], label_visibility="collapsed")
    hour = col3.text_input("Hour", "", placeholder="16-18", label_visibility="collapsed")
    use_bm25 = col4.checkbox("bm25", value=False, help="Lexical ranker (off by default)")

    if query:
        filters = Filters(app=None if app_choice == "any" else app_choice)
        if hour and "-" in hour:
            lo, hi = hour.split("-", 1)
            try:
                filters.local_hour_min, filters.local_hour_max = int(lo), int(hi)
            except ValueError:
                st.warning("Hour window should look like 16-18")
        with st.spinner("searching…"):
            result = run_search(conn, client, query, limit=12,
                                filters=filters, use_bm25=use_bm25)
        st.caption(
            f"{len(result.hits)} results · {result.latency_ms} ms · "
            f"{result.candidate_count} candidates · "
            f"{'vec0' if result.used_vec_index else 'numpy'}"
        )
        for hit in result.hits:
            episode_block(hit)
    else:
        total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        st.caption(f"{total:,} dictations stored. Everything is searchable, always.")
        for row in conn.execute(
            "SELECT id, ts, app, formatted FROM episodes ORDER BY ts_epoch DESC LIMIT 8"
        ).fetchall():
            episode_block(row)

# --------------------------------------------------------------------------
with tabs[1]:
    st.subheader("What Kivi has learned")
    counts = dict(
        (r["status"], r["n"]) for r in conn.execute(
            "SELECT status, COUNT(*) AS n FROM memories GROUP BY status"
        ).fetchall()
    )
    episodes_total = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active memories", counts.get("active", 0))
    c2.metric("Superseded", counts.get("superseded", 0))
    c3.metric("Expired", counts.get("expired", 0))
    c4.metric("From dictations", f"{episodes_total:,}")

    kind = st.radio("Type", ["all", *policy.MEMORY_TYPES], horizontal=True)
    sql = "SELECT * FROM memories WHERE status = 'active'"
    params: list = []
    if kind != "all":
        sql += " AND type = ?"
        params.append(kind)
    sql += " ORDER BY confidence DESC, subject LIMIT 60"

    for row in conn.execute(sql, params).fetchall():
        aliases = json.loads(row["aliases_json"] or "[]")
        header = f"**{row['subject']}**"
        if row["body"]:
            header += f" — {row['body'][:90]}"
        with st.expander(header):
            st.markdown(
                pill(row["type"]) + pill(f"confidence {row['confidence']:.2f}")
                + (pill(row["entity_type"]) if row["entity_type"] else "")
                + (pill("pinned", "ok") if row["pinned"] else "")
                + (pill(f"due {row['due_at']}") if row["due_at"] else ""),
                unsafe_allow_html=True,
            )
            if aliases:
                st.caption("also called: " + ", ".join(aliases))
            sources = conn.execute(
                "SELECT e.id, e.ts, e.app, e.formatted, s.quote"
                " FROM memory_sources s JOIN episodes e ON e.id = s.source_id"
                " WHERE s.memory_id = ? AND s.source_kind = 'episode'"
                " ORDER BY e.ts_epoch DESC LIMIT 6",
                (row["id"],),
            ).fetchall()
            total_sources = conn.execute(
                "SELECT COUNT(*) FROM memory_sources WHERE memory_id = ?", (row["id"],)
            ).fetchone()[0]
            st.markdown(f"**Because you said this** ({total_sources} dictations)")
            for source in sources:
                episode_block(source, source["quote"])
            if row["supersedes_id"]:
                old = conn.execute(
                    "SELECT subject, body, valid_to FROM memories WHERE id = ?",
                    (row["supersedes_id"],),
                ).fetchone()
                if old:
                    st.info(
                        f"This replaced an earlier version on "
                        f"{(old['valid_to'] or '')[:10]}: {old['body'] or old['subject']}"
                    )

# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("What Kivi refused to remember")
    st.caption(
        "Every candidate that was considered and turned down, with the reason. "
        "This is the part most memory products do not show you."
    )
    reasons = conn.execute(
        "SELECT reason, COUNT(*) AS n FROM candidates WHERE status = 'rejected'"
        " GROUP BY reason ORDER BY n DESC"
    ).fetchall()
    for row in reasons:
        st.markdown(f"{pill(str(row['n']), 'bad')} {row['reason']}", unsafe_allow_html=True)

    st.divider()
    status = st.radio("Show", ["rejected", "pending", "promoted"], horizontal=True)
    rows = conn.execute(
        "SELECT * FROM candidates WHERE status = ?"
        " ORDER BY evidence_count DESC, last_seen DESC LIMIT 60",
        (status,),
    ).fetchall()
    for row in rows:
        with st.expander(f"**{row['subject']}** — {(row['reason'] or '')[:80]}"):
            st.markdown(
                pill(row["type"]) + pill(f"stance {row['stance']}")
                + pill(f"{row['evidence_count']} episode(s)")
                + (pill("stated", "ok") if row["explicit"] else pill("inferred")),
                unsafe_allow_html=True,
            )
            if row["quote"]:
                st.markdown(f'<div class="quote">“{row["quote"]}”</div>',
                            unsafe_allow_html=True)

# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Recall")
    st.caption(
        "Searches memories, then widens by one hop through the graph. "
        "Answers stay grounded in the dictations underneath."
    )
    rcol1, rcol2 = st.columns([5, 1])
    rquery = rcol1.text_input("Recall", "", placeholder="who is working on DSPM?",
                              label_visibility="collapsed")
    use_graph = rcol2.checkbox("graph", value=True)

    if rquery:
        with st.spinner("recalling…"):
            result = run_recall(conn, client, rquery, limit=6, use_graph=use_graph)
        st.caption(
            f"{len(result.hits)} memories · {result.expanded} via graph · "
            f"{result.latency_ms} ms · {len(result.episode_ids)} source dictations"
        )
        for hit in result.hits:
            via = "direct" if hit.via == "direct" else hit.via
            st.markdown(
                pill(via, "ok" if hit.via == "direct" else "")
                + pill(hit.type) + f"**{hit.subject}** "
                + (hit.body or ""),
                unsafe_allow_html=True,
            )
        if result.episode_ids:
            with st.expander(f"Source dictations ({len(result.episode_ids)})"):
                placeholders = ",".join("?" * min(len(result.episode_ids), 20))
                for row in conn.execute(
                    f"SELECT id, ts, app, formatted FROM episodes"
                    f" WHERE id IN ({placeholders}) ORDER BY ts_epoch DESC",
                    result.episode_ids[:20],
                ).fetchall():
                    episode_block(row)

# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Why")
    st.caption("Every episode leaves a trace, including the ones nothing was learned from.")
    decision = st.radio(
        "Outcome", ["all", "learned", "nothing learned", "skipped"], horizontal=True
    )
    sql = "SELECT * FROM traces WHERE kind = 'extract'"
    params = []
    if decision != "all":
        sql += " AND decision = ?"
        params.append(decision)
    sql += " ORDER BY created_at DESC LIMIT 40"

    for row in conn.execute(sql, params).fetchall():
        episode = conn.execute(
            "SELECT id, ts, app, formatted FROM episodes WHERE id = ?",
            (row["subject_id"],),
        ).fetchone()
        if not episode:
            continue
        with st.expander(f"{row['decision']} — {episode['formatted'][:80]}"):
            episode_block(episode)
            st.markdown(f"**Decision** {row['decision']} — {row['reason']}")
            try:
                decisions = json.loads(row["candidates_json"] or "[]")
            except ValueError:
                decisions = []
            for item in decisions:
                colour = "ok" if item.get("status") == "promoted" else "bad"
                st.markdown(
                    pill(item.get("status", ""), colour)
                    + f"**{item.get('subject')}** ({item.get('type')}) — "
                    + str(item.get("reason", "")),
                    unsafe_allow_html=True,
                )
            st.caption(
                f"{row['model'] or '—'} · {row['tokens_in']}/{row['tokens_out']} tokens · "
                f"{row['latency_ms']} ms"
                + (" · cached" if row["cached"] else "")
            )

# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Policy in force")
    st.caption(
        "Everything Kivi will and will not remember, in one place. These values "
        "are provisional until reconciled with the written product position."
    )
    st.json(policy.describe())
