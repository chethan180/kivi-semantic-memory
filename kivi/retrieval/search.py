"""Hybrid search over episodes.

Three rankers, fused with Reciprocal Rank Fusion:

  vector   semantic similarity - finds the right meaning under different words
  bm25     lexical - finds the exact rare token, including a name the recogniser
           mangled, which the embedding of the cleaned text has smoothed away
  recency  a mild prior, because dictation questions skew towards recent work

RRF rather than score normalisation: cosine similarity and bm25 are on
incomparable scales, and every scheme for forcing them onto one is a tuning
parameter pretending to be a principle. Rank position is comparable by
construction.

Filters are applied *before* ranking when present. Over-fetching from a vector
index and filtering afterwards silently loses results whenever the filter is
narrow - which is exactly the case the assignment's own example asks for
("around 5 PM yesterday in Slack").
"""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from kivi.db.connection import vec_available
from kivi.llm.cache import pack_vector
from kivi.llm.gemini import GeminiClient
from kivi.retrieval.embed import cosine_ranking, load_matrix

# Standard RRF constant. Damps the influence of any single ranker's top hit.
RRF_K = 60

# Vector leads; bm25 is a corrective for rare tokens rather than an equal
# partner. Measured: weighting them equally made hybrid *worse* than vector
# alone on every corpus, because a lexical ranker fed a natural-language
# question returns plausible-looking noise at rank 1 and RRF cannot tell that
# rank 1 from a good one. See eval/results for the ablation.
WEIGHTS = {"vector": 1.0, "bm25": 0.5, "recency": 0.15}

# Everything that is not a word character or whitespace is replaced with a
# space. A narrower rule that only listed FTS5's operators left ordinary
# punctuation attached to terms, so the final word of every question was
# searched as `migration?` and matched nothing - the last, and often most
# discriminating, term of every query was silently wasted.
_FTS_SPECIALS = re.compile(r"[^\w\s]", re.UNICODE)

# Dropped before an FTS5 query is built. A question phrased in English is mostly
# function words; ORing them matches most of the corpus and drowns the one term
# that actually discriminates.
_STOPWORDS = frozenset("""
a about after all also am an and any are as at be because been before being
but by can did do does doing done for from had has have he her here hers him
his how i if in into is it its just me more most my no not now of on one only
or other our out over own said same she should so some such than that the
their them then there these they this those to too up us very was we were what
when where which while who whom why will with would you your yours
did i my me tell show find get give need want know about
""".split())


@dataclass
class Filters:
    """Structured narrowing. Every field is optional and ANDed."""

    app: str | None = None
    apps: list[str] | None = None
    after: dt.datetime | None = None
    before: dt.datetime | None = None
    local_hour_min: int | None = None
    local_hour_max: int | None = None
    local_date: str | None = None
    source: str | None = None

    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in (
                "app", "apps", "after", "before",
                "local_hour_min", "local_hour_max", "local_date", "source",
            )
        )

    def where(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if self.app:
            clauses.append("e.app = ?")
            params.append(self.app.lower())
        if self.apps:
            clauses.append(f"e.app IN ({','.join('?' * len(self.apps))})")
            params.extend(a.lower() for a in self.apps)
        if self.after:
            clauses.append("e.ts_epoch >= ?")
            params.append(int(self.after.timestamp()))
        if self.before:
            clauses.append("e.ts_epoch <= ?")
            params.append(int(self.before.timestamp()))
        if self.local_date:
            clauses.append("e.local_date = ?")
            params.append(self.local_date)
        if self.local_hour_min is not None and self.local_hour_max is not None:
            if self.local_hour_min <= self.local_hour_max:
                clauses.append("e.local_hour BETWEEN ? AND ?")
                params.extend([self.local_hour_min, self.local_hour_max])
            else:
                # A window that wraps midnight, e.g. 22:00-02:00.
                clauses.append("(e.local_hour >= ? OR e.local_hour <= ?)")
                params.extend([self.local_hour_min, self.local_hour_max])
        elif self.local_hour_min is not None:
            clauses.append("e.local_hour >= ?")
            params.append(self.local_hour_min)
        elif self.local_hour_max is not None:
            clauses.append("e.local_hour <= ?")
            params.append(self.local_hour_max)
        if self.source:
            clauses.append("e.source = ?")
            params.append(self.source)
        return (" AND ".join(clauses) if clauses else "1=1"), params


@dataclass
class Hit:
    episode_id: str
    score: float
    ts: str                      # UTC, the ordering key
    app: str | None
    formatted: str
    raw_asr: str
    local_ts: str = ""           # the wall clock the speaker actually saw
    local_hour: int | None = None
    ranks: dict[str, int] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def when(self) -> str:
        """How this moment should be shown to a person, or to the model.

        Always local. Showing UTC here is not a cosmetic problem: asked for a
        message sent at 4:59 PM, the model was handed "11:29", compared the two,
        and reported truthfully that it had found nothing matching - while
        holding the right record.
        """
        return (self.local_ts or self.ts)[:16].replace("T", " ")


@dataclass
class SearchResult:
    hits: list[Hit]
    latency_ms: int
    candidate_count: int
    used_vec_index: bool
    filtered: bool
    rankers: list[str]
    query: str

    def ids(self) -> list[str]:
        return [hit.episode_id for hit in self.hits]


def fts_query(text: str) -> str:
    """Build a safe FTS5 query from natural language.

    Stopwords are removed before the terms are ORed. Without that, a question
    like "Who is working on DSPM?" becomes `who OR is OR working OR on OR DSPM`,
    which matches a large share of the corpus on its function words alone; bm25
    then ranks near-irrelevant episodes first and RRF, which only sees rank
    position, promotes them as confidently as a real hit.

    Returns "" when nothing discriminating survives, which correctly disables the
    lexical ranker for that query rather than having it contribute noise.
    """
    cleaned = _FTS_SPECIALS.sub(" ", text.lower())
    terms = [
        t for t in cleaned.split()
        if len(t) > 1 and t not in _STOPWORDS and not t.isdigit()
    ]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


def _filtered_ids(conn: sqlite3.Connection, filters: Filters) -> list[str]:
    where, params = filters.where()
    rows = conn.execute(
        f"SELECT e.id FROM episodes e WHERE {where}", params
    ).fetchall()
    return [row["id"] for row in rows]


def _bm25_ranking(
    conn: sqlite3.Connection,
    query: str,
    limit: int,
    allowed: set[str] | None,
) -> list[tuple[str, float]]:
    match = fts_query(query)
    if not match:
        return []
    try:
        rows = conn.execute(
            "SELECT e.id AS id, bm25(episodes_fts) AS score"
            " FROM episodes_fts JOIN episodes e ON e.rowid = episodes_fts.rowid"
            " WHERE episodes_fts MATCH ? ORDER BY score LIMIT ?",
            (match, limit * 8),
        ).fetchall()
    except sqlite3.OperationalError:
        # A query that still upsets the FTS5 parser should degrade to no lexical
        # signal, not take the whole search down.
        return []
    # bm25() returns negative numbers, better (more relevant) being more negative.
    out = [(row["id"], -float(row["score"])) for row in rows]
    if allowed is not None:
        out = [item for item in out if item[0] in allowed]
    return out[:limit]


def _vector_ranking(
    conn: sqlite3.Connection,
    client: GeminiClient,
    query: str,
    limit: int,
    allowed: list[str] | None,
) -> tuple[list[tuple[str, float]], bool]:
    model = client.settings.embed_model
    dim = client.settings.embed_dim
    vector = client.embed_one(query, task_type="RETRIEVAL_QUERY")

    # Filtered search: exact cosine over just the surviving rows. Correct by
    # construction, and at this corpus size faster than it sounds.
    if allowed is not None:
        ids, matrix = load_matrix(conn, model, dim, allowed)
        return cosine_ranking(vector, ids, matrix, limit), False

    if vec_available(conn):
        try:
            rows = conn.execute(
                "SELECT e.id AS id, v.distance AS distance"
                " FROM episode_vec v JOIN episodes e ON e.rowid = v.rowid"
                " WHERE v.embedding MATCH ? AND k = ?"
                " ORDER BY v.distance",
                (pack_vector(vector), limit),
            ).fetchall()
            if rows:
                return [(r["id"], -float(r["distance"])) for r in rows], True
        except sqlite3.OperationalError:
            pass  # index missing or not yet built; fall through to numpy

    ids, matrix = load_matrix(conn, model, dim)
    return cosine_ranking(vector, ids, matrix, limit), False


def _gate(ranking: list[tuple[str, float]], ratio: float = 0.55) -> list[tuple[str, float]]:
    """Keep only hits scoring within `ratio` of the best one.

    Absolute bm25 scores are not comparable across queries, but they are
    comparable *within* one. This drops the long tail of episodes that matched a
    single incidental token, which RRF would otherwise treat as real evidence.
    """
    if not ranking:
        return ranking
    best = ranking[0][1]
    if best <= 0:
        return ranking
    return [item for item in ranking if item[1] >= best * ratio]


def _recency_ranking(
    conn: sqlite3.Connection, limit: int, allowed: set[str] | None
) -> list[tuple[str, float]]:
    rows = conn.execute(
        "SELECT id, ts_epoch FROM episodes ORDER BY ts_epoch DESC LIMIT ?",
        (limit * 4,),
    ).fetchall()
    out = [(row["id"], float(row["ts_epoch"])) for row in rows]
    if allowed is not None:
        out = [item for item in out if item[0] in allowed]
    return out[:limit]


def search(
    conn: sqlite3.Connection,
    client: GeminiClient,
    query: str,
    *,
    limit: int = 10,
    filters: Filters | None = None,
    use_vector: bool = True,
    use_bm25: bool = False,
    use_recency: bool = False,
) -> SearchResult:
    """Search episodes. The ranker flags exist so the evaluation can ablate them.

    Both non-vector rankers default OFF, and that is an evidence-led choice
    rather than an architectural one. The ablation in eval/results is consistent
    across all three corpora:

        corpus   vector-only hit@10   + bm25 hit@10
        house          0.833              0.789
        mixed          0.978              0.933
        office         0.833              0.822

    Adding a lexical ranker costs 1-5 points. The likely reason is a property of
    the corpus, not of bm25: it is model-generated, so its vocabulary is clean
    and consistent, and the embeddings already capture almost everything the
    tokens could. The ASR manglings that would give bm25 its edge live in
    `raw_asr`, but the evaluation questions are all phrased with correct
    spellings, so that advantage is never exercised.

    The mechanism is kept and tested because Sarvam's real corpus is ASR output,
    where a mangled proper noun is exactly the case embeddings smooth away. The
    deciding experiment - questions phrased with the mangled spelling - is noted
    in the README as not yet run.

    Recency is a prior about *temporal* questions ("what did I say yesterday")
    and injects the newest episodes regardless of relevance otherwise. Callers
    that know the question is time-oriented turn it on; filters do the rest.
    """
    started = time.perf_counter()
    filters = filters or Filters()
    filtered = not filters.is_empty()

    allowed_list = _filtered_ids(conn, filters) if filtered else None
    allowed_set = set(allowed_list) if allowed_list is not None else None

    rankings: dict[str, list[tuple[str, float]]] = {}
    used_vec_index = False

    # Depth differs per ranker on purpose. RRF only sees rank position, so a
    # ranker contributing a long weak tail donates the same kind of evidence at
    # position 30 as it does at position 1. The vector ranker degrades
    # gracefully and can be read deep; bm25's tail is mostly incidental token
    # overlap, so it is read shallow and additionally gated on score below.
    if use_vector:
        ranking, used_vec_index = _vector_ranking(
            conn, client, query, limit * 4, allowed_list
        )
        rankings["vector"] = ranking
    if use_bm25:
        rankings["bm25"] = _gate(_bm25_ranking(conn, query, limit, allowed_set))
    if use_recency:
        rankings["recency"] = _recency_ranking(conn, limit, allowed_set)

    fused: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    raw: dict[str, dict[str, float]] = {}
    for name, ranking in rankings.items():
        weight = WEIGHTS.get(name, 1.0)
        for position, (episode_id, score) in enumerate(ranking, start=1):
            fused[episode_id] = fused.get(episode_id, 0.0) + weight / (RRF_K + position)
            ranks.setdefault(episode_id, {})[name] = position
            raw.setdefault(episode_id, {})[name] = score

    top = sorted(fused.items(), key=lambda item: -item[1])[:limit]
    hits: list[Hit] = []
    if top:
        placeholders = ",".join("?" * len(top))
        rows = {
            row["id"]: row
            for row in conn.execute(
                f"SELECT id, ts, ts_epoch, tz_offset_min, local_date, local_hour,"
                f" app, formatted, raw_asr FROM episodes"
                f" WHERE id IN ({placeholders})",
                [item[0] for item in top],
            ).fetchall()
        }
        for episode_id, score in top:
            row = rows.get(episode_id)
            if row is None:
                continue
            offset = dt.timedelta(minutes=int(row["tz_offset_min"] or 0))
            local = dt.datetime.fromtimestamp(
                int(row["ts_epoch"]), dt.timezone.utc
            ) + offset
            hits.append(
                Hit(
                    episode_id=episode_id,
                    score=score,
                    ts=row["ts"],
                    app=row["app"],
                    formatted=row["formatted"],
                    raw_asr=row["raw_asr"],
                    local_ts=local.strftime("%Y-%m-%dT%H:%M"),
                    local_hour=row["local_hour"],
                    ranks=ranks.get(episode_id, {}),
                    scores=raw.get(episode_id, {}),
                )
            )

    return SearchResult(
        hits=hits,
        latency_ms=int((time.perf_counter() - started) * 1000),
        candidate_count=len(fused),
        used_vec_index=used_vec_index,
        filtered=filtered,
        rankers=sorted(rankings),
        query=query,
    )
