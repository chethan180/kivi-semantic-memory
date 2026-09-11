"""Kivi command line.

Phase 1 ships `doctor` only. Import, reset and eval commands arrive with their
phases, so that every command in `RUN.md` is one that actually works.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from kivi import db as kivi_db
from kivi import ingest as ingest_mod
from kivi.config import get_settings
from kivi.llm.cache import LLMCache
from kivi.llm.gemini import GeminiClient, LLMError
from kivi.llm import pricing
from kivi.records import RecordError

app = typer.Typer(
    add_completion=False,
    help="Kivi semantic memory - episodes, curated memories, and Hey Kivi.",
)
console = Console()

OK = "[green]OK[/green]"
WARN = "[yellow]WARN[/yellow]"
FAIL = "[red]FAIL[/red]"


@app.command()
def doctor(
    check_models: bool = typer.Option(
        True, "--check-models/--no-check-models",
        help="Call the Gemini API to verify the key and the configured models exist.",
    ),
) -> None:
    """Report configuration, model access, cache state and database state.

    Exits non-zero if something would stop the system from running, so it can be
    used as a setup gate in CI or by a reviewing agent.
    """
    settings = get_settings()
    _configure_logging(settings.log_level)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="bold", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")

    problems: list[str] = []

    # --- credentials ---------------------------------------------------------
    if settings.has_key:
        masked = f"{settings.gemini_api_key[:6]}...{settings.gemini_api_key[-4:]}"
        table.add_row("GEMINI_API_KEY", OK, f"set ({masked})")
    else:
        table.add_row("GEMINI_API_KEY", FAIL, "not set - copy .env.example to .env")
        problems.append("GEMINI_API_KEY is not set")

    # --- cache and database --------------------------------------------------
    # Construct the cache first: it creates the DB file and its own tables, so
    # reporting on the file beforehand would show a stale "missing" on first run.
    db_path = settings.resolved_db_path
    cache = LLMCache(db_path, enabled=settings.llm_cache)

    size_kb = db_path.stat().st_size / 1024
    table.add_row("Database", OK, f"{db_path} ({size_kb:,.0f} KB)")

    stats = cache.stats()
    mode = "enabled" if settings.llm_cache else "DISABLED"
    if settings.offline:
        mode += ", OFFLINE replay (cache misses will raise)"
    table.add_row(
        "LLM cache",
        OK if settings.llm_cache else WARN,
        f"{mode} - {stats['cached_responses']} responses,"
        f" {stats['cached_embeddings']} embeddings",
    )

    calls = stats["calls_today"]
    if calls:
        summary = ", ".join(f"{model}: {n}" for model, n in calls.items())
    else:
        summary = "no API calls today"
    limit = settings.daily_call_limit
    table.add_row(
        "Daily calls",
        OK,
        f"{summary} (guard: {limit if limit else 'off'})",
    )

    # --- models --------------------------------------------------------------
    configured = {
        "generation": settings.gen_model,
        "generation (heavy)": settings.gen_model_heavy,
        "embedding": settings.embed_model,
    }

    if check_models and settings.has_key:
        try:
            with GeminiClient(settings, cache) as client:
                available = {
                    m["name"].removeprefix("models/") for m in client.list_models()
                }
        except LLMError as exc:
            table.add_row("Gemini API", FAIL, str(exc)[:160])
            problems.append("could not reach the Gemini API")
            available = None
        else:
            table.add_row("Gemini API", OK, f"{len(available)} models visible")

        if available is not None:
            for label, model in configured.items():
                if model in available:
                    price = pricing.lookup(model)
                    detail = model
                    if price:
                        detail += (
                            f"  (${price.input_per_m:g}/${price.output_per_m:g} per 1M)"
                        )
                    else:
                        detail += "  (price unknown - cost will report as null)"
                    table.add_row(f"Model: {label}", OK, detail)
                else:
                    table.add_row(f"Model: {label}", FAIL, f"{model} not available")
                    problems.append(f"model {model} is not available to this key")
    else:
        for label, model in configured.items():
            table.add_row(f"Model: {label}", WARN, f"{model} (not verified)")

    table.add_row("Embedding dim", OK, str(settings.embed_dim))

    # --- retrieval readiness -------------------------------------------------
    from kivi.db.connection import vec_available
    from kivi.retrieval.embed import embedding_coverage

    with kivi_db.connect(db_path) as conn:
        version = kivi_db.current_version(conn)
        pending = kivi_db.pending_migrations(conn)
        try:
            done, total = embedding_coverage(conn, settings.embed_model, settings.embed_dim)
        except Exception:
            done, total = 0, 0
        has_vec = vec_available(conn)

    if pending:
        table.add_row(
            "Schema", WARN,
            f"v{version}, {len(pending)} migration(s) pending - run `kivi init-db`",
        )
    else:
        table.add_row("Schema", OK, f"v{version}, up to date")

    if total == 0:
        table.add_row("Episodes", WARN, "none imported yet - run `kivi import`")
    elif done < total:
        table.add_row(
            "Embeddings", WARN,
            f"{done:,}/{total:,} episodes embedded - run `kivi embed`",
        )
    else:
        table.add_row("Embeddings", OK, f"{done:,}/{total:,} episodes embedded")

    table.add_row(
        "Vector index", OK if has_vec else WARN,
        "sqlite-vec loaded" if has_vec
        else "sqlite-vec unavailable; exact numpy fallback in use",
    )
    table.add_row("Pricing data", OK, f"as of {pricing.PRICING_AS_OF}")

    console.print(table)

    if problems:
        console.print()
        for problem in problems:
            console.print(f"  [red]x[/red] {problem}")
        raise typer.Exit(code=1)

    console.print("\n[green]All checks passed.[/green]")


@app.command()
def smoke() -> None:
    """Send one tiny generation and one embedding, end to end. Costs a fraction of a cent."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    with GeminiClient(settings) as client:
        result = client.generate(
            "Reply with exactly one word: ready",
            max_output_tokens=2048,
        )
        console.print(
            f"[bold]generate[/bold] ({result.model}): {result.text.strip()!r}"
            f"  tokens {result.tokens_in}/{result.tokens_out}"
            f"  cost {_fmt_cost(result.cost_usd)}"
            f"  {result.latency_ms} ms"
            f"{'  [dim](cached)[/dim]' if result.cached else ''}"
        )

        vector = client.embed_one("Umar and Abhi are working on DSPM project tickets")
        console.print(
            f"[bold]embed[/bold] ({settings.embed_model}): dim={len(vector)}"
            f"  first 3 = {[round(v, 4) for v in vector[:3]]}"
        )

    console.print("\n[green]Smoke test passed.[/green]")


@app.command("gen-corpus")
def gen_corpus(
    profile: str = typer.Option(
        "all", help="office | house | mixed | all"
    ),
    count: int = typer.Option(500, help="Records per profile."),
    seed: int = typer.Option(7, help="Plan seed. Same seed gives the same plan."),
    workers: int = typer.Option(4, help="Parallel generation batches."),
    out_dir: Path = typer.Option(Path("data/corpus"), help="Where to write."),
    ingest_after: bool = typer.Option(
        True, "--ingest/--no-ingest", help="Import each corpus after generating it."
    ),
) -> None:
    """Generate development corpora with deterministic ground truth.

    The plan - what each record must convey - is decided in seeded Python before
    any model call. The model only writes the surface text, so the ground truth
    is independent of what the model chose to say.
    """
    from kivi.corpus.generate import generate
    from kivi.corpus.profiles import PROFILES

    settings = get_settings()
    _configure_logging(settings.log_level)

    keys = list(PROFILES) if profile == "all" else [profile]
    for key in keys:
        if key not in PROFILES:
            console.print(f"[red]Unknown profile {key!r}.[/red] Choose from {list(PROFILES)}.")
            raise typer.Exit(code=1)

    grand_total = 0
    with GeminiClient(settings) as client:
        for key in keys:
            console.print(f"\n[bold]Generating {count} records for [cyan]{key}[/cyan][/bold]")
            with console.status(f"calling {settings.gen_model}...") as status:
                def progress(done: int, _status=status, _key=key) -> None:
                    _status.update(f"{_key}: {done}/{count} records written")

                corpus_path, truth_path, records, plan = generate(
                    key, count, client, seed=seed, out_dir=out_dir,
                    workers=workers, progress=progress,
                )

            planted = sum(1 for r in records if r["meta"]["planted_kind"] != "filler")
            console.print(
                f"  [green]{len(records)} records[/green] -> {corpus_path}"
                f"\n  {planted} planted, {len(plan.plants)} plants,"
                f" {len(plan.questions)} questions -> {truth_path}"
            )
            grand_total += len(records)

            if ingest_after:
                with kivi_db.connect(settings.resolved_db_path) as conn:
                    kivi_db.migrate(conn)
                    report = ingest_mod.ingest_file(conn, corpus_path, source=key)
                console.print(f"  ingested: {report.summary()}")

    console.print(f"\n[green]Done.[/green] {grand_total} records generated.")


@app.command()
def embed(
    batch: int = typer.Option(100, help="Texts per embedding request."),
) -> None:
    """Embed episodes that do not have a vector yet. Resumable and idempotent."""
    from kivi.retrieval.embed import embed_episodes, embedding_coverage

    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        with GeminiClient(settings) as client:
            done, total = embedding_coverage(
                conn, settings.embed_model, settings.embed_dim
            )
            if done >= total and total:
                console.print(f"[green]All {total:,} episodes already embedded.[/green]")
                return
            with console.status("embedding...") as status:
                def progress(n: int, of: int) -> None:
                    status.update(f"embedded {n:,}/{of:,}")

                stats_out = embed_episodes(
                    conn, client, batch=batch, progress=progress
                )
        done, total = embedding_coverage(conn, settings.embed_model, settings.embed_dim)

    console.print(
        f"[green]{stats_out['embedded']:,} embedded.[/green]"
        f"  coverage {done:,}/{total:,}"
        f"  vec0 index: {'yes' if stats_out.get('vec_index') else 'no (numpy fallback)'}"
    )


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for."),
    limit: int = typer.Option(8, help="Results to show."),
    app_filter: str = typer.Option(None, "--app", help="Filter by application."),
    since: str = typer.Option(None, "--since", help="e.g. 7d, 24h, or 2026-08-01."),
    hour: str = typer.Option(
        None, "--hour", help="Local hour window, e.g. '16-18' for 'around 5 PM'."
    ),
    no_vector: bool = typer.Option(False, "--no-vector", help="Ablate the vector ranker."),
    bm25: bool = typer.Option(
        False, "--bm25", help="Add the lexical ranker (off by default; see the ablation)."
    ),
    explain: bool = typer.Option(False, "--explain", help="Show per-ranker positions."),
) -> None:
    """Hybrid search over episodes: vector + bm25 + recency, fused with RRF."""
    from kivi.retrieval.search import Filters, search as run_search

    settings = get_settings()
    _configure_logging(settings.log_level)

    filters = Filters(app=app_filter)
    if since:
        filters.after = _parse_since(since)
    if hour:
        lo, _, hi = hour.partition("-")
        filters.local_hour_min = int(lo)
        filters.local_hour_max = int(hi or lo)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        with GeminiClient(settings) as client:
            result = run_search(
                conn, client, query, limit=limit, filters=filters,
                use_vector=not no_vector, use_bm25=bm25,
            )

    if not result.hits:
        console.print("[yellow]Nothing matched.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("When", no_wrap=True)
    table.add_column("App", no_wrap=True)
    table.add_column("RRF", justify="right", no_wrap=True)
    if explain:
        table.add_column("vec/bm25/rec", no_wrap=True)
    table.add_column("Formatted")
    for hit in result.hits:
        cells = [hit.ts[:16].replace("T", " "), hit.app or "-", f"{hit.score:.4f}"]
        if explain:
            cells.append(
                "/".join(
                    str(hit.ranks.get(name, "-")) for name in ("vector", "bm25", "recency")
                )
            )
        text = hit.formatted
        cells.append(text if len(text) <= 86 else text[:83] + "...")
        table.add_row(*cells)
    console.print(table)
    console.print(
        f"[dim]{result.latency_ms} ms   {result.candidate_count} candidates fused"
        f"   rankers: {', '.join(result.rankers)}"
        f"   {'filtered' if result.filtered else 'unfiltered'}"
        f"   vec0: {'yes' if result.used_vec_index else 'numpy'}[/dim]"
    )


@app.command("eval-retrieval")
def eval_retrieval(
    corpus_dir: Path = typer.Option(Path("data/corpus"), help="Where ground truth lives."),
    out_dir: Path = typer.Option(Path("eval/results"), help="Where to write results."),
    limit: int = typer.Option(10, help="Depth to retrieve."),
    ablate: bool = typer.Option(
        True, "--ablate/--no-ablate",
        help="Also run vector-only and bm25-only, to show what hybrid buys.",
    ),
) -> None:
    """Score retrieval against planted ground truth, before memory is involved.

    A failure here is a retrieval bug. A failure later with the evidence present
    is a reasoning bug. Keeping them separable is the point of running this alone.
    """
    from kivi.evaluation.retrieval import run_retrieval_eval, write_results

    settings = get_settings()
    _configure_logging(settings.log_level)

    truths = sorted(corpus_dir.glob("*_ground_truth.json"))
    if not truths:
        console.print(f"[yellow]No ground truth found in {corpus_dir}.[/yellow]")
        raise typer.Exit(code=1)

    configs = [("hybrid", True, True, False)]
    if ablate:
        configs += [
            ("vector-only", True, False, False),
            ("bm25-only", False, True, False),
            ("hybrid+recency", True, True, True),
        ]

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        with GeminiClient(settings) as client:
            for truth_path in truths:
                corpus = truth_path.stem.replace("_ground_truth", "")
                console.print(f"\n[bold cyan]{corpus}[/bold cyan]")

                table = Table(show_header=True, header_style="bold")
                table.add_column("Config")
                table.add_column("n", justify="right")
                for k in (1, 3, 5, 10):
                    table.add_column(f"hit@{k}", justify="right")
                table.add_column("recall", justify="right")
                table.add_column("MRR", justify="right")
                table.add_column("p50", justify="right")
                table.add_column("p95", justify="right")

                first: Any = None
                for name, vec, bm, rec in configs:
                    with console.status(f"{corpus} / {name}..."):
                        result = run_retrieval_eval(
                            conn, client, truth_path, limit=limit,
                            use_vector=vec, use_bm25=bm, use_recency=rec,
                            config_name=name, source=corpus,
                        )
                    summary = result.summary()
                    if not summary:
                        continue
                    write_results(result, out_dir)
                    if first is None:
                        first = result
                    table.add_row(
                        name, str(summary["questions"]),
                        *[f"{summary[f'hit@{k}']:.3f}" for k in (1, 3, 5, 10)],
                        f"{summary['recall']:.3f}", f"{summary['mrr']:.3f}",
                        f"{summary['p50_ms']}ms", f"{summary['p95_ms']}ms",
                    )
                console.print(table)

                if first is not None:
                    kind_table = Table(show_header=True, header_style="bold")
                    kind_table.add_column("Question kind (hybrid)")
                    kind_table.add_column("n", justify="right")
                    for k in (1, 5, 10):
                        kind_table.add_column(f"hit@{k}", justify="right")
                    kind_table.add_column("recall", justify="right")
                    for kind, row in first.by_kind().items():
                        kind_table.add_row(
                            kind, str(row["n"]),
                            *[f"{row[f'hit@{k}']:.3f}" for k in (1, 5, 10)],
                            f"{row['recall']:.3f}",
                        )
                    console.print(kind_table)

    console.print(f"\n[green]Per-question results written to {out_dir}.[/green]")


def _parse_since(value: str):
    import datetime as _dt

    now = _dt.datetime.now(_dt.timezone.utc)
    text = value.strip().lower()
    if text.endswith("d") and text[:-1].isdigit():
        return now - _dt.timedelta(days=int(text[:-1]))
    if text.endswith("h") and text[:-1].isdigit():
        return now - _dt.timedelta(hours=int(text[:-1]))
    if text.endswith("w") and text[:-1].isdigit():
        return now - _dt.timedelta(weeks=int(text[:-1]))
    return _dt.datetime.fromisoformat(value).replace(tzinfo=_dt.timezone.utc)


@app.command()
def extract(
    limit: int = typer.Option(None, help="Only process this many episodes."),
    source: str = typer.Option(None, help="Restrict to one corpus (office/house/mixed)."),
    batch: int = typer.Option(8, help="Episodes per extraction call."),
    heavy: bool = typer.Option(False, "--heavy", help="Use the heavy model."),
    embed_after: bool = typer.Option(True, "--embed/--no-embed", help="Embed new memories."),
    redo: bool = typer.Option(
        False, "--redo",
        help="Clear all learned memory and re-run. Episodes and the model cache "
             "are kept, so this is free after the first run.",
    ),
) -> None:
    """Run the memory pipeline: salience -> extraction -> promotion.

    Idempotent. An episode that already has an `extract` trace is skipped, so a
    run that dies partway can simply be repeated.
    """
    from kivi.memory.pipeline import extract_all
    from kivi.memory.promote import expire_commitments
    from kivi.retrieval.embed import embed_memories

    settings = get_settings()
    _configure_logging(settings.log_level)
    model = settings.gen_model_heavy if heavy else settings.gen_model

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)

        if redo:
            # Everything derived from episodes, nothing that cost money.
            for table in (
                "memory_sources", "memory_vectors", "edges", "memories",
                "candidate_evidence", "candidate_surface", "candidates",
                "relations",
            ):
                conn.execute(f"DELETE FROM {table}")
            conn.execute("DELETE FROM traces WHERE kind = 'extract'")
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            conn.commit()
            console.print("[dim]Cleared learned memory; episodes and cache kept.[/dim]")

        with GeminiClient(settings) as client:
            with console.status("extracting...") as status:
                def progress(done: int, total: int) -> None:
                    status.update(f"extracted {done:,}/{total:,} episodes")

                report = extract_all(
                    conn, client, limit=limit, source=source,
                    batch_size=batch, model=model, progress=progress,
                )
            expired = expire_commitments(conn)
            conn.commit()

            if embed_after:
                with console.status("embedding memories..."):
                    embed_memories(conn, client)

        counts = conn.execute(
            "SELECT status, COUNT(*) AS n FROM memories GROUP BY status"
        ).fetchall()

    console.print(f"[green]{report.summary()}[/green]")
    console.print(
        f"  edges: {report.edges}   expired commitments: {expired}"
        f"   tokens {report.tokens_in:,}/{report.tokens_out:,}"
        f"   cost ${report.cost_usd:.4f}"
    )
    console.print(
        f"  personalization: {report.profiles} profiles built, "
        f"{report.styles_read} re-read by the model"
    )
    if counts:
        console.print(
            "  memories  " + "   ".join(f"{r['status']}: {r['n']}" for r in counts)
        )
    if report.reject_reasons:
        console.print("\n[bold]Why candidates were refused[/bold]")
        for reason, n in sorted(report.reject_reasons.items(), key=lambda x: -x[1]):
            console.print(f"  {n:5}  {reason}")


@app.command()
def memories(
    limit: int = typer.Option(20, help="How many to show."),
    type_filter: str = typer.Option(None, "--type", help="entity | preference | commitment"),
    status: str = typer.Option("active", help="active | superseded | expired | all"),
) -> None:
    """List what Kivi has learned, with its provenance count."""
    settings = get_settings()
    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        sql = (
            "SELECT m.*, (SELECT COUNT(*) FROM memory_sources s"
            "             WHERE s.memory_id = m.id) AS sources FROM memories m WHERE 1=1"
        )
        params: list = []
        if status != "all":
            sql += " AND m.status = ?"
            params.append(status)
        if type_filter:
            sql += " AND m.type = ?"
            params.append(type_filter)
        sql += " ORDER BY m.confidence DESC, m.subject LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        console.print("[yellow]No memories yet. Run `kivi extract`.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Type", no_wrap=True)
    table.add_column("Subject", no_wrap=True)
    table.add_column("Conf", justify="right", no_wrap=True)
    table.add_column("Src", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Body")
    for row in rows:
        body = row["body"] or ""
        table.add_row(
            row["type"], row["subject"][:28], f"{row['confidence']:.2f}",
            str(row["sources"]), row["status"],
            body if len(body) <= 60 else body[:57] + "...",
        )
    console.print(table)


@app.command()
def candidates(
    limit: int = typer.Option(20, help="How many to show."),
    status: str = typer.Option("rejected", help="rejected | pending | promoted | all"),
) -> None:
    """What Kivi considered and refused, with the reason.

    This is the "what did it deliberately ignore" surface. A rejection that is
    only a claim in a README cannot be checked; this is a table.
    """
    settings = get_settings()
    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        sql = "SELECT * FROM candidates WHERE 1=1"
        params: list = []
        if status != "all":
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY evidence_count DESC, last_seen DESC LIMIT ?"
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        by_reason = conn.execute(
            "SELECT status, COUNT(*) AS n FROM candidates GROUP BY status"
        ).fetchall()

    if not rows:
        console.print("[yellow]No candidates match.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Type", no_wrap=True)
    table.add_column("Subject", no_wrap=True)
    table.add_column("Ev", justify="right", no_wrap=True)
    table.add_column("Stance", no_wrap=True)
    table.add_column("Reason")
    for row in rows:
        table.add_row(
            row["type"], row["subject"][:26], str(row["evidence_count"]),
            row["stance"], (row["reason"] or "")[:70],
        )
    console.print(table)
    console.print(
        "\n" + "   ".join(f"{r['status']}: {r['n']}" for r in by_reason)
    )


@app.command()
def recall(
    query: str = typer.Argument(..., help="What to recall."),
    limit: int = typer.Option(8, help="Direct memory hits."),
    no_graph: bool = typer.Option(False, "--no-graph", help="Ablate graph expansion."),
    show_sources: bool = typer.Option(False, "--sources", help="Show source episodes."),
) -> None:
    """Search memories, widened by one graph hop, with provenance."""
    from kivi.memory.search import recall as run_recall

    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        with GeminiClient(settings) as client:
            result = run_recall(
                conn, client, query, limit=limit, use_graph=not no_graph
            )
        episodes_by_id = {}
        if show_sources and result.episode_ids:
            placeholders = ",".join("?" * len(result.episode_ids))
            episodes_by_id = {
                row["id"]: row
                for row in conn.execute(
                    f"SELECT id, ts, app, formatted FROM episodes"
                    f" WHERE id IN ({placeholders})",
                    result.episode_ids,
                ).fetchall()
            }

    if not result.hits:
        console.print("[yellow]Nothing recalled.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Via", no_wrap=True)
    table.add_column("Type", no_wrap=True)
    table.add_column("Subject", no_wrap=True)
    table.add_column("Conf", justify="right", no_wrap=True)
    table.add_column("Src", justify="right", no_wrap=True)
    table.add_column("Body")
    for hit in result.hits:
        style = "dim" if hit.via.startswith("graph") else ""
        table.add_row(
            hit.via, hit.type, hit.subject[:26], f"{hit.confidence:.2f}",
            str(len(hit.sources)),
            (hit.body or "")[:52], style=style,
        )
    console.print(table)
    console.print(
        f"[dim]{result.latency_ms} ms   {len(result.hits)} hits"
        f"   {result.expanded} via graph   {len(result.episode_ids)} source episodes[/dim]"
    )

    if show_sources:
        console.print("\n[bold]Provenance[/bold]")
        for episode_id in result.episode_ids[:12]:
            row = episodes_by_id.get(episode_id)
            if row:
                console.print(
                    f"  [dim]{row['ts'][:16]} {row['app'] or '-':10}[/dim] {row['formatted'][:82]}"
                )


@app.command()
def ask(
    question: str = typer.Argument(..., help="What to ask Hey Kivi."),
    verbose: bool = typer.Option(False, "-v", help="Show tool calls and evidence."),
) -> None:
    """Ask Hey Kivi a question, from the terminal."""
    from kivi.agent import ask as run_ask

    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        with GeminiClient(settings) as client:
            with console.status("thinking..."):
                answer = run_ask(conn, client, question)

    console.print()
    if answer.abstained:
        console.print(f"[yellow]{answer.answer}[/yellow]")
        console.print(f"[dim]abstained: {answer.abstain_reason}[/dim]")
    else:
        # markup=False: citations look like [ep_1a2b3c], which rich would parse
        # as style tags and silently delete - swallowing the very thing that
        # makes the answer checkable.
        console.print(answer.answer, markup=False)

    console.print(
        f"\n[dim]{answer.latency_ms} ms · {answer.rounds} round(s) · "
        f"{len(answer.tool_calls)} tool call(s) · {len(answer.citations)} citation(s) · "
        f"${answer.cost_usd:.6f}[/dim]"
    )
    if answer.dropped_citations:
        console.print(
            f"[red]{len(answer.dropped_citations)} unverifiable reference(s) removed: "
            f"{', '.join(answer.dropped_citations)}[/red]"
        )

    if verbose:
        for call in answer.tool_calls:
            console.print(f"\n[bold]{call['tool']}[/bold] {call['args']}")
            result = call["result"]
            if isinstance(result, dict):
                console.print(f"  [dim]{str(result)[:400]}[/dim]")
        if answer.sources:
            console.print("\n[bold]Evidence available[/bold]")
            for source in answer.sources[:12]:
                console.print(
                    f"  [{source['kind']}] {source['ref']}  {source['text'][:78]}"
                )


@app.command()
def policy() -> None:
    """Show the memory policy currently in force."""
    import json as _json

    from kivi.memory import policy as pol

    console.print_json(_json.dumps(pol.describe(), indent=2))


@app.command("eval-memory")
def eval_memory(
    corpus_dir: Path = typer.Option(Path("data/corpus"), help="Where ground truth lives."),
    graph: bool = typer.Option(True, "--graph/--no-graph", help="Run the graph ablation."),
) -> None:
    """Score the memory stage: stance guard, store growth, graph value."""
    from kivi.evaluation.memory import (
        evaluate_graph, evaluate_stance_guard, memory_growth,
    )

    settings = get_settings()
    _configure_logging(settings.log_level)
    truths = sorted(corpus_dir.glob("*_ground_truth.json"))
    if not truths:
        console.print(f"[yellow]No ground truth in {corpus_dir}.[/yellow]")
        raise typer.Exit(code=1)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)

        # 1. The privacy claim.
        stance = evaluate_stance_guard(conn, truths)
        console.print("\n[bold]1. Stance guard[/bold] - episodes about other people")
        colour = "green" if not stance.content_leaks else "red"
        console.print(
            f"  [{colour}]no sensitive content stored:"
            f" {stance.total - len(stance.content_leaks)}/{stance.total}"
            f"  ({stance.content_precision * 100:.1f}%)[/{colour}]"
        )
        console.print(
            f"  [dim]nothing at all learned:"
            f" {stance.protected_episodes}/{stance.total}"
            f"  ({stance.strict_precision * 100:.1f}%)[/dim]"
        )
        for leak in stance.content_leaks[:8]:
            console.print(f"  [red]CONTENT LEAK[/red] {leak['episode']}: {leak['text'][:96]}")
            for memory in leak["memories"]:
                console.print(f"      -> {memory['type']}: {memory['subject']}")
        for leak in stance.name_only_leaks[:5]:
            names = ", ".join(m["subject"] for m in leak["memories"])
            console.print(
                f"  [yellow]name only[/yellow] {leak['episode']}: learned {names}"
                f" [dim](nothing sensitive)[/dim]"
            )
        if len(stance.name_only_leaks) > 5:
            console.print(f"  ... and {len(stance.name_only_leaks) - 5} more name-only")

        # 2. Does the store stay small?
        console.print("\n[bold]2. Store growth[/bold] - must be sublinear")
        growth = memory_growth(conn)
        table = Table(show_header=True, header_style="bold")
        table.add_column("Episodes", justify="right")
        table.add_column("Memories", justify="right")
        table.add_column("Ratio", justify="right")
        for point in growth:
            table.add_row(
                f"{point['episodes']:,}", f"{point['memories']:,}",
                f"{point['memories'] / point['episodes'] * 100:.1f}%",
            )
        console.print(table)

        counts = conn.execute(
            "SELECT status, COUNT(*) AS n FROM memories GROUP BY status"
        ).fetchall()
        rejected = conn.execute(
            "SELECT COUNT(*) AS n FROM candidates WHERE status = 'rejected'"
        ).fetchone()["n"]
        edges = conn.execute("SELECT COUNT(*) AS n FROM edges").fetchone()["n"]
        console.print(
            "  " + "   ".join(f"{r['status']}: {r['n']}" for r in counts)
            + f"   rejected candidates: {rejected}   edges: {edges}"
        )

        # 3. Does the graph earn its place?
        if graph:
            console.print("\n[bold]3. Graph ablation[/bold] - distributed facts only")
            with GeminiClient(settings) as client:
                with console.status("running recall with and without graph..."):
                    ablations = evaluate_graph(conn, client, truths)
            if not ablations:
                console.print("  [yellow]no distributed-fact questions found[/yellow]")
            else:
                without = sum(a.found_without for a in ablations)
                with_g = sum(a.found_with for a in ablations)
                expected = sum(len(a.expected) for a in ablations)
                improved = sum(1 for a in ablations if a.found_with > a.found_without)
                console.print(
                    f"  members recovered without graph: {without}/{expected}"
                    f"  ({without / expected * 100:.1f}%)"
                )
                console.print(
                    f"  members recovered with graph:    {with_g}/{expected}"
                    f"  ({with_g / expected * 100:.1f}%)"
                )
                delta = with_g - without
                colour = "green" if delta > 0 else ("yellow" if delta == 0 else "red")
                console.print(
                    f"  [{colour}]delta {delta:+d} across {len(ablations)} questions;"
                    f" {improved} improved[/{colour}]"
                )


@app.command("gen-relational")
def gen_relational(
    seed: int = typer.Option(11, help="Generation seed."),
    out_dir: Path = typer.Option(Path("data/corpus"), help="Where to write."),
    ingest_after: bool = typer.Option(True, "--ingest/--no-ingest"),
) -> None:
    """Generate dictations addressed to seven specific people, in their styles.

    Two bosses, two friends, three family, at deliberately uneven volumes so the
    evaluation can plot adherence against how much evidence a profile has.
    """
    from kivi.corpus.recipients import RECIPIENTS
    from kivi.corpus.relational import generate_all

    settings = get_settings()
    _configure_logging(settings.log_level)

    total = sum(r.volume for r in RECIPIENTS)
    console.print(
        f"[bold]{len(RECIPIENTS)} recipients, {total} dictations[/bold]  "
        + "  ".join(f"{r.name}({r.volume})" for r in RECIPIENTS)
    )

    with GeminiClient(settings) as client:
        with console.status("generating...") as status:
            def progress(n: int) -> None:
                status.update(f"{n}/{total} written")

            corpus_path, truth_path, records = generate_all(
                client, seed=seed, out_dir=out_dir, progress=progress
            )

    console.print(f"[green]{len(records)} records[/green] -> {corpus_path}")

    if ingest_after:
        with kivi_db.connect(settings.resolved_db_path) as conn:
            kivi_db.migrate(conn)
            report = ingest_mod.ingest_file(conn, corpus_path, source="relational")
        console.print(f"  ingested: {report.summary()}")


@app.command("learn-style")
def learn_style(
    read: bool = typer.Option(
        True, "--read/--counts-only",
        help="Also have the model read the messages and describe the style. "
             "One call for all recipients; the counted features are computed "
             "either way.",
    ),
) -> None:
    """Rebuild every recipient's writing profile from the dictations."""
    from kivi.memory import style_llm
    from kivi.memory.style import MIN_SAMPLES, rebuild_profiles

    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        profiles = rebuild_profiles(conn)

        if read and profiles:
            with GeminiClient(settings) as client:
                with console.status("reading the messages..."):
                    learned = style_llm.learn_styles(conn, client)
                    saved = style_llm.save(conn, learned)
            console.print(f"[dim]model described {saved} recipients[/dim]\n")
            for item in learned:
                console.print(f"[bold]{item.recipient}[/bold] "
                              f"[dim]({item.relation})[/dim]  {item.summary}")
                console.print(f"  [dim]vs others:[/dim] {item.distinctive}")
                for rule in item.rules:
                    console.print(f"    - {rule}")
                console.print()

    if not profiles:
        console.print("[yellow]No dictations carry a recipient yet.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("To", no_wrap=True)
    table.add_column("Relation", no_wrap=True)
    table.add_column("n", justify="right", no_wrap=True)
    table.add_column("Greeting", no_wrap=True)
    table.add_column("Words", justify="right", no_wrap=True)
    table.add_column("Formality", justify="right", no_wrap=True)
    table.add_column("How they write to them")
    for profile in profiles:
        greeting = profile.greeting or "-"
        if profile.greeting and profile.greeting_rate < 0.5:
            greeting += f" ({profile.greeting_rate:.0%})"
        table.add_row(
            profile.display, profile.relation, str(profile.n_samples), greeting,
            f"{profile.mean_words:.0f}", f"{profile.formality:.2f}",
            profile.describe(),
            style="" if profile.active else "dim",
        )
    console.print(table)
    console.print(
        f"[dim]A profile is applied once it has {MIN_SAMPLES}+ dictations; "
        f"dimmed rows have fewer.[/dim]"
    )


@app.command("eval-personalization")
def eval_personalization(
    drafts: int = typer.Option(3, help="Drafts per recipient, per condition."),
    out_dir: Path = typer.Option(Path("eval/results"), help="Where to write."),
) -> None:
    """Does the learned style actually change what Kivi writes?

    Same source note, same task, drafted for each person with and without their
    learned profile. Scored by counting, not by an LLM judge.
    """
    from kivi.evaluation.personalization import (
        run_personalization_eval, write_results,
    )

    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        with GeminiClient(settings) as client:
            with console.status("drafting...") as status:
                def progress(n: int, of: int) -> None:
                    status.update(f"{n}/{of} drafts")

                result = run_personalization_eval(
                    conn, client, drafts_per_recipient=drafts, progress=progress
                )
        path = write_results(result, out_dir)

    table = Table(show_header=True, header_style="bold")
    table.add_column("To", no_wrap=True)
    table.add_column("n", justify="right", no_wrap=True)
    table.add_column("off", justify="right")
    table.add_column("on", justify="right")
    table.add_column("delta", justify="right")
    table.add_column("greeting", justify="right")
    table.add_column("words off/on", justify="right")

    rows = sorted(result.by_recipient().items(), key=lambda kv: -kv[1]["n_samples"])
    for name, row in rows:
        delta = row["with_style"] - row["without_style"]
        colour = "green" if delta > 0.05 else ("yellow" if delta >= -0.01 else "red")
        table.add_row(
            name, str(row["n_samples"]),
            f"{row['without_style']:.2f}", f"{row['with_style']:.2f}",
            f"[{colour}]{delta:+.2f}[/{colour}]",
            f"{row['greeting_off']:.1f} -> {row['greeting_on']:.1f}",
            f"{row['mean_words_off']:.0f} -> {row['mean_words_on']:.0f}",
        )
    console.print(table)

    overall = result.overall()
    if overall:
        console.print(
            f"\n[bold]Overall[/bold]  without style {overall['without_style']:.3f}"
            f"  ->  with style {overall['with_style']:.3f}"
            f"  ([green]{overall['delta']:+.3f}[/green])"
        )
    console.print(f"[dim]Per-draft detail written to {path}[/dim]")


@app.command("exp-amma")
def exp_amma(
    count: int = typer.Option(45, help="Messages to generate."),
    parts: int = typer.Option(3, help="Chronological parts to learn across."),
    regenerate: bool = typer.Option(False, "--regenerate", help="Rebuild the log."),
) -> None:
    """Can the model learn a CONDITIONAL habit from messages alone?

    The corpus addresses the person's mother as "Mama" when the message is
    affectionate and "Amma" when it is routine. The learner is never told this.
    It is fed the log in three chronological parts and, after each, asked to
    write new messages - so the test is whether the rule shows up in output, not
    whether it can be described.
    """
    from kivi.experiments import amma as exp
    from kivi.experiments import run_amma

    settings = get_settings()
    _configure_logging(settings.log_level)

    with GeminiClient(settings) as client:
        records = [] if regenerate else exp.load_log()
        if not records:
            with console.status(f"generating {count} messages..."):
                records = exp.generate(client, count=count)
            exp.write_log(records)
        console.print(f"[dim]{len(records)} messages -> {exp.LOG_PATH}[/dim]")

        # Gate: an experiment run on a corpus that lacks the rule measures nothing.
        fidelity = exp.verify(records)
        colour = "green" if fidelity["fidelity"] >= 0.85 else "red"
        console.print(
            f"corpus fidelity: [{colour}]{fidelity['fidelity']:.0%}[/{colour}]"
            f"  ({fidelity['correct']} correct, {fidelity['wrong']} wrong form,"
            f" {fidelity['neither']} no name)   Mama {fidelity['mama']} /"
            f" Amma {fidelity['amma']}\n"
        )
        if fidelity["fidelity"] < 0.6:
            console.print("[red]The planted rule is not reliably in the data. "
                          "Regenerate before trusting anything below.[/red]")

        with console.status("learning across parts...") as status:
            def progress(stage: int, of: int) -> None:
                status.update(f"part {stage}/{of}")

            results = run_amma.run(client, records, parts=parts, progress=progress)

    for result in results:
        console.print(
            f"\n[bold]── After part {result.stage}  "
            f"({result.messages_seen} messages seen) ──[/bold]"
        )
        console.print(f"  [dim]{result.profile.get('summary', '')}[/dim]")
        for form in result.forms():
            console.print(f"  address form: [cyan]{form}[/cyan]")
        for rule in result.profile.get("conditional_rules", [])[:4]:
            console.print(f"  when [cyan]{rule.get('when')}[/cyan] -> {rule.get('then')}")
        for note in result.profile.get("uncertain", [])[:3]:
            console.print(f"  [yellow]unsure:[/yellow] {note}")

        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        table.add_column(" ", no_wrap=True)
        table.add_column("kind", no_wrap=True)
        table.add_column("asked for", no_wrap=True)
        table.add_column("want", no_wrap=True)
        table.add_column("got", no_wrap=True)
        table.add_column("message")
        for draft in result.drafts:
            table.add_row(
                "[green]OK[/green]" if draft["ok"] else "[red]XX[/red]",
                "unseen" if draft["unseen"] else "seen",
                draft["instruction"][:34], draft["expected"], draft["got"],
                draft["text"][:56],
            )
        console.print(table)
        console.print(
            f"  seen {result.seen_score:.0%}   "
            f"[bold]unseen {result.unseen_score:.0%}[/bold]   "
            f"overall {result.correct}/{len(result.drafts)}"
        )

    out = Path("eval/results/exp_amma.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([
        {
            "stage": r.stage, "messages_seen": r.messages_seen,
            "score": r.score, "seen_score": r.seen_score,
            "unseen_score": r.unseen_score,
            "profile": r.profile, "drafts": r.drafts,
        }
        for r in results
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Full transcript -> {out}[/dim]")


@app.command("exp-mixed")
def exp_mixed(
    regenerate: bool = typer.Option(False, "--regenerate"),
    parts: int = typer.Option(3, help="Learning passes per person."),
) -> None:
    """Three people, three kinds of habit, one mixed log, saved to the database.

    Amma's rule is lexical (which name), Sanjay's is structural (bad news always
    carries a proposal), Priya's is pragmatic (hedging only when nothing is
    urgent). None is described to the learner.
    """
    from kivi.experiments import run_mixed
    from kivi.experiments.patterns import PEOPLE
    from kivi.memory.style import rebuild_profiles

    settings = get_settings()
    _configure_logging(settings.log_level)

    with GeminiClient(settings) as client:
        records = [] if regenerate else run_mixed.load_log()
        if not records:
            with console.status("generating the mixed log..."):
                records = run_mixed.generate(client)
            run_mixed.write_log(records)
        console.print(f"[dim]{len(records)} dictations -> {run_mixed.LOG_PATH}[/dim]\n")

        # Gate: if the rule is not in the data, nothing below means anything.
        table = Table(show_header=True, header_style="bold")
        table.add_column("Person"); table.add_column("Rule kind")
        table.add_column("n", justify="right"); table.add_column("fidelity", justify="right")
        table.add_column("planted rule")
        for person in PEOPLE:
            f = run_mixed.fidelity(person, records)
            colour = "green" if f["rate"] >= 0.85 else "red"
            table.add_row(person.name, person.kind, str(f["n"]),
                          f"[{colour}]{f['rate']:.0%}[/{colour}]", person.rule_summary)
        console.print(table)

        # Through the real pipeline, so the profiles land where the product reads.
        with kivi_db.connect(settings.resolved_db_path) as conn:
            kivi_db.migrate(conn)
            ingest_mod.ingest_file(conn, run_mixed.LOG_PATH, source="experiment")
            rebuild_profiles(conn)

            all_stages: list = []
            for person in PEOPLE:
                with console.status(f"learning {person.name}..."):
                    stages = run_mixed.learn_person(
                        client, person, records, parts=parts
                    )
                all_stages.append((person, stages))
                if stages:
                    run_mixed.persist(conn, person, stages[-1].profile)

    for person, stages in all_stages:
        console.print(f"\n[bold cyan]── {person.name} ({person.kind}) ──[/bold cyan]")
        console.print(f"  [dim]planted: {person.rule_summary}[/dim]")
        for stage in stages:
            console.print(
                f"\n  [bold]after {stage.seen} messages[/bold]   "
                f"seen {stage.seen_score:.0%}   "
                f"[bold]unseen {stage.unseen_score:.0%}[/bold]"
            )
            console.print(f"    [dim]{stage.profile.get('summary', '')[:150]}[/dim]")
            for rule in stage.profile.get("conditional_rules", [])[:3]:
                console.print(
                    f"    when [cyan]{str(rule.get('when'))[:60]}[/cyan]"
                    f" -> {str(rule.get('then'))[:70]}"
                )
        final = stages[-1] if stages else None
        if final:
            for draft in final.drafts:
                if draft["unseen"]:
                    mark = "[green]OK[/green]" if draft["ok"] else "[red]XX[/red]"
                    console.print(
                        f"    {mark} want {draft['expected']:9} got "
                        f"{draft['got']:9} | {draft['text'][:60]}"
                    )

    with kivi_db.connect(settings.resolved_db_path) as conn:
        console.print("\n[bold]Saved to recipient_styles[/bold]")
        for row in conn.execute(
            "SELECT display, relation, n_samples, summary, rules_json"
            " FROM recipient_styles WHERE summary IS NOT NULL AND summary != ''"
        ).fetchall():
            rules = json.loads(row["rules_json"] or "[]")
            console.print(
                f"  [cyan]{row['display']}[/cyan] ({row['relation']}, "
                f"{row['n_samples']} msgs): {row['summary'][:110]}"
            )
            for rule in rules[:3]:
                console.print(f"      - {rule[:100]}")

    out = Path("eval/results/exp_mixed.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([
        {
            "person": person.name, "kind": person.kind,
            "planted": person.rule_summary,
            "stages": [
                {"stage": s.stage, "seen": s.seen, "seen_score": s.seen_score,
                 "unseen_score": s.unseen_score, "profile": s.profile,
                 "drafts": s.drafts}
                for s in stages
            ],
        }
        for person, stages in all_stages
    ], indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"\n[dim]Transcript -> {out}[/dim]")


@app.command("eval-agent")
def eval_agent(
    corpus_dir: Path = typer.Option(Path("data/corpus"), help="Where ground truth lives."),
    per_kind: int = typer.Option(
        10, help="Answerable questions per kind. 0 runs all of them."
    ),
    false_premise: int = typer.Option(
        -1, help="Questions with no answer to include. -1 runs all of them."
    ),
    out_dir: Path = typer.Option(Path("eval/results"), help="Where to write results."),
) -> None:
    """Run Hey Kivi on the planted questions and score what it actually answers.

    The other evaluations stop at retrieval and memory. This one scores the
    answer a person sees: an answerable question must cite real records and
    contain the planted expected answer, and a question the history cannot
    answer must cite nothing. Defaults - every false-premise question plus ten
    of each answerable kind, 56 turns - cost about $1.75. Every question writes
    a row to agent.jsonl.
    """
    from kivi.evaluation.agent import load_cases, run_agent_eval, write_results

    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        cases = load_cases(
            conn, corpus_dir,
            per_kind=per_kind or None,
            false_premise=None if false_premise < 0 else false_premise,
        )
        if not cases:
            console.print(
                f"[yellow]No planted questions found in {corpus_dir} whose "
                f"dictations are imported.[/yellow] Run the imports in RUN.md §4."
            )
            raise typer.Exit(code=1)
        console.print(f"[dim]{len(cases)} questions[/dim]")

        with GeminiClient(settings) as client:
            with console.status("asking...") as status:
                def progress(n: int, of: int) -> None:
                    status.update(f"{n}/{of} answered")

                evaluation = run_agent_eval(
                    conn, client, cases,
                    db_path=settings.resolved_db_path, progress=progress,
                )
        rows_path, summary_path = write_results(evaluation, out_dir)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Question kind")
    table.add_column("n", justify="right")
    table.add_column("passed", justify="right")
    table.add_column("rate", justify="right")
    table.add_column("abstained", justify="right")
    table.add_column("p50", justify="right")
    table.add_column("cost", justify="right")
    for kind, row in evaluation.by_kind().items():
        table.add_row(
            kind, str(row["n"]), str(row["passed"]), f"{row['rate']:.0%}",
            str(row["abstained"]), f"{row['p50_ms']}ms", f"${row['cost_usd']:.3f}",
        )
    console.print(table)

    s = evaluation.summary()
    console.print(
        f"\n[bold]Answerable[/bold]  {s['answerable_passed']}/{s['answerable']}"
        f" cited real records and gave the expected answer ({s['answerable_rate']:.0%});"
        f" {s['unambiguous_passed']}/{s['unambiguous']} excluding questions whose"
        f" answer differs between corpora ({s['unambiguous_rate']:.0%})"
    )
    console.print(
        f"[dim]  stricter: {s['cited_planted']}/{s['answerable']} cited one of the"
        f" exact dictations the generator planted ({s['cited_planted_rate']:.0%})[/dim]"
    )
    console.print(
        f"[bold]False premise[/bold]  {s['false_premise_refused']}/{s['false_premise']}"
        f" made no supported claim ({s['false_premise_rate']:.0%})"
    )
    console.print(
        f"[dim]latency p50 {s['p50_ms']} ms, p95 {s['p95_ms']} ms   "
        f"tokens {s['tokens_in']:,}/{s['tokens_out']:,}   cost ${s['cost_usd']:.4f}"
        f" (${s['cost_per_question_usd']:.4f} a question)   "
        f"database +{s['db_growth_bytes'] / 1024:,.0f} KB[/dim]"
    )

    failures = [r for r in evaluation.results if not r.ok]
    if failures:
        console.print(f"\n[bold]Failures[/bold] - each one is a row in {rows_path}")
        for r in failures:
            console.print(f"  [red]XX[/red] {r.kind:18} {r.question[:70]}")
            console.print(f"       [dim]{r.why}: {r.answer[:110]!r}[/dim]")
    console.print(f"\n[dim]Per-question detail -> {rows_path}\nSummary -> {summary_path}[/dim]")


@app.command("corpus-report")
def corpus_report(
    corpus_dir: Path = typer.Option(Path("data/corpus"), help="Where the corpora live."),
) -> None:
    """Variety report per corpus: planted-case inventory, app mix, time spread.

    The assignment asks whether the corpus is varied enough to exercise the
    system and reveal where it fails, so that has to be inspectable rather than
    asserted.
    """
    import collections
    import json

    files = sorted(corpus_dir.glob("*_episodes.jsonl"))
    if not files:
        console.print(f"[yellow]No corpora found in {corpus_dir}.[/yellow]")
        raise typer.Exit(code=1)

    for path in files:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        name = path.stem.replace("_episodes", "")
        truth_path = corpus_dir / f"{name}_ground_truth.json"
        truth = json.loads(truth_path.read_text(encoding="utf-8")) if truth_path.exists() else {}

        console.print(f"\n[bold cyan]{name}[/bold cyan]  -  {len(records)} records")

        # Not every corpus is planted: the relational one is written to seven
        # people to exercise style, and carries neither field. Reading them
        # strictly crashed the report on that corpus - the first command a
        # reviewer runs from RUN.md's evaluation section.
        kinds = collections.Counter(
            r["meta"].get("planted_kind", "(not planted)") for r in records
        )
        table = Table(show_header=True, header_style="bold")
        table.add_column("Planted case")
        table.add_column("Records", justify="right")
        table.add_column("Share", justify="right")
        table.add_column("Tests")
        for kind, n in kinds.most_common():
            table.add_row(kind, str(n), f"{n / len(records) * 100:.1f}%", _KIND_TESTS.get(kind, ""))
        console.print(table)

        apps = collections.Counter(r["app"] for r in records)
        domains = collections.Counter(r["meta"].get("domain", "-") for r in records)
        stamps = sorted(r["ts"] for r in records)
        words = [len(r["formatted"].split()) for r in records]
        words.sort()

        console.print(
            f"  apps      " + "  ".join(f"{a}: {n}" for a, n in apps.most_common())
        )
        console.print(
            f"  domains   " + "  ".join(f"{d}: {n}" for d, n in domains.most_common())
        )
        console.print(f"  span      {stamps[0][:10]} -> {stamps[-1][:10]}")
        console.print(
            f"  length    median {words[len(words) // 2]} words,"
            f" p10 {words[len(words) // 10]}, p90 {words[int(len(words) * 0.9)]}"
        )

        if truth:
            questions = collections.Counter(q["kind"] for q in truth.get("questions", []))
            console.print(
                f"  plants    {len(truth.get('plants', []))}"
                f"   must-not-learn episodes: {len(truth.get('must_not_learn_episodes', []))}"
            )
            console.print(
                "  questions " + "  ".join(f"{k}: {n}" for k, n in questions.most_common())
            )


_KIND_TESTS = {
    "filler": "ordinary traffic",
    "noise": "must stay searchable but never remembered",
    "third_party": "must produce NO memory (stance guard)",
    "distributed_fact": "answer split across 3 episodes",
    "inferred_preference": "needs 3 occurrences, low confidence",
    "knowledge_update": "old value must retire, not vanish",
    "commitment": "time-anchored, expires",
    "explicit_preference": "learned from a single statement",
}


@app.command()
def cost() -> None:
    """Model usage and spend to date, from the cache ledger."""
    settings = get_settings()
    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        rows = conn.execute(
            "SELECT model, kind, COUNT(*) AS n, SUM(tokens_in) AS ti,"
            " SUM(tokens_out) AS to_, SUM(cost_usd) AS cost"
            " FROM llm_cache GROUP BY model, kind ORDER BY cost DESC"
        ).fetchall()
        calls = conn.execute("SELECT COUNT(*) AS n FROM api_call_log").fetchone()["n"]

    if not rows:
        console.print("[yellow]No model calls recorded yet.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Kind")
    table.add_column("Cached calls", justify="right")
    table.add_column("Tokens in", justify="right")
    table.add_column("Tokens out", justify="right")
    table.add_column("Cost", justify="right")

    total = 0.0
    unknown = False
    for row in rows:
        if row["cost"] is None:
            unknown = True
        else:
            total += row["cost"]
        table.add_row(
            row["model"], row["kind"], f"{row['n']:,}",
            f"{row['ti'] or 0:,}", f"{row['to_'] or 0:,}",
            "unknown" if row["cost"] is None else f"${row['cost']:.4f}",
        )
    console.print(table)
    console.print(f"\n[bold]Total: ${total:.4f}[/bold] across {calls:,} API calls made.")
    if unknown:
        console.print("[yellow]Some models have no published price; those rows are excluded.[/yellow]")


@app.command("init-db")
def init_db() -> None:
    """Create or migrate the database. Safe to run repeatedly."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        before = kivi_db.current_version(conn)
        newly = kivi_db.migrate(conn, verbose=True)
        after = kivi_db.current_version(conn)

    if newly:
        for name in newly:
            console.print(f"  applied [cyan]{name}[/cyan]")
        console.print(
            f"\n[green]Schema at version {after}[/green] (was {before})."
        )
    else:
        console.print(f"[green]Schema already at version {after}.[/green] Nothing to do.")


@app.command("import")
def import_corpus(
    path: Path = typer.Argument(..., help="JSONL corpus file."),
    source: str = typer.Option("import", help="Label recorded on every episode."),
    strict: bool = typer.Option(
        False, "--strict", help="Abort on the first malformed record instead of skipping it."
    ),
) -> None:
    """Import dictation records. Idempotent - re-importing inserts nothing new."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        try:
            report = ingest_mod.ingest_file(conn, path, source=source, strict=strict)
        except (FileNotFoundError, RecordError) as exc:
            console.print(f"[red]Import failed:[/red] {exc}")
            raise typer.Exit(code=1)
        total = conn.execute("SELECT COUNT(*) AS n FROM episodes").fetchone()["n"]

    console.print(f"[green]{report.summary()}[/green]  -  {total} episodes in total.")
    for error in report.errors[:10]:
        console.print(f"  [yellow]skipped[/yellow] {error}")
    if len(report.errors) > 10:
        console.print(f"  ... and {len(report.errors) - 10} more")


@app.command()
def reset(
    include_cache: bool = typer.Option(
        False,
        "--include-cache",
        help="Also clear the model response/embedding cache. Off by default so a "
             "reset does not re-bill the corpus on the next run.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Empty the system: episodes, memories, traces, and conversations."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    if not yes:
        what = "all data AND the model cache" if include_cache else "all data (model cache kept)"
        typer.confirm(f"Delete {what} from {settings.resolved_db_path}?", abort=True)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        deleted = ingest_mod.reset(conn, include_cache=include_cache)

    total = sum(deleted.values())
    console.print(f"[green]Reset complete.[/green] {total} rows deleted.")
    for table, count in deleted.items():
        if count:
            console.print(f"  {table}: {count}")
    if not include_cache:
        console.print("\n[dim]Model cache kept. Use --include-cache to clear it too.[/dim]")


@app.command()
def stats() -> None:
    """Row counts per table, plus episode coverage."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        counts = ingest_mod.stats(conn)
        version = kivi_db.current_version(conn)
        span = conn.execute(
            "SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM episodes"
        ).fetchone()
        by_app = conn.execute(
            "SELECT COALESCE(app, '(none)') AS app, COUNT(*) AS n FROM episodes"
            " GROUP BY app ORDER BY n DESC"
        ).fetchall()
        by_source = conn.execute(
            "SELECT source, COUNT(*) AS n FROM episodes GROUP BY source ORDER BY n DESC"
        ).fetchall()

    size_kb = settings.resolved_db_path.stat().st_size / 1024
    console.print(f"[bold]Schema[/bold] v{version}   [bold]Size[/bold] {size_kb:,.0f} KB")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Table")
    table.add_column("Rows", justify="right")
    for name, count in counts.items():
        style = "dim" if count == 0 else ""
        table.add_row(name, f"{count:,}", style=style)
    console.print(table)

    if span and span["lo"]:
        console.print(f"\n[bold]Episodes span[/bold] {span['lo']}  ->  {span['hi']}")
    if by_source:
        console.print(
            "[bold]By source[/bold]  "
            + "   ".join(f"{r['source']}: {r['n']:,}" for r in by_source)
        )
    if by_app:
        console.print(
            "[bold]By app[/bold]     "
            + "   ".join(f"{r['app']}: {r['n']:,}" for r in by_app)
        )


@app.command()
def episodes(
    limit: int = typer.Option(10, help="How many to show."),
    app_filter: str = typer.Option(None, "--app", help="Filter by application."),
    search: str = typer.Option(None, "--search", help="Full-text search (FTS5/bm25)."),
) -> None:
    """List episodes, newest first. Proves ingest and the FTS index both work."""
    settings = get_settings()
    _configure_logging(settings.log_level)

    with kivi_db.connect(settings.resolved_db_path) as conn:
        kivi_db.migrate(conn)
        if search:
            sql = (
                "SELECT e.id, e.ts, e.app, e.formatted, bm25(episodes_fts) AS score"
                " FROM episodes_fts JOIN episodes e ON e.rowid = episodes_fts.rowid"
                " WHERE episodes_fts MATCH ?"
            )
            params: list = [search]
            if app_filter:
                sql += " AND e.app = ?"
                params.append(app_filter.lower())
            sql += " ORDER BY score LIMIT ?"
            params.append(limit)
        else:
            sql = "SELECT id, ts, app, formatted, NULL AS score FROM episodes"
            params = []
            if app_filter:
                sql += " WHERE app = ?"
                params.append(app_filter.lower())
            sql += " ORDER BY ts_epoch DESC LIMIT ?"
            params.append(limit)
        rows = conn.execute(sql, params).fetchall()

    if not rows:
        console.print("[yellow]No episodes matched.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("When", no_wrap=True)
    table.add_column("App", no_wrap=True)
    if search:
        table.add_column("bm25", justify="right", no_wrap=True)
    table.add_column("Formatted")
    for row in rows:
        cells = [row["ts"][:16].replace("T", " "), row["app"] or "-"]
        if search:
            cells.append(f"{row['score']:.2f}")
        text = row["formatted"]
        cells.append(text if len(text) <= 96 else text[:93] + "...")
        table.add_row(*cells)
    console.print(table)


def _configure_logging(level: str) -> None:
    """Set up logging without leaking the API key.

    httpx logs the full request URL at INFO, and the key travels as a query
    parameter, so an INFO-level run would print the credential to the terminal
    and into any captured log. Pin httpx to WARNING regardless of our own level.
    """
    logging.basicConfig(level=level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def _fmt_cost(cost: float | None) -> str:
    return "unknown" if cost is None else f"${cost:.6f}"


if __name__ == "__main__":
    app()
