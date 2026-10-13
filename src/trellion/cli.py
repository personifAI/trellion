# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Command-line entry point.

Examples:
    trellion build --config configs/build.default.yaml
    trellion query --graph out/learning-graph.csv "What depends on Function?"
    trellion eval  --graph out/learning-graph.csv \
                     --queries src/trellion/eval/golden/learning-graph.jsonl
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from trellion.build.pipeline import run as run_build
from trellion.common.config import BuildConfig
from trellion.eval.benchmark import load_golden, run as run_eval
from trellion.query.answer import answer_query
from trellion.query.graph import CKG

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


@app.callback()
def _boot() -> None:
    """Refuse to run when a secret is delivered both as NAME and NAME_FILE,
    or a named secret file cannot be read (noomesh followup 1247). One line,
    before any command touches the gateway."""
    from trellion.live.secret_env import refuse_on_conflicts

    refuse_on_conflicts()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )


def _resolve_dsn(dsn: str | None) -> str | None:
    """CLI --dsn beats DATABASE_URL beats PAGEINDEX_DSN (same as pack/unpack)."""
    import os

    return dsn or os.environ.get("DATABASE_URL") or os.environ.get("PAGEINDEX_DSN")


@app.command()
def build(
    config: Path = typer.Option(..., exists=True, help="YAML build config."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    dsn: str = typer.Option(None, "--dsn", help="Postgres DSN for the snapshot store."),
    csv_only: bool = typer.Option(
        False, "--csv-only", help="Skip the snapshot store; emit only the debug CSV."
    ),
) -> None:
    """Run the full build pipeline and store the graph as a queryable snapshot.

    The snapshot store (Postgres) is the canonical destination; query it with
    `trellion query --snapshot-id <id>`. The CSV in the output dir is a
    debug/fixture artifact.
    """
    _configure_logging(verbose)
    with config.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = BuildConfig(**raw)
    out = asyncio.run(run_build(cfg))
    console.print(f"[green]Build complete[/] -> {out}")

    if csv_only:
        return
    resolved = _resolve_dsn(dsn)
    if not resolved:
        console.print(
            "[yellow]No DSN — snapshot not stored.[/] Pass --dsn or set "
            "DATABASE_URL to persist a queryable snapshot (CSV is debug-only)."
        )
        return
    from trellion.db.engine import init_engine
    from trellion.snapshots.batch_emit import emit_build_snapshot

    _, async_engine = init_engine(resolved)
    snapshot_id, stats = asyncio.run(
        emit_build_snapshot(
            async_engine,
            out.concepts,
            out.dag_edges,
            out.communities,
            input_dir=cfg.input_dir,
            label=config.stem,
        )
    )
    console.print(
        f"[green]snapshot stored[/] id={snapshot_id} "
        f"concepts={stats['concepts']} edges={stats['dag_edges']}"
    )
    console.print(f"query it: trellion query --snapshot-id {snapshot_id} \"<your question>\"")


@app.command()
def query(
    text: str = typer.Argument(..., help="Natural-language query."),
    snapshot_id: str = typer.Option(
        None, "--snapshot-id", help="Snapshot to query (the canonical path)."
    ),
    graph: Path = typer.Option(
        None, exists=True, help="Debug/fixture CKG CSV produced by `build`."
    ),
    dsn: str = typer.Option(None, "--dsn", help="Postgres DSN (with --snapshot-id)."),
    max_hops: int = typer.Option(5, help="Max hop depth for T3/T5."),
    show_tokens: bool = typer.Option(True, help="Print the retrieval-side token count."),
) -> None:
    """Query a graph — a stored snapshot (canonical) or a debug CSV."""
    if bool(snapshot_id) == bool(graph):
        console.print("[red]Pass exactly one of --snapshot-id or --graph[/]")
        raise typer.Exit(1)
    if snapshot_id:
        resolved = _resolve_dsn(dsn)
        if not resolved:
            console.print("[red]No DSN — pass --dsn or set DATABASE_URL[/]")
            raise typer.Exit(1)
        from trellion.db.engine import init_engine
        from trellion.snapshots import load_ckg_from_snapshot

        _, async_engine = init_engine(resolved)
        ckg = asyncio.run(load_ckg_from_snapshot(async_engine, snapshot_id))
    else:
        ckg = CKG.load(graph)
    ans = answer_query(ckg, text, max_hops=max_hops)
    console.print(f"[bold cyan]{ans.query_type}[/] {ans.content}")
    if show_tokens:
        console.print(
            f"[dim]retrieved_tokens={ans.retrieved_tokens}  concepts={len(ans.concept_ids)}[/]"
        )


@app.command()
def eval(
    graph: Path = typer.Option(..., exists=True),
    queries: Path = typer.Option(..., exists=True, help="JSONL golden query file."),
    max_hops: int = typer.Option(5),
    report: Path = typer.Option(None, help="Optional path to dump the full JSON scorecard."),
) -> None:
    """Run the eval harness: score answers against a golden set and print metrics."""
    ckg = CKG.load(graph)
    golden = load_golden(queries)
    result = run_eval(ckg, golden, max_hops=max_hops)

    # Per-row breakdown.
    t = Table("id", "type", "got", "f1", "tokens", "rds", "path")
    for r in result.rows:
        path_cell = ""
        if r.path_score is not None:
            path_cell = f"exact={'Y' if r.path_score.exact else 'N'}"
        t.add_row(
            r.id,
            r.expected_type,
            r.got_type if r.type_ok else f"[red]{r.got_type}[/]",
            f"{r.set_score.f1:.2f}",
            str(r.tokens),
            f"{r.rds:.4f}",
            path_cell,
        )
    console.print(t)

    s = result.summary
    console.print("\n[bold]Summary[/]")
    console.print(f"  count:              {s['count']}")
    console.print(f"  type_accuracy:      {s['type_accuracy']:.2%}")
    console.print(f"  avg_f1:             {s['avg_f1']:.3f}")
    console.print(f"  avg_precision:      {s['avg_precision']:.3f}")
    console.print(f"  avg_recall:         {s['avg_recall']:.3f}")
    console.print(f"  avg_tokens:         {s['avg_tokens']:.1f}")
    console.print(f"  avg_rds:            {s['avg_rds']:.4f}")
    if s["path_exact_accuracy"] is not None:
        console.print(f"  path_exact_accuracy:{s['path_exact_accuracy']:.2%}")

    if s.get("by_type"):
        bt = Table("type", "count", "avg_f1", "avg_tokens")
        for tname, b in sorted(s["by_type"].items()):
            bt.add_row(tname, str(b["count"]), f"{b['avg_f1']:.3f}", f"{b['avg_tokens']:.1f}")
        console.print(bt)

    if report is not None:
        report.write_text(json.dumps(result.to_jsonable(), indent=2, default=str))
        console.print(f"[green]wrote scorecard -> {report}[/]")


@app.command()
def inspect(
    graph: Path = typer.Option(..., exists=True),
) -> None:
    """Print a short summary of a CKG CSV."""
    ckg = CKG.load(graph)
    console.print(f"Concepts: {len(ckg.concepts)}")
    console.print(f"Edges:    {sum(len(v) for v in ckg.forward.values())}")
    console.print(f"Taxonomy: {len(ckg.taxonomy_members)} groups")
    t = Table("taxonomy", "count")
    for tax, members in sorted(ckg.taxonomy_members.items(), key=lambda p: -len(p[1]))[:20]:
        t.add_row(tax, str(len(members)))
    console.print(t)


# ----------------------------------------------------------------------
# Phase 6: snapshot consolidation + Parquet shard transport
# ----------------------------------------------------------------------


@app.command()
def consolidate(
    kind: str = typer.Option("topic", help="topic | time_window | user_attention"),
    canonical_name: str = typer.Option(None, help="topic seed (when kind=topic)"),
    entity_type: str = typer.Option("topic", help="seed entity type (when kind=topic)"),
    hops: int = typer.Option(2, help="BFS depth (when kind=topic)"),
    user_id: str = typer.Option(None, help="user (when kind=user_attention)"),
    days: int = typer.Option(30, help="lookback window (when kind=user_attention)"),
    from_: str = typer.Option(None, "--from", help="ISO-8601 start (when kind=time_window)"),
    to: str = typer.Option(None, "--to", help="ISO-8601 end (when kind=time_window)"),
    label: str = typer.Option(None, help="optional human-readable label"),
    dsn: str = typer.Option(None, "--dsn", help="Postgres DSN (defaults to env vars)"),
) -> None:
    """Run a consolidation cycle and store the snapshot."""
    import os
    from datetime import datetime, timedelta, timezone
    from trellion.db.engine import init_engine
    from trellion.snapshots import SliceSpec, SnapshotOrchestrator

    dsn = dsn or os.environ.get("DATABASE_URL") or os.environ.get("PAGEINDEX_DSN")
    if not dsn:
        console.print("[red]No DSN — pass --dsn or set DATABASE_URL[/]")
        raise typer.Exit(1)

    if kind == "topic" and canonical_name:
        spec = SliceSpec(
            kind="topic",
            params={"entity_type": entity_type, "canonical_name": canonical_name, "hops": hops},
            label=label,
        )
    elif kind == "user_attention" and user_id:
        spec = SliceSpec(
            kind="user_attention",
            params={"user_id": user_id, "days": days},
            label=label,
        )
    elif kind == "time_window":
        # Default window: last 365 days if no explicit bounds given
        end = datetime.fromisoformat(to.replace("Z", "+00:00")) if to else datetime.now(timezone.utc)
        start = (
            datetime.fromisoformat(from_.replace("Z", "+00:00")) if from_
            else end - timedelta(days=365)
        )
        spec = SliceSpec(
            kind="time_window",
            params={"from": start.isoformat(), "to": end.isoformat()},
            label=label,
        )
    else:
        console.print(
            "[red]For kind=topic pass --canonical-name; "
            "for kind=user_attention pass --user-id; "
            "for kind=time_window optionally pass --from/--to (defaults to last 365d)[/]"
        )
        raise typer.Exit(1)

    _, async_engine = init_engine(dsn)
    orchestrator = SnapshotOrchestrator(async_engine)
    result = asyncio.run(orchestrator.consolidate(spec))
    console.print(f"[green]{result.status}[/] snapshot={result.snapshot_id} stats={result.stats}")


@app.command("entity-cleanup")
def entity_cleanup(
    apply: bool = typer.Option(False, "--apply", help="stage + delete (default: preview only)"),
    max_delete: int = typer.Option(5000, help="cap on entities removed per --apply run"),
    examples: int = typer.Option(30, help="how many examples the preview prints"),
    dsn: str = typer.Option(None, "--dsn", help="Postgres DSN (defaults to env vars)"),
) -> None:
    """Stage junk entities (math symbols, bare numbers, stopwords) for removal.

    Uses the validator that now gates extraction, and graph_prune's staging
    tables, so every removal is restorable for the retention window.
    """
    from trellion.db.engine import init_engine
    from trellion.live.junk_entity_cleanup import run

    dsn = _resolve_dsn(dsn)
    if not dsn:
        console.print("[red]No DSN: pass --dsn or set DATABASE_URL[/]")
        raise typer.Exit(1)
    _, async_engine = init_engine(dsn)
    report = asyncio.run(run(async_engine, apply=apply, max_delete=max_delete))
    typer.echo(report.render(examples))
    if report.dry_run:
        typer.echo("\nPreview only. Re-run with --apply to stage and remove.")


@app.command("vidmem-backfill")
def vidmem_backfill(
    dsn: str = typer.Option(None, "--dsn", help="Postgres DSN"),
) -> None:
    """One-time frame-embedding backfill (proposal 28 Part 1, pure SQL).

    Copies embeddings already stored on trellion_traversal_records onto
    their frames so pgvector search covers pre-migration data. The old
    FAISS files are not read — they only ever duplicated these vectors —
    and can be deleted once `still_unembedded` is 0.
    """
    resolved = _resolve_dsn(dsn)
    if not resolved:
        console.print("[red]No DSN — pass --dsn or set DATABASE_URL[/]")
        raise typer.Exit(1)
    from trellion.db.engine import init_engine
    from trellion.live.recall import backfill_frame_embeddings

    _, engine = init_engine(resolved)
    out = asyncio.run(backfill_frame_embeddings(engine))
    console.print(
        f"[green]backfilled[/] updated={out['updated']} "
        f"still_unembedded={out['still_unembedded']}"
    )


@app.command("bench-live")
def bench_live(
    dsn: str = typer.Option(None, "--dsn", help="Postgres DSN (pageindex tables)."),
    top_k: int = typer.Option(3, help="Recall/MRR cutoff."),
    real_embeddings: bool = typer.Option(
        False,
        "--real-embeddings",
        help="Use PageIndexEmbedder (env-configured) instead of the fake "
        "token-hash embedder. The sales-claim number requires this.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the raw report JSON."),
) -> None:
    """P1.5 gate: live-path retrieval, intelligent (graph) vs vanilla (vector).

    Seeds the committed fixture corpus + hand-authored graph into the target
    database and scores both production strategies over the golden set.
    Run it against a BENCH database, not the production one.
    """
    from trellion.eval.live_bench import (
        FakeDeterministicEmbedder,
        format_report,
        run_live_bench,
    )

    resolved = _resolve_dsn(dsn)
    if not resolved:
        console.print("[red]No DSN — pass --dsn or set DATABASE_URL[/]")
        raise typer.Exit(1)

    embedder = None
    if real_embeddings:
        import os

        from trellion.live.pageindex_embedder import PageIndexEmbedder
        from trellion.live.pageindex_store import PageIndexStore

        store = PageIndexStore(dsn=resolved)
        embedder = PageIndexEmbedder(
            store=store,
            embedding_url=os.environ.get("PAGEINDEX_EMBEDDING_URL", ""),
        )
    else:
        embedder = FakeDeterministicEmbedder()

    report = asyncio.run(run_live_bench(resolved, embedder, top_k=top_k))
    if as_json:
        console.print_json(json.dumps(report))
    else:
        console.print(format_report(report))
    if report["relational_delta"] <= 0:
        console.print(
            "[yellow]graph did not beat vector on relational queries — "
            "the P2 gate is NOT met on this corpus/embedder[/]"
        )


@app.command()
def pack(
    out: Path = typer.Option(..., help="Output tarball path."),
    persona_id: str = typer.Option(None, help="Persona to scope the shard."),
    topic_entity_id: str = typer.Option(None, help="Seed entity id (for topic slice)."),
    topic_hops: int = typer.Option(2, help="BFS depth for topic slice."),
    include_faiss: bool = typer.Option(
        False,
        "--include-faiss/--no-faiss",
        help="DEPRECATED (proposal 28): frame embeddings live in Postgres "
        "now and ride the shard tables; FAISS sidecars are legacy.",
    ),
    faiss_path: str = typer.Option(None, help="FAISS base path (override env)."),
    dsn: str = typer.Option(None, "--dsn", help="Postgres DSN"),
) -> None:
    """Pack a memory slice into a Parquet shard tarball."""
    import os
    from trellion.db.engine import init_engine
    from trellion.live.video.paths import video_memory_dir
    from trellion.transport import ParquetShardPackager, ShardSliceSpec

    dsn = dsn or os.environ.get("DATABASE_URL") or os.environ.get("PAGEINDEX_DSN")
    if not dsn:
        console.print("[red]No DSN — pass --dsn or set DATABASE_URL[/]")
        raise typer.Exit(1)
    if not persona_id and not topic_entity_id:
        console.print("[red]Need --persona-id or --topic-entity-id[/]")
        raise typer.Exit(1)

    _, async_engine = init_engine(dsn)
    pkg = ParquetShardPackager(
        async_engine,
        faiss_base_path=video_memory_dir(faiss_path),
    )
    spec = ShardSliceSpec(
        persona_id=persona_id,
        topic_entity_id=topic_entity_id,
        topic_hops=topic_hops,
    )
    out = asyncio.run(pkg.pack(spec, out, include_faiss=include_faiss))
    console.print(f"[green]packed[/] -> {out}")


@app.command()
def unpack(
    tarball: Path = typer.Argument(..., exists=True, help="Shard tarball to apply."),
    faiss_path: str = typer.Option(None, help="FAISS base path (override env)."),
    skip_hash: bool = typer.Option(False, "--skip-hash", help="Skip hash verification."),
    dsn: str = typer.Option(None, "--dsn", help="Postgres DSN"),
) -> None:
    """Unpack a Parquet shard tarball into the live store."""
    import os
    from trellion.db.engine import init_engine
    from trellion.live.video.paths import video_memory_dir
    from trellion.transport import ParquetShardPackager

    dsn = dsn or os.environ.get("DATABASE_URL") or os.environ.get("PAGEINDEX_DSN")
    if not dsn:
        console.print("[red]No DSN — pass --dsn or set DATABASE_URL[/]")
        raise typer.Exit(1)

    _, async_engine = init_engine(dsn)
    pkg = ParquetShardPackager(
        async_engine,
        faiss_base_path=video_memory_dir(faiss_path),
    )
    manifest = asyncio.run(pkg.unpack(tarball, verify_hashes=not skip_hash))
    console.print(
        f"[green]unpacked[/] shard={manifest.shard_id} persona={manifest.persona_id} "
        f"tables={manifest.table_files}"
    )


if __name__ == "__main__":
    app()
