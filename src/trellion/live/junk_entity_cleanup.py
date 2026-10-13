# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Stage the junk entities that are already in the graph (dry-run by default).

``trellion.live.entity_quality`` stops new junk (``a``, ``x_t``, ``theta``,
``0.003``) from being created. This clears what is already stored, using the
SAME validator, through the SAME safety net the ``graph_prune`` heartbeat
uses: each entity is snapshotted into ``graph_prune_staging`` (entity row +
its edges + its mentions, same payload shape, so the existing
``POST /api/v1/graph/prune/restore`` brings it back) and held for the normal
30-day retention before ``graph_prune`` purges the staging row.

The payload also records the entity's cluster memberships. The restore
endpoint ignores that key; it is there so a human can see what the entity was
part of. Embeddings, graph signals and cluster membership rows go with the
entity (ON DELETE CASCADE) and regenerate on the next embed / recluster, as
they do after any prune.

Nothing here runs unless ``apply=True``. The default pass only SELECTs.

    trellion entity-cleanup                 # preview: counts + examples
    trellion entity-cleanup --apply         # stage + delete (capped)
"""

from __future__ import annotations

import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from trellion.live import entity_quality
from trellion import json_fast as json

log = structlog.get_logger(__name__)

STAGING_REASON = "junk_entity"
# Same advisory lock graph_prune takes: the two must never interleave their
# SELECT -> DELETE windows over the same tables.
_PRUNE_LOCK_SQL = "SELECT pg_try_advisory_lock(hashtext('noo:graph_prune')::bigint)"
_PRUNE_UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtext('noo:graph_prune')::bigint)"


_SCHEMA_RE = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def staging_table(schema: Optional[str]) -> str:
    """The staging table's name, schema-qualified when the caller says where it is.

    The engine this runs on pins ``search_path=trellion,public`` (every trellion
    engine does), and in the cluster layout ``graph_prune_staging`` lives in the
    gateway's ``mesh`` schema, which that path cannot see: an unqualified insert
    fails 42P01 and the whole apply stages and deletes nothing. The caller that
    knows the layout (noomesh's ``worker.rag.entity_cleanup``) passes ``mesh``.
    ``None`` keeps the standalone behaviour: resolve through the search_path.
    The name is interpolated into SQL, so it must be a plain lowercase identifier.
    """
    if schema is None:
        return "graph_prune_staging"
    if not isinstance(schema, str) or not _SCHEMA_RE.match(schema):
        raise ValueError("staging_schema must be a plain lowercase schema name")
    return f'"{schema}".graph_prune_staging'


@dataclass
class Candidate:
    entity_id: str
    name: str
    etype: str
    reason: str
    aliases_json: Any = None
    created_at: Any = None
    edges: int = 0
    mentions: int = 0
    clusters: int = 0
    # The owning account NAME (entities.user_id; NULL = shared). Carried as the
    # payload's top-level ``user_id`` -- the key the gateway's restore and an
    # account erase both read -- so a restore keeps a private entity private.
    user_id: Any = None


@dataclass
class CleanupReport:
    dry_run: bool
    scanned: int = 0
    candidates: list[Candidate] = field(default_factory=list)
    staged: int = 0
    skipped_over_cap: int = 0
    run_id: Optional[str] = None
    lock_busy: bool = False

    def by_reason(self) -> Counter:
        return Counter(c.reason for c in self.candidates)

    def by_type(self) -> Counter:
        return Counter(c.etype for c in self.candidates)

    def render(self, examples: int = 30) -> str:
        out = [
            f"{'(dry-run) ' if self.dry_run else ''}scanned {self.scanned} entities, "
            f"{len(self.candidates)} junk",
            "  with edges:    "
            f"{sum(1 for c in self.candidates if c.edges)} "
            f"({sum(c.edges for c in self.candidates)} edge rows)",
            "  with mentions: "
            f"{sum(1 for c in self.candidates if c.mentions)} "
            f"({sum(c.mentions for c in self.candidates)} mention rows)",
            "  in clusters:   "
            f"{sum(1 for c in self.candidates if c.clusters)} "
            f"({sum(c.clusters for c in self.candidates)} membership rows)",
            "  by rule: " + ", ".join(f"{k}={v}" for k, v in sorted(self.by_reason().items())),
            "  by type: " + ", ".join(f"{k}={v}" for k, v in sorted(self.by_type().items())),
        ]
        shown = sorted(self.candidates, key=lambda c: (-c.edges, c.name))[:examples]
        out.append(f"  examples (most-connected first, {len(shown)}):")
        for c in shown:
            out.append(
                f"    {c.name!r:<28} {c.etype:<13} {c.reason:<13} "
                f"edges={c.edges} mentions={c.mentions} clusters={c.clusters}"
            )
        if self.skipped_over_cap:
            out.append(f"  {self.skipped_over_cap} more over the --max-delete cap (re-run to continue)")
        if self.lock_busy:
            out.append("  graph_prune holds the lock right now; nothing was changed")
        if not self.dry_run and not self.lock_busy:
            out.append(
                f"  staged+removed {self.staged} (run {self.run_id}); restorable for the "
                "staging retention window via POST /api/v1/graph/prune/restore"
            )
        return "\n".join(out)


async def find_candidates(engine: AsyncEngine) -> tuple[int, list[Candidate]]:
    """Read-only: every entity the validator rejects, with its footprint."""
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                text("""
                    SELECT e.entity_id::text, e.canonical_name, e.type,
                           e.aliases_json, e.created_at,
                           (SELECT count(*) FROM edges d
                             WHERE d.src_entity_id = e.entity_id
                                OR d.dst_entity_id = e.entity_id) AS edges,
                           (SELECT count(*) FROM mentions m
                             WHERE m.entity_id = e.entity_id) AS mentions,
                           (SELECT count(*) FROM entity_cluster_members c
                             WHERE c.entity_id = e.entity_id) AS clusters,
                           e.user_id
                    FROM entities e
                """)
            )
        ).fetchall()
    out: list[Candidate] = []
    for r in rows:
        reason = entity_quality.check_entity(r[1], r[2])
        if reason:
            out.append(Candidate(r[0], r[1], r[2], reason, r[3], r[4], r[5], r[6], r[7], r[8]))
    return len(rows), out


async def _snapshot(conn, c: Candidate) -> dict:
    """Restore payload: SAME keys as graph_prune's, plus cluster memberships."""
    edges = (
        await conn.execute(
            text("""
                SELECT edge_id::text, src_entity_id::text, rel_type,
                       dst_entity_id::text, evidence_json, created_at,
                       confidence, classified_at, classifier_model
                FROM edges WHERE src_entity_id = CAST(:e AS uuid) OR dst_entity_id = CAST(:e AS uuid)
            """),
            {"e": c.entity_id},
        )
    ).fetchall()
    mentions = (
        await conn.execute(
            text("""
                SELECT mention_id::text, entity_id::text, page_id::text, span_json, quote
                FROM mentions WHERE entity_id = CAST(:e AS uuid)
            """),
            {"e": c.entity_id},
        )
    ).fetchall()
    members = (
        await conn.execute(
            text("""
                SELECT cluster_id::text, membership_score
                FROM entity_cluster_members WHERE entity_id = CAST(:e AS uuid)
            """),
            {"e": c.entity_id},
        )
    ).fetchall()
    iso = lambda t: t.isoformat() if t else None  # noqa: E731
    return {
        "entity": {
            "entity_id": c.entity_id,
            "canonical_name": c.name,
            "type": c.etype,
            "aliases_json": c.aliases_json,
            "created_at": iso(c.created_at),
        },
        "edges": [
            {
                "edge_id": r[0], "src_entity_id": r[1], "rel_type": r[2],
                "dst_entity_id": r[3], "evidence_json": r[4], "created_at": iso(r[5]),
                "confidence": r[6], "classified_at": iso(r[7]), "classifier_model": r[8],
            }
            for r in edges
        ],
        "mentions": [
            {
                "mention_id": r[0], "entity_id": r[1], "page_id": r[2],
                "span_json": r[3], "quote": r[4],
            }
            for r in mentions
        ],
        # ALWAYS present (null for a shared entity), same as the gateway prune's
        # payload: the restore refuses a row without it rather than guess.
        "user_id": c.user_id,
        "cluster_members": [
            {"cluster_id": r[0], "membership_score": r[1]} for r in members
        ],
        "quality_rule": c.reason,
    }


async def run(
    engine: AsyncEngine,
    *,
    apply: bool = False,
    max_delete: int = 5000,
    batch: int = 200,
    restrict_to: Optional[set[str]] = None,
    staging_schema: Optional[str] = None,
) -> CleanupReport:
    """Preview (default) or stage-then-delete every junk entity.

    ``max_delete`` bounds one apply run, most-connected junk LAST so a cap
    never leaves the cheap orphans behind while touching the edge-bearing
    rows first; re-run to continue.
    """
    staging = staging_table(staging_schema)  # refuses a bad name before any work
    scanned, cands = await find_candidates(engine)
    if restrict_to is not None:  # tests: never touch rows they did not create
        cands = [c for c in cands if c.entity_id in restrict_to]
    report = CleanupReport(dry_run=not apply, scanned=scanned, candidates=cands)
    if not apply:
        return report

    cands.sort(key=lambda c: (c.edges, c.name))
    todo = cands[:max_delete]
    report.skipped_over_cap = len(cands) - len(todo)
    report.run_id = "entq-" + str(uuid.uuid4())[:8]

    async with engine.connect() as conn:
        got = (await conn.execute(text(_PRUNE_LOCK_SQL))).scalar()
        await conn.commit()
        if not got:
            report.lock_busy = True
            return report
        try:
            for i in range(0, len(todo), batch):
                for c in todo[i:i + batch]:
                    payload = await _snapshot(conn, c)
                    await conn.execute(
                        text(f"""
                            INSERT INTO {staging}
                                (entity_id, canonical_name, entity_type, reason, payload, run_id)
                            VALUES (CAST(:e AS uuid), :n, :t, :r, CAST(:p AS jsonb), :run)
                        """),
                        {"e": c.entity_id, "n": c.name, "t": c.etype, "r": STAGING_REASON,
                         "p": json.dumps(payload, default=str), "run": report.run_id},
                    )
                    # Edges, mentions, cluster members, embeddings, signals
                    # cascade (all FK ON DELETE CASCADE): no dangling endpoint.
                    await conn.execute(
                        text("DELETE FROM entities WHERE entity_id = CAST(:e AS uuid)"),
                        {"e": c.entity_id},
                    )
                    report.staged += 1
                # staging row + delete commit together per batch
                await conn.commit()
        except Exception:
            await conn.rollback()
            raise
        finally:
            await conn.execute(text(_PRUNE_UNLOCK_SQL))
            await conn.commit()
    log.info(
        "junk_entity_cleanup.applied", run_id=report.run_id, staged=report.staged,
        by_reason=dict(report.by_reason()), skipped_over_cap=report.skipped_over_cap,
    )
    return report
