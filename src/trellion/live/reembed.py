# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Provenance upkeep and the re-embed worker (noomesh i150).

Runs inside the PageIndex embedder's loop (:class:`PageIndexEmbedder`), which
every host already starts, so there is no second thing to deploy. One
:meth:`ReembedWorker.run_pass` does, in order:

1. **Legacy claim.** Every vector written before migration 106 has a NULL
   ``embedding_model``; all of them came from the default space, so they are
   labelled with the space's current model (``vector_dims`` for the width).
   Batched, idempotent, restart-safe. Skipped while a cutover is under way
   (the current model is about to change under it).
2. **No migration** (cancelled or finished): clear leftover ``embedding_next``
   staging columns, in batches.
3. **Migration ``filling``**: for each table, one claimed batch of rows whose
   ``embedding_next_model`` is not the migration's ``next_model``. Source text
   is re-embedded with ``"<space>@next"`` (or the vector is derived, see
   :data:`TABLES`), and written to ``embedding_next`` /
   ``embedding_next_model``. Rows written while the migration runs are picked
   up by the same scan. Progress is PUT to the gateway.
4. **Migration ``cutover``**: per table, one transaction swaps the staged
   column into the live one; then the progress PUT carries
   ``cutover_done: true`` and the gateway flips the space's model.

The live column keeps serving throughout: nothing here touches ``embedding``
until the swap, and every ordinary vector write clears the row's staged
vector (it was computed from content that just changed).

Re-embedding, table by table (``kind``):

``text``     source text is stored; re-embedded with ``<space>@next``.
``entity``   the neighbourhood description trellion-intel embeds
             (``name (type): REL -> neighbour (type), ...``), rebuilt from the
             stored ``entities`` and ``edges``.
``cluster``  derived: leaf centroid = mean of its members' staged vectors;
             parent centroid = member-weighted mean of its children's.
``episode``  derived: copied from the traversal record it was archived with.
``none``     no stored source and no derivation (``trellion_concepts``): it is
             counted (``total`` = its rows, ``done`` stays 0) and never
             invented; the gateway's completion check will not pass while such
             rows exist.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from trellion import embedding_space
from trellion.embedding_space import Embeddings, SpaceRecord

log = structlog.get_logger(__name__)

#: Width of every vector column and staging column (migration 100 / 106).
COLUMN_DIMS = 1024

#: Same clip the ordinary entry worker applies before embedding.
ENTRY_CLIP_CHARS = 1500

LEGACY_BATCH = 2000
LEGACY_MAX_BATCHES_PER_PASS = 50
#: A sweep over a big table is a seq scan: do it at start, on a state
#: transition, and otherwise this often.
SWEEP_EVERY_S = 300.0
#: A table whose scan came back empty is not scanned again for this long.
TABLE_IDLE_S = 60.0
PROGRESS_EVERY_S = 30.0

EmbedNext = Callable[[list[str]], Awaitable[Embeddings]]


@dataclass(frozen=True)
class TableSpec:
    name: str
    pk: str
    vec: str  # the live vector column
    kind: str  # text | entity | cluster | episode | none
    #: ``kind == "text"``: FROM/JOIN for the claim, aliased ``t`` (the vector
    #: row) and exposing a ``src`` text column.
    source_from: str = ""
    src: str = ""


TABLES: tuple[TableSpec, ...] = (
    TableSpec(
        "page_embeddings", "page_id", "embedding", "text",
        "page_embeddings t JOIN pages s ON s.page_id = t.page_id", "s.text",
    ),
    TableSpec(
        "chunk_embeddings", "chunk_id", "embedding", "text",
        "chunk_embeddings t JOIN page_chunks s ON s.chunk_id = t.chunk_id", "s.text",
    ),
    TableSpec("noo_entries", "id", "embedding", "text", "noo_entries t", "t.content"),
    TableSpec(
        "trellion_traversal_records", "id", "query_embedding", "text",
        "trellion_traversal_records t", "t.query_text",
    ),
    TableSpec("entity_embeddings", "entity_id", "embedding", "entity"),
    TableSpec("trellion_episodes", "id", "embedding", "episode"),
    TableSpec("entity_clusters", "cluster_id", "centroid", "cluster"),
    TableSpec("trellion_concepts", "id", "centroid", "none"),
)

TABLE_NAMES = tuple(t.name for t in TABLES)


def _lit(values) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


def _parse_vec(s: Optional[str]) -> list[float]:
    if not s:
        return []
    return [float(x) for x in s.strip("[]").split(",") if x != ""]


def entity_description(name: str, etype: str, neighbours: list[dict]) -> str:
    """The text trellion-intel embeds for an entity (graph_embedder.py).

    ``neighbours`` are ``{name, type, rel_type, direction}`` in descending
    confidence order. Kept byte-identical to the intel builder so a re-embed
    reproduces the original recipe.
    """
    header = f"{name} ({etype})"
    if not neighbours:
        return header
    parts = []
    for nb in neighbours:
        arrow = "->" if nb["direction"] == "outgoing" else "<-"
        parts.append(f"{nb['rel_type']} {arrow} {nb['name']} ({nb['type']})")
    return f"{header}: {', '.join(parts)}"


class ReembedWorker:
    """Legacy labelling, staging-column cleanup, re-embed and cutover."""

    def __init__(
        self,
        engine: AsyncEngine,
        embed_next: EmbedNext,
        *,
        batch_size: int = 32,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._embed_next = embed_next
        self._batch = batch_size
        self._clock = clock
        self._cursor: dict[str, Optional[str]] = {}
        self._idle_until: dict[str, float] = {}
        self._last_sweep = float("-inf")
        self._last_report = float("-inf")
        self._saw_migration = False
        self._cutover_pending = False

    # ------------------------------------------------------------------
    # Pass
    # ------------------------------------------------------------------

    async def in_cutover(self) -> bool:
        """True while the space is mid-swap: ordinary writers should wait."""
        rec = await embedding_space.aget_space()
        return bool(rec and rec.migration_state == "cutover")

    async def run_pass(self) -> int:
        """One pass. Returns the number of rows touched (0 = idle)."""
        rec = await embedding_space.aget_space()
        if rec is None:
            return 0
        state = rec.migration_state
        worked = 0
        now = self._clock()

        if state != "cutover" and now - self._last_sweep >= SWEEP_EVERY_S:
            worked += await self.claim_legacy(rec.model)

        if state == "filling" and rec.next_model:
            self._saw_migration = True
            worked += await self._fill(rec)
        elif state == "cutover" and rec.next_model:
            self._saw_migration = True
            worked += await self._cutover(rec)
        else:
            if self._saw_migration or now - self._last_sweep >= SWEEP_EVERY_S:
                worked += await self.clear_staging()
                self._saw_migration = False
            self._cursor.clear()
            self._idle_until.clear()
            self._cutover_pending = False

        if state != "cutover" and now - self._last_sweep >= SWEEP_EVERY_S:
            self._last_sweep = now
        return worked

    # ------------------------------------------------------------------
    # Legacy claim
    # ------------------------------------------------------------------

    async def claim_legacy(self, model: str) -> int:
        """Label every NULL-model vector with the space's current model."""
        total = 0
        for spec in TABLES:
            for _ in range(LEGACY_MAX_BATCHES_PER_PASS):
                async with self._engine.begin() as conn:
                    res = await conn.execute(
                        text(self.legacy_claim_sql(spec)),
                        {"m": model, "n": LEGACY_BATCH},
                    )
                n = res.rowcount or 0
                total += n
                if n < LEGACY_BATCH:
                    break
        if total:
            log.info("reembed.legacy_claimed", rows=total, model=model)
        return total

    @staticmethod
    def legacy_claim_sql(spec: TableSpec) -> str:
        return (
            f"UPDATE {spec.name} SET embedding_model = :m, "
            f"embedding_dim = vector_dims({spec.vec}) "
            f"WHERE ctid = ANY(ARRAY(SELECT ctid FROM {spec.name} "
            f"WHERE embedding_model IS NULL AND {spec.vec} IS NOT NULL "
            f"LIMIT :n FOR NO KEY UPDATE SKIP LOCKED))"
        )

    # ------------------------------------------------------------------
    # Staging cleanup
    # ------------------------------------------------------------------

    async def clear_staging(self) -> int:
        total = 0
        for spec in TABLES:
            for _ in range(LEGACY_MAX_BATCHES_PER_PASS):
                async with self._engine.begin() as conn:
                    res = await conn.execute(
                        text(self.clear_staging_sql(spec)), {"n": LEGACY_BATCH}
                    )
                n = res.rowcount or 0
                total += n
                if n < LEGACY_BATCH:
                    break
        if total:
            log.info("reembed.staging_cleared", rows=total)
        return total

    @staticmethod
    def clear_staging_sql(spec: TableSpec) -> str:
        return (
            f"UPDATE {spec.name} SET embedding_next = NULL, embedding_next_model = NULL "
            f"WHERE ctid = ANY(ARRAY(SELECT ctid FROM {spec.name} "
            f"WHERE embedding_next_model IS NOT NULL OR embedding_next IS NOT NULL "
            f"LIMIT :n FOR NO KEY UPDATE SKIP LOCKED))"
        )

    # ------------------------------------------------------------------
    # Filling
    # ------------------------------------------------------------------

    async def _fill(self, rec: SpaceRecord) -> int:
        nxt = rec.next_model or ""
        worked = 0
        now = self._clock()
        for spec in TABLES:
            if spec.kind == "none" or self._idle_until.get(spec.name, 0.0) > now:
                continue
            n = await self._fill_table(spec, rec)
            worked += n
            if n == 0:
                # A scan from the start found nothing: sleep on this table.
                if self._cursor.get(spec.name) is None:
                    self._idle_until[spec.name] = self._clock() + TABLE_IDLE_S
                self._cursor[spec.name] = None
        if self._clock() - self._last_report >= PROGRESS_EVERY_S:
            await self.report(nxt, cutover_done=False)
        return worked

    async def _fill_table(self, spec: TableSpec, rec: SpaceRecord) -> int:
        nxt = rec.next_model or ""
        try:
            if spec.kind == "text":
                return await self._fill_text(spec, rec)
            if spec.kind == "entity":
                return await self._fill_entities(spec, rec)
            if spec.kind == "cluster":
                return await self._fill_clusters(spec, rec)
            if spec.kind == "episode":
                return await self._copy_episodes(spec, nxt)
        except Exception as exc:  # noqa: BLE001 - one table must not stall the rest
            reason = getattr(exc, "reason", None)
            log.warning(
                "reembed.table_failed", table=spec.name, error=str(exc)[:200], reason=reason
            )
        return 0

    def _check_vectors(self, vecs: Embeddings, rec: SpaceRecord, n: int) -> str:
        """Refuse a batch that is the wrong size or width; return the model."""
        want = rec.next_dims or COLUMN_DIMS
        if len(vecs) != n:
            raise ValueError(f"embedder returned {len(vecs)} vectors for {n} inputs")
        for v in vecs:
            if len(v) != want:
                raise ValueError(
                    f"refusing a {len(v)}-dim vector for a vector({want}) column"
                )
        model = vecs.model or rec.next_model or ""
        if rec.next_model and model != rec.next_model:
            log.warning(
                "reembed.model_mismatch", served=model, expected=rec.next_model
            )
        return model

    def claim_sql(self, spec: TableSpec, *, cursor: bool) -> str:
        """The claim for a ``text`` table. ``:next`` and ``:limit`` bind."""
        after = f" AND t.{spec.pk} > CAST(:after AS uuid)" if cursor else ""
        return (
            f"SELECT t.{spec.pk} AS pk, {spec.src} AS src FROM {spec.source_from} "
            f"WHERE t.{spec.vec} IS NOT NULL "
            f"AND t.embedding_next_model IS DISTINCT FROM :next{after} "
            f"ORDER BY t.{spec.pk} LIMIT :limit FOR NO KEY UPDATE OF t SKIP LOCKED"
        )

    @staticmethod
    def store_next_sql(spec: TableSpec) -> str:
        return (
            f"UPDATE {spec.name} SET embedding_next = CAST(:vec AS vector), "
            f"embedding_next_model = :model WHERE {spec.pk} = CAST(:pk AS uuid)"
        )

    async def _fill_text(self, spec: TableSpec, rec: SpaceRecord) -> int:
        nxt = rec.next_model or ""
        after = self._cursor.get(spec.name)
        params: dict[str, Any] = {"next": nxt, "limit": self._batch}
        if after:
            params["after"] = after
        async with self._engine.connect() as conn:
            async with conn.begin():
                rows = (
                    await conn.execute(
                        text(self.claim_sql(spec, cursor=bool(after))), params
                    )
                ).fetchall()
                if not rows:
                    return 0
                texts = [
                    (r[1][:ENTRY_CLIP_CHARS] if spec.name == "noo_entries" else r[1]) or ""
                    for r in rows
                ]
                vecs = await self._embed_next(texts)
                model = self._check_vectors(vecs, rec, len(rows))
                await conn.execute(
                    text(self.store_next_sql(spec)),
                    [
                        {"vec": _lit(v), "model": model, "pk": str(r[0])}
                        for r, v in zip(rows, vecs)
                    ],
                )
        self._cursor[spec.name] = str(rows[-1][0])
        return len(rows)

    # -- entity embeddings -------------------------------------------------

    _ENTITY_CLAIM = (
        "SELECT t.entity_id FROM entity_embeddings t "
        "WHERE t.embedding IS NOT NULL AND t.embedding_next_model IS DISTINCT FROM :next"
        "{after} ORDER BY t.entity_id LIMIT :limit FOR NO KEY UPDATE OF t SKIP LOCKED"
    )

    _ENTITY_NEIGHBOURS = (
        "SELECT ent.entity_id, ent.canonical_name, ent.type, nb.name, nb.type, "
        "       nb.rel_type, nb.direction "
        "FROM entities ent "
        "LEFT JOIN LATERAL ("
        "  SELECT n.canonical_name AS name, n.type AS type, e.rel_type AS rel_type, "
        "         CASE WHEN e.src_entity_id = ent.entity_id THEN 'outgoing' "
        "              ELSE 'incoming' END AS direction, "
        "         ROW_NUMBER() OVER (ORDER BY COALESCE(e.confidence, 1.0) DESC) AS rn "
        "  FROM edges e "
        "  JOIN entities n ON n.entity_id = CASE WHEN e.src_entity_id = ent.entity_id "
        "                                        THEN e.dst_entity_id ELSE e.src_entity_id END "
        "  WHERE e.src_entity_id = ent.entity_id OR e.dst_entity_id = ent.entity_id "
        "  ORDER BY COALESCE(e.confidence, 1.0) DESC LIMIT 20"
        ") nb ON TRUE "
        "WHERE ent.entity_id = ANY(CAST(:ids AS uuid[])) "
        "ORDER BY ent.entity_id, nb.rn"
    )

    async def _fill_entities(self, spec: TableSpec, rec: SpaceRecord) -> int:
        nxt = rec.next_model or ""
        after = self._cursor.get(spec.name)
        params: dict[str, Any] = {"next": nxt, "limit": self._batch}
        if after:
            params["after"] = after
        sql = self._ENTITY_CLAIM.format(
            after=" AND t.entity_id > CAST(:after AS uuid)" if after else ""
        )
        async with self._engine.connect() as conn:
            async with conn.begin():
                ids = [str(r[0]) for r in (await conn.execute(text(sql), params)).fetchall()]
                if not ids:
                    return 0
                rows = (
                    await conn.execute(text(self._ENTITY_NEIGHBOURS), {"ids": ids})
                ).fetchall()
                by_id: dict[str, dict] = {}
                for eid, name, etype, nname, ntype, rel, direction in rows:
                    ent = by_id.setdefault(
                        str(eid), {"name": name, "type": etype, "nb": []}
                    )
                    if nname is not None:
                        ent["nb"].append(
                            {"name": nname, "type": ntype, "rel_type": rel, "direction": direction}
                        )
                present = [i for i in ids if i in by_id]
                if present:
                    texts = [
                        entity_description(by_id[i]["name"], by_id[i]["type"], by_id[i]["nb"])
                        for i in present
                    ]
                    vecs = await self._embed_next(texts)
                    model = self._check_vectors(vecs, rec, len(present))
                    await conn.execute(
                        text(self.store_next_sql(spec)),
                        [
                            {"vec": _lit(v), "model": model, "pk": i}
                            for i, v in zip(present, vecs)
                        ],
                    )
        self._cursor[spec.name] = ids[-1]
        return len(ids)

    # -- cluster centroids -------------------------------------------------

    #: Clusters whose inputs are all staged. A leaf (has member rows) needs
    #: every member's vector re-embedded; a parent (no member rows) needs every
    #: child centroid done. Selecting only READY clusters means nothing blocks
    #: the scan and no cursor is needed.
    CLUSTER_READY_SQL = (
        "SELECT c.cluster_id, c.level FROM entity_clusters c "
        "WHERE c.centroid IS NOT NULL AND c.embedding_next_model IS DISTINCT FROM :next AND ("
        "  (EXISTS (SELECT 1 FROM entity_cluster_members m WHERE m.cluster_id = c.cluster_id)"
        "   AND EXISTS (SELECT 1 FROM entity_cluster_members m "
        "               JOIN entity_embeddings ee ON ee.entity_id = m.entity_id "
        "               WHERE m.cluster_id = c.cluster_id AND ee.embedding_next_model = :next)"
        "   AND NOT EXISTS (SELECT 1 FROM entity_cluster_members m "
        "               JOIN entity_embeddings ee ON ee.entity_id = m.entity_id "
        "               WHERE m.cluster_id = c.cluster_id "
        "                 AND ee.embedding_next_model IS DISTINCT FROM :next))"
        "  OR (NOT EXISTS (SELECT 1 FROM entity_cluster_members m WHERE m.cluster_id = c.cluster_id)"
        "   AND EXISTS (SELECT 1 FROM entity_clusters ch WHERE ch.parent_cluster_id = c.cluster_id "
        "               AND ch.centroid IS NOT NULL)"
        "   AND NOT EXISTS (SELECT 1 FROM entity_clusters ch "
        "               WHERE ch.parent_cluster_id = c.cluster_id AND ch.centroid IS NOT NULL "
        "                 AND ch.embedding_next_model IS DISTINCT FROM :next))"
        ") ORDER BY c.level, c.cluster_id LIMIT :limit FOR NO KEY UPDATE OF c SKIP LOCKED"
    )

    async def _fill_clusters(self, spec: TableSpec, rec: SpaceRecord) -> int:
        import numpy as np

        nxt = rec.next_model or ""
        out: list[dict] = []
        async with self._engine.connect() as conn:
            async with conn.begin():
                ready = (
                    await conn.execute(
                        text(self.CLUSTER_READY_SQL), {"next": nxt, "limit": self._batch}
                    )
                ).fetchall()
                for cid, _level in ready:
                    members = (
                        await conn.execute(
                            text(
                                "SELECT ee.embedding_next::text FROM entity_cluster_members m "
                                "JOIN entity_embeddings ee ON ee.entity_id = m.entity_id "
                                "WHERE m.cluster_id = :c AND ee.embedding_next_model = :next"
                            ),
                            {"c": cid, "next": nxt},
                        )
                    ).fetchall()
                    if members:
                        # Leaf: the mean of its members' vectors.
                        arr = np.array([_parse_vec(r[0]) for r in members], dtype=np.float64)
                        centroid = arr.mean(axis=0)
                    else:
                        # Parent: the member-weighted mean of its children's.
                        kids = (
                            await conn.execute(
                                text(
                                    "SELECT embedding_next::text, member_count "
                                    "FROM entity_clusters WHERE parent_cluster_id = :c "
                                    "AND centroid IS NOT NULL"
                                ),
                                {"c": cid},
                            )
                        ).fetchall()
                        arr = np.array([_parse_vec(k[0]) for k in kids], dtype=np.float64)
                        w = np.array([max(int(k[1] or 0), 1) for k in kids], dtype=np.float64)
                        centroid = (arr * w[:, None]).sum(axis=0) / w.sum()
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroid = centroid / norm
                    out.append(
                        {"vec": _lit(centroid.tolist()), "model": nxt, "pk": str(cid)}
                    )
                if out:
                    await conn.execute(text(self.store_next_sql(spec)), out)
        return len(out)

    # -- episodes ----------------------------------------------------------

    @staticmethod
    def episode_copy_sql() -> str:
        return (
            "WITH pick AS ("
            " SELECT e.id, t.embedding_next, t.embedding_next_model "
            " FROM trellion_episodes e "
            " JOIN trellion_traversal_records t ON t.video_frame_id = e.id "
            " WHERE e.embedding IS NOT NULL AND t.embedding_next_model = :next "
            "   AND e.embedding_next_model IS DISTINCT FROM :next "
            " LIMIT :limit FOR NO KEY UPDATE OF e SKIP LOCKED) "
            "UPDATE trellion_episodes e SET embedding_next = p.embedding_next, "
            "embedding_next_model = p.embedding_next_model "
            "FROM pick p WHERE e.id = p.id"
        )

    async def _copy_episodes(self, spec: TableSpec, nxt: str) -> int:
        async with self._engine.begin() as conn:
            res = await conn.execute(
                text(self.episode_copy_sql()), {"next": nxt, "limit": self._batch}
            )
        return res.rowcount or 0

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    @staticmethod
    def count_sql(spec: TableSpec, *, swapped: bool = False) -> str:
        done = "embedding_model" if swapped else "embedding_next_model"
        return (
            f"SELECT count(*), count(*) FILTER (WHERE {done} = :next) "
            f"FROM {spec.name} WHERE {spec.vec} IS NOT NULL"
        )

    async def counts(self, nxt: str, *, swapped: bool = False) -> dict[str, tuple[int, int]]:
        """``table -> (done, total)`` for the progress report."""
        out: dict[str, tuple[int, int]] = {}
        async with self._engine.connect() as conn:
            for spec in TABLES:
                row = (
                    await conn.execute(
                        text(self.count_sql(spec, swapped=swapped)), {"next": nxt}
                    )
                ).one()
                out[spec.name] = (int(row[1] or 0), int(row[0] or 0))
        return out

    async def report(self, nxt: str, *, cutover_done: bool) -> bool:
        try:
            tables = await self.counts(nxt, swapped=cutover_done)
        except Exception as exc:  # noqa: BLE001
            log.warning("reembed.counts_failed", error=str(exc)[:200])
            return False
        self._last_report = self._clock()
        view = await embedding_space.aput_progress(nxt, tables, cutover_done)
        if cutover_done and view is not None:
            # The gateway flipped the space (or refused): re-read it now.
            await embedding_space.aget_space(force=True)
        return view is not None

    # ------------------------------------------------------------------
    # Cutover
    # ------------------------------------------------------------------

    @staticmethod
    def cutover_sql(spec: TableSpec) -> str:
        return (
            f"UPDATE {spec.name} SET {spec.vec} = embedding_next, "
            f"embedding_model = embedding_next_model, "
            f"embedding_dim = COALESCE(CAST(:dim AS integer), vector_dims(embedding_next)), "
            f"embedding_next = NULL, embedding_next_model = NULL "
            f"WHERE embedding_next_model = :next"
        )

    async def _cutover(self, rec: SpaceRecord) -> int:
        nxt = rec.next_model or ""
        total = 0
        for spec in TABLES:
            async with self._engine.begin() as conn:
                res = await conn.execute(
                    text(self.cutover_sql(spec)), {"next": nxt, "dim": rec.next_dims}
                )
            n = res.rowcount or 0
            total += n
            if n:
                log.info("reembed.cutover_table", table=spec.name, rows=n)
        await self.report(nxt, cutover_done=True)
        return total
