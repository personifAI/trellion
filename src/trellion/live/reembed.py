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

No transaction is held while a batch embeds (with ``NOO_EMBED_BULK`` that wait
is minutes). Every embedding loop here is CLAIM (short transaction: pick rows,
read each one's id and an md5 of its source, commit), EMBED (nothing open),
WRITE BACK (short transaction: update a row only if its source still hashes to
what was read and it still needs this work). A row that changed meanwhile is
skipped and picked up again by the normal selection; a row two workers picked
is written once, the second write being a no-op. See :meth:`ReembedWorker._fill_text`.

The live column keeps serving throughout: nothing here touches ``embedding``
until the swap, and every ordinary vector write clears the row's staged
vector (it was computed from content that just changed).

Re-embedding, table by table (``kind``):

``text``     source text is stored; re-embedded with ``<space>@next``.
``entity``   the neighbourhood description the entity-vector writer embeds
             (``name (type): REL -> neighbour (type), ...``,
             :func:`trellion_protocols.embedding_text.entity_description`),
             rebuilt from the stored ``entities`` and ``edges``.
``cluster``  derived: leaf centroid = mean of its members' staged vectors;
             parent centroid = member-weighted mean of its children's.
``episode``  derived: copied from the traversal record it was archived with.
``none``     no stored source and no derivation (``trellion_concepts``).

UNSOURCED rows (concept centroids, episodes with no traversal record, and rows
whose text the embedder rejects even when sent alone) are never invented. They
are left out of ``done``/``total`` and reported per table as ``unsourced``.
At cutover a row with no source has its vector and model set to NULL (an
old-model vector must not sit under the new model), so it drops out of search
until whatever owns it regenerates it on its next run. Nothing in trellion,
trellion-intel or trellion-evolve writes concept centroids. A SKIPPED text row
(or an episode skipped with it) still has a source: it keeps its old vector
under its old model label, out of search, and the stale sweep retries it with
the new model after the migration (see :meth:`ReembedWorker.skipped_null_sql`).

A skipped row also skips what is DERIVED from it, or the migration could never
complete (the gateway refuses ``complete`` until every table's done==total):
an episode whose every traversal record is skipped (or has no vector), and a
cluster with no member and no child left to derive from. A cluster that keeps
some inputs is derived from those alone. See :meth:`ReembedWorker.skip_dependents`.

STALE sweep (no migration running): rows whose ``embedding_model`` is set but
is not the space's current model, and that have a source, are re-embedded into
the LIVE column with the plain space model. That heals a cutover race and
anything else written under an old model. Same cadence as the legacy claim.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# The entity text recipe lives in trellion-protocols: the entity-vector writer
# and this re-embed call one function, so their text cannot drift.
from trellion_protocols.embedding_text import (
    ENTITY_DESCRIPTION_MAX_NEIGHBOURS,
    ENTITY_NEIGHBOUR_ORDER_SQL,
    entity_description,
)

from trellion import embedding_space
from trellion.embedding_space import Embeddings, SpaceRecord
from trellion.live.embed_throttle import EmbedUnavailable

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
    #: SQL over alias ``t``: true for rows with an embedding but no source and
    #: no derivation to re-embed them from.
    unsourced: str = "FALSE"


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
    TableSpec(
        "trellion_episodes", "id", "embedding", "episode",
        unsourced="NOT EXISTS (SELECT 1 FROM trellion_traversal_records r "
                  "WHERE r.video_frame_id = t.id)",
    ),
    TableSpec("entity_clusters", "cluster_id", "centroid", "cluster"),
    TableSpec("trellion_concepts", "id", "centroid", "none", unsourced="TRUE"),
)

TABLE_NAMES = tuple(t.name for t in TABLES)


def _lit(values) -> str:
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


def _parse_vec(s: Optional[str]) -> list[float]:
    if not s:
        return []
    return [float(x) for x in s.strip("[]").split(",") if x != ""]


class ReembedWorker:
    """Legacy labelling, staging-column cleanup, re-embed and cutover."""

    def __init__(
        self,
        engine: AsyncEngine,
        embed_next: EmbedNext,
        *,
        embed_live: Optional[EmbedNext] = None,
        batch_size: int = 32,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._engine = engine
        self._embed_next = embed_next
        self._embed_live = embed_live
        #: (table, pk, model-key) of rows the embedder rejects even alone.
        #: In memory: reset on restart, which is fine.
        self._skipped: set[tuple[str, str, str]] = set()
        self._alone_failures: dict[tuple[str, str, str], int] = {}
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
                if self._saw_migration:
                    self._skipped = {k for k in self._skipped if k[2].startswith("live:")}
                    self._alone_failures.clear()
                self._saw_migration = False
            if now - self._last_sweep >= SWEEP_EVERY_S:
                worked += await self.stale_sweep(rec)
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
            await self.skip_dependents(nxt)
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

    def _check_vectors(self, vecs: list, model: Optional[str], rec: SpaceRecord, n: int,
                       *, live: bool = False) -> str:
        """Refuse a batch that is the wrong size or width; return the model."""
        want = (rec.dims if live else rec.next_dims) or COLUMN_DIMS
        if len(vecs) != n:
            raise ValueError(f"embedder returned {len(vecs)} vectors for {n} inputs")
        for v in vecs:
            if len(v) != want:
                raise ValueError(
                    f"refusing a {len(v)}-dim vector for a vector({want}) column"
                )
        expected = rec.model if live else rec.next_model
        model = model or expected or ""
        if expected and model != expected:
            log.warning("reembed.model_mismatch", served=model, expected=expected)
        return model

    async def _embed_rows(
        self, spec: TableSpec, pks: list[str], texts: list[str], embed: EmbedNext, key: str
    ) -> tuple[list, Optional[str]]:
        """Embed a batch; if it is rejected, retry the rows one at a time.

        Returns ``(vectors, model)`` aligned to ``pks`` with ``None`` for a row
        that fails alone (it is added to the skip set and logged). An outage
        (every row fails, with the throttle reporting overload) re-raises so
        nothing is skipped because the gateway was down.
        """
        try:
            out = await embed(texts)
            return list(out), out.model
        except Exception as batch_exc:  # noqa: BLE001
            first = batch_exc
        vecs: list = []
        model: Optional[str] = None
        ok = 0
        failed: list[tuple[int, Exception]] = []
        for i, (pk, t) in enumerate(zip(pks, texts)):
            try:
                v = await embed([t])
                vecs.append(v[0])
                model = model or v.model
                ok += 1
            except Exception as exc:  # noqa: BLE001
                vecs.append(None)
                failed.append((i, exc))
        for i, exc in failed:
            k = (spec.name, pks[i], key)
            unavailable = isinstance(exc, EmbedUnavailable)
            if unavailable and ok == 0:
                n = self._alone_failures.get(k, 0) + 1
                self._alone_failures[k] = n
                if n < 3:
                    raise first
            self._skipped.add(k)
            log.warning(
                "reembed.row_skipped", table=spec.name, pk=pks[i], key=key,
                error=str(exc)[:200],
            )
        return vecs, model

    def claim_sql(self, spec: TableSpec, *, cursor: bool) -> str:
        """The claim for a ``text`` table. ``:next`` and ``:limit`` bind."""
        after = f" AND t.{spec.pk} > CAST(:after AS uuid)" if cursor else ""
        return (
            f"SELECT t.{spec.pk} AS pk, {spec.src} AS src, md5({spec.src}) AS seen "
            f"FROM {spec.source_from} "
            f"WHERE t.{spec.vec} IS NOT NULL "
            f"AND t.embedding_next_model IS DISTINCT FROM :next{after} "
            f"ORDER BY t.{spec.pk} LIMIT :limit FOR NO KEY UPDATE OF t SKIP LOCKED"
        )

    @staticmethod
    def store_next_sql(spec: TableSpec) -> str:
        """Write a staged vector. ``text`` and ``entity`` rows are guarded.

        The guard is the write-back recheck: the row must still have a vector
        and not yet be staged for ``:next`` (so a second worker's write is a
        no-op), and a ``text`` row's source must still hash to ``:seen``. An
        ``entity`` row's source is rechecked by the caller (it compares the
        description it embedded to the current one). Cluster centroids are
        derived in the same transaction as their write; unguarded.
        """
        if spec.kind == "text":
            return (
                f"UPDATE {spec.name} u SET embedding_next = CAST(:vec AS vector), "
                f"embedding_next_model = :model FROM ("
                f"SELECT t.{spec.pk} AS pk FROM {spec.source_from} "
                f"WHERE t.{spec.pk} = CAST(:pk AS uuid) AND t.{spec.vec} IS NOT NULL "
                f"AND t.embedding_next_model IS DISTINCT FROM :next "
                f"AND md5({spec.src}) IS NOT DISTINCT FROM :seen) c "
                f"WHERE u.{spec.pk} = c.pk"
            )
        if spec.kind == "entity":
            return (
                f"UPDATE {spec.name} SET embedding_next = CAST(:vec AS vector), "
                f"embedding_next_model = :model WHERE {spec.pk} = CAST(:pk AS uuid) "
                f"AND {spec.vec} IS NOT NULL AND embedding_next_model IS DISTINCT FROM :next"
            )
        return (
            f"UPDATE {spec.name} SET embedding_next = CAST(:vec AS vector), "
            f"embedding_next_model = :model WHERE {spec.pk} = CAST(:pk AS uuid)"
        )

    async def _write_back(self, sql: str, items: list[dict]) -> int:
        """Run guarded updates in one short transaction; return rows written."""
        written = 0
        async with self._engine.begin() as conn:
            for it in items:
                res = await conn.execute(text(sql), it)
                written += 1 if (res.rowcount or 0) > 0 else 0
        return written

    def _is_skipped(self, spec: TableSpec, pk: str, key: str) -> bool:
        return (spec.name, pk, key) in self._skipped

    def _skipped_pks(self, table: str, key: str) -> list[str]:
        return [k[1] for k in self._skipped if k[0] == table and k[2] == key]

    # -- rows whose source was skipped -------------------------------------

    #: Episodes with a traversal record, none of which is staged or can still
    #: be (a record is claimable only while it has a vector and is not skipped).
    DEAD_EPISODES_SQL = (
        "SELECT e.id FROM trellion_episodes e "
        "WHERE e.embedding IS NOT NULL AND e.embedding_next_model IS DISTINCT FROM :next "
        "AND EXISTS (SELECT 1 FROM trellion_traversal_records r WHERE r.video_frame_id = e.id) "
        "AND NOT EXISTS (SELECT 1 FROM trellion_traversal_records r "
        "  WHERE r.video_frame_id = e.id AND (r.embedding_next_model = :next "
        "    OR (r.query_embedding IS NOT NULL AND r.id <> ALL(CAST(:skip AS uuid[])))))"
    )

    #: Clusters with no member vector and no child centroid that is staged or
    #: can still be (``entity_embeddings.embedding`` is NOT NULL, so every
    #: unskipped member row is claimable).
    DEAD_CLUSTERS_SQL = (
        "SELECT c.cluster_id FROM entity_clusters c "
        "WHERE c.centroid IS NOT NULL AND c.embedding_next_model IS DISTINCT FROM :next "
        "AND NOT EXISTS (SELECT 1 FROM entity_cluster_members m "
        "  JOIN entity_embeddings ee ON ee.entity_id = m.entity_id "
        "  WHERE m.cluster_id = c.cluster_id AND (ee.embedding_next_model = :next "
        "    OR ee.entity_id <> ALL(CAST(:skip_ent AS uuid[])))) "
        "AND NOT EXISTS (SELECT 1 FROM entity_clusters ch "
        "  WHERE ch.parent_cluster_id = c.cluster_id AND ch.centroid IS NOT NULL "
        "    AND (ch.embedding_next_model = :next "
        "      OR ch.cluster_id <> ALL(CAST(:skip_cl AS uuid[]))))"
    )

    async def skip_dependents(self, nxt: str) -> int:
        """Skip derived rows that have nothing left to derive from.

        Without this, an episode whose traversal record was skipped, or a
        cluster whose members were all skipped, stays in ``total`` and never
        reaches ``done``, so the migration can never complete. Skipped like a
        poison row: reported as unsourced and dropped at cutover. A dead leaf
        can kill its parent, so clusters repeat until nothing new is found.
        """
        added = 0
        async with self._engine.connect() as conn:
            rows = (
                await conn.execute(
                    text(self.DEAD_EPISODES_SQL),
                    {"next": nxt, "skip": self._skipped_pks("trellion_traversal_records", nxt)},
                )
            ).fetchall()
            added += self._skip_derived("trellion_episodes", rows, nxt)
            for _ in range(16):  # cluster levels
                rows = (
                    await conn.execute(
                        text(self.DEAD_CLUSTERS_SQL),
                        {
                            "next": nxt,
                            "skip_ent": self._skipped_pks("entity_embeddings", nxt),
                            "skip_cl": self._skipped_pks("entity_clusters", nxt),
                        },
                    )
                ).fetchall()
                n = self._skip_derived("entity_clusters", rows, nxt)
                added += n
                if n == 0:
                    break
        return added

    def _skip_derived(self, table: str, rows, nxt: str) -> int:
        n = 0
        for (pk,) in rows:
            k = (table, str(pk), nxt)
            if k not in self._skipped:
                self._skipped.add(k)
                n += 1
                log.warning("reembed.row_skipped", table=table, pk=str(pk), key=nxt,
                            error="nothing left to derive it from")
        return n

    async def _fill_text(self, spec: TableSpec, rec: SpaceRecord) -> int:
        nxt = rec.next_model or ""
        after = self._cursor.get(spec.name)
        params: dict[str, Any] = {"next": nxt, "limit": self._batch}
        if after:
            params["after"] = after
        # CLAIM: a short transaction, committed before anything is embedded.
        async with self._engine.connect() as conn:
            async with conn.begin():
                rows = (
                    await conn.execute(
                        text(self.claim_sql(spec, cursor=bool(after))), params
                    )
                ).fetchall()
        if not rows:
            return 0
        # EMBED: no transaction and no connection held.
        live = [r for r in rows if not self._is_skipped(spec, str(r[0]), nxt)]
        if live:
            pks = [str(r[0]) for r in live]
            texts = [
                (r[1][:ENTRY_CLIP_CHARS] if spec.name == "noo_entries" else r[1]) or ""
                for r in live
            ]
            vecs, served = await self._embed_rows(spec, pks, texts, self._embed_next, nxt)
            good = [(r, v) for r, v in zip(live, vecs) if v is not None]
            if good:
                model = self._check_vectors([v for _, v in good], served, rec, len(good))
                # WRITE BACK: only rows whose source is still what was read.
                written = await self._write_back(
                    self.store_next_sql(spec),
                    [{"vec": _lit(v), "model": model, "pk": str(r[0]), "seen": r[2],
                      "next": nxt} for r, v in good],
                )
                self._log_writeback(spec, len(good), written)
        self._cursor[spec.name] = str(rows[-1][0])
        return len(rows)

    @staticmethod
    def _log_writeback(spec: TableSpec, attempted: int, written: int) -> None:
        if written < attempted:
            log.info("reembed.writeback_skipped", table=spec.name,
                     skipped=attempted - written,
                     note="source changed or already staged; the next scan retries")

    # -- entity embeddings -------------------------------------------------

    _ENTITY_CLAIM = (
        "SELECT t.entity_id FROM entity_embeddings t "
        "WHERE t.embedding IS NOT NULL AND t.embedding_next_model IS DISTINCT FROM :next"
        "{after} ORDER BY t.entity_id LIMIT :limit FOR NO KEY UPDATE OF t SKIP LOCKED"
    )

    #: Neighbour selection per trellion-protocols' contract: the same rows,
    #: cap and order the entity-vector writer uses.
    _ENTITY_NEIGHBOURS = (
        "SELECT ent.entity_id, ent.canonical_name, ent.type, nb.name, nb.type, "
        "       nb.rel_type, nb.direction "
        "FROM entities ent "
        "LEFT JOIN LATERAL ("
        "  SELECT n.canonical_name AS name, n.type AS type, e.rel_type AS rel_type, "
        "         CASE WHEN e.src_entity_id = ent.entity_id THEN 'outgoing' "
        "              ELSE 'incoming' END AS direction, "
        f"         ROW_NUMBER() OVER (ORDER BY {ENTITY_NEIGHBOUR_ORDER_SQL}) AS rn "
        "  FROM edges e "
        "  JOIN entities n ON n.entity_id = CASE WHEN e.src_entity_id = ent.entity_id "
        "                                        THEN e.dst_entity_id ELSE e.src_entity_id END "
        "  WHERE e.src_entity_id = ent.entity_id OR e.dst_entity_id = ent.entity_id "
        f"  ORDER BY {ENTITY_NEIGHBOUR_ORDER_SQL} LIMIT {ENTITY_DESCRIPTION_MAX_NEIGHBOURS}"
        ") nb ON TRUE "
        "WHERE ent.entity_id = ANY(CAST(:ids AS uuid[])) "
        "ORDER BY ent.entity_id, nb.rn"
    )

    async def _describe(self, conn, ids: list[str]) -> dict[str, str]:
        """``entity_id -> description`` for the ids that still exist."""
        rows = (await conn.execute(text(self._ENTITY_NEIGHBOURS), {"ids": ids})).fetchall()
        by_id: dict[str, dict] = {}
        for eid, name, etype, nname, ntype, rel, direction in rows:
            ent = by_id.setdefault(str(eid), {"name": name, "type": etype, "nb": []})
            if nname is not None:
                ent["nb"].append(
                    {"name": nname, "type": ntype, "rel_type": rel, "direction": direction}
                )
        return {i: entity_description(e["name"], e["type"], e["nb"]) for i, e in by_id.items()}

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
                desc = await self._describe(conn, ids)
        present = [i for i in ids if i in desc and not self._is_skipped(spec, i, nxt)]
        if present:
            vecs, served = await self._embed_rows(
                spec, present, [desc[i] for i in present], self._embed_next, nxt
            )
            good = [(i, v) for i, v in zip(present, vecs) if v is not None]
            if good:
                model = self._check_vectors([v for _, v in good], served, rec, len(good))
                written = await self._write_entities(
                    spec, good, {i: desc[i] for i, _ in good}, model, nxt
                )
                self._log_writeback(spec, len(good), written)
        self._cursor[spec.name] = ids[-1]
        return len(ids)

    async def _write_entities(
        self, spec: TableSpec, good: list, embedded: dict[str, str], model: str, nxt: str,
        *, live: bool = False,
    ) -> int:
        """Write entity vectors whose description is still what was embedded.

        The recheck runs in the write transaction: the description is rebuilt
        from the current ``entities``/``edges`` and compared with the text the
        vector came from.
        """
        written = 0
        async with self._engine.begin() as conn:
            now = await self._describe(conn, [i for i, _ in good])
            for i, v in good:
                if now.get(i) != embedded[i]:
                    continue
                params = (
                    {"vec": _lit(v), "model": model, "dim": len(v), "pk": i, "cur": nxt}
                    if live
                    else {"vec": _lit(v), "model": model, "pk": i, "next": nxt}
                )
                sql = self.store_live_sql(spec) if live else self.store_next_sql(spec)
                res = await conn.execute(text(sql), params)
                written += 1 if (res.rowcount or 0) > 0 else 0
        return written

    # -- cluster centroids -------------------------------------------------

    #: Clusters whose inputs are all staged. A leaf (has member rows) needs
    #: every member's vector re-embedded; a parent (no member rows) needs every
    #: child centroid done. Skipped inputs (``:skip_ent`` / ``:skip_cl``) are
    #: not waited for. Selecting only READY clusters means nothing blocks the
    #: scan and no cursor is needed.
    CLUSTER_READY_SQL = (
        "SELECT c.cluster_id, c.level FROM entity_clusters c "
        "WHERE c.centroid IS NOT NULL AND c.embedding_next_model IS DISTINCT FROM :next "
        "AND c.cluster_id <> ALL(CAST(:skip_cl AS uuid[])) AND ("
        "  (EXISTS (SELECT 1 FROM entity_cluster_members m WHERE m.cluster_id = c.cluster_id)"
        "   AND EXISTS (SELECT 1 FROM entity_cluster_members m "
        "               JOIN entity_embeddings ee ON ee.entity_id = m.entity_id "
        "               WHERE m.cluster_id = c.cluster_id AND ee.embedding_next_model = :next)"
        "   AND NOT EXISTS (SELECT 1 FROM entity_cluster_members m "
        "               JOIN entity_embeddings ee ON ee.entity_id = m.entity_id "
        "               WHERE m.cluster_id = c.cluster_id "
        "                 AND ee.embedding_next_model IS DISTINCT FROM :next "
        "                 AND ee.entity_id <> ALL(CAST(:skip_ent AS uuid[]))))"
        "  OR (NOT EXISTS (SELECT 1 FROM entity_cluster_members m WHERE m.cluster_id = c.cluster_id)"
        "   AND EXISTS (SELECT 1 FROM entity_clusters ch WHERE ch.parent_cluster_id = c.cluster_id "
        "               AND ch.centroid IS NOT NULL AND ch.embedding_next_model = :next)"
        "   AND NOT EXISTS (SELECT 1 FROM entity_clusters ch "
        "               WHERE ch.parent_cluster_id = c.cluster_id AND ch.centroid IS NOT NULL "
        "                 AND ch.embedding_next_model IS DISTINCT FROM :next "
        "                 AND ch.cluster_id <> ALL(CAST(:skip_cl AS uuid[]))))"
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
                        text(self.CLUSTER_READY_SQL),
                        {
                            "next": nxt,
                            "limit": self._batch,
                            "skip_ent": self._skipped_pks("entity_embeddings", nxt),
                            "skip_cl": self._skipped_pks("entity_clusters", nxt),
                        },
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
                                    "AND centroid IS NOT NULL AND embedding_next_model = :next"
                                ),
                                {"c": cid, "next": nxt},
                            )
                        ).fetchall()
                        if not kids:
                            continue  # nothing staged to derive from (raced a rewrite)
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
        done = "t.embedding_model" if swapped else "t.embedding_next_model"
        u = spec.unsourced
        return (
            f"SELECT count(*) FILTER (WHERE NOT ({u})), "
            f"count(*) FILTER (WHERE NOT ({u}) AND {done} = :next), "
            f"count(*) FILTER (WHERE {u}) "
            f"FROM {spec.name} t WHERE t.{spec.vec} IS NOT NULL"
        )

    async def counts(self, nxt: str, *, swapped: bool = False) -> dict[str, tuple[int, int, int]]:
        """``table -> (done, total, unsourced)`` for the progress report.

        Unsourced = no source/derivation, plus rows skipped for this migration.
        Neither is in ``done`` or ``total``.
        """
        out: dict[str, tuple[int, int, int]] = {}
        async with self._engine.connect() as conn:
            for spec in TABLES:
                row = (
                    await conn.execute(
                        text(self.count_sql(spec, swapped=swapped)), {"next": nxt}
                    )
                ).one()
                total, done, unsourced = int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)
                if not swapped:
                    skipped = sum(1 for k in self._skipped if k[0] == spec.name and k[2] == nxt)
                    total -= skipped
                    unsourced += skipped
                out[spec.name] = (done, max(total, 0), unsourced)
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

    @staticmethod
    def unsourced_null_sql(spec: TableSpec) -> str:
        """After the swap: an unsourced row keeps no vector under the new model."""
        return (
            f"UPDATE {spec.name} t SET {spec.vec} = NULL, embedding_model = NULL, "
            f"embedding_dim = NULL, embedding_next = NULL, embedding_next_model = NULL "
            f"WHERE t.{spec.vec} IS NOT NULL AND ({spec.unsourced})"
        )

    @staticmethod
    def skipped_null_sql(spec: TableSpec) -> str:
        """Rows skipped for this migration, at cutover.

        ``text`` and ``episode`` rows keep their old vector under its OLD model
        label (only the staging is cleared). Every read filters by model, so
        the row is out of search either way, and the stale sweep re-embeds it
        with the new model once the migration is over (an episode copies from
        its healed traversal record). Nulling it would lose it for good:
        nothing else re-embeds a traversal record.

        ``entity_embeddings.embedding`` is NOT NULL, so its row is deleted (the
        graph embedder treats "no row" as stale and regenerates it). A cluster
        is nulled; the owner's next clustering run rebuilds it.
        """
        if spec.kind in ("text", "episode"):
            return (
                f"UPDATE {spec.name} SET embedding_next = NULL, embedding_next_model = NULL "
                f"WHERE {spec.pk} = ANY(CAST(:ids AS uuid[])) "
                f"AND embedding_next_model IS DISTINCT FROM :next"
            )
        if spec.name == "entity_embeddings":
            return (
                "DELETE FROM entity_embeddings WHERE entity_id = ANY(CAST(:ids AS uuid[])) "
                "AND embedding_next_model IS DISTINCT FROM :next"
            )
        return (
            f"UPDATE {spec.name} SET {spec.vec} = NULL, embedding_model = NULL, "
            f"embedding_dim = NULL, embedding_next = NULL, embedding_next_model = NULL "
            f"WHERE {spec.pk} = ANY(CAST(:ids AS uuid[])) "
            f"AND embedding_next_model IS DISTINCT FROM :next"
        )

    async def _cutover(self, rec: SpaceRecord) -> int:
        nxt = rec.next_model or ""
        total = 0
        for spec in TABLES:
            skipped = [k[1] for k in self._skipped if k[0] == spec.name and k[2] == nxt]
            async with self._engine.begin() as conn:
                res = await conn.execute(
                    text(self.cutover_sql(spec)), {"next": nxt, "dim": rec.next_dims}
                )
                swapped = res.rowcount or 0
                dropped = 0
                if skipped:
                    r = await conn.execute(
                        text(self.skipped_null_sql(spec)), {"ids": skipped, "next": nxt}
                    )
                    dropped += r.rowcount or 0
                if spec.unsourced != "FALSE":
                    r = await conn.execute(text(self.unsourced_null_sql(spec)))
                    dropped += r.rowcount or 0
            total += swapped + dropped
            if swapped or dropped:
                log.info(
                    "reembed.cutover_table", table=spec.name, swapped=swapped, unsourced_nulled=dropped
                )
        await self.report(nxt, cutover_done=True)
        return total

    # ------------------------------------------------------------------
    # Stale sweep (no migration running)
    # ------------------------------------------------------------------

    @staticmethod
    def stale_claim_sql(spec: TableSpec, *, skip: bool) -> str:
        skip_sql = f" AND t.{spec.pk} <> ALL(CAST(:skip AS uuid[]))" if skip else ""
        return (
            f"SELECT t.{spec.pk} AS pk, {spec.src} AS src, md5({spec.src}) AS seen "
            f"FROM {spec.source_from} "
            f"WHERE t.{spec.vec} IS NOT NULL AND t.embedding_model IS NOT NULL "
            f"AND t.embedding_model <> :cur{skip_sql} "
            f"ORDER BY t.{spec.pk} LIMIT :limit FOR NO KEY UPDATE OF t SKIP LOCKED"
        )

    @staticmethod
    def store_live_sql(spec: TableSpec) -> str:
        """Write a healed live vector, only while the row is still stale.

        ``embedding_model <> :cur`` is the stale predicate again, so a second
        worker (or a writer that already stored a current-model vector) makes
        this a no-op. A ``text`` row's source must also still hash to ``:seen``.
        """
        stale = (
            f"t.{spec.vec} IS NOT NULL AND t.embedding_model IS NOT NULL "
            f"AND t.embedding_model <> :cur"
        )
        if spec.kind == "text":
            return (
                f"UPDATE {spec.name} u SET {spec.vec} = CAST(:vec AS vector), "
                f"embedding_model = :model, embedding_dim = :dim, "
                f"embedding_next = NULL, embedding_next_model = NULL FROM ("
                f"SELECT t.{spec.pk} AS pk FROM {spec.source_from} "
                f"WHERE t.{spec.pk} = CAST(:pk AS uuid) AND {stale} "
                f"AND md5({spec.src}) IS NOT DISTINCT FROM :seen) c "
                f"WHERE u.{spec.pk} = c.pk"
            )
        return (
            f"UPDATE {spec.name} t SET {spec.vec} = CAST(:vec AS vector), "
            f"embedding_model = :model, embedding_dim = :dim, "
            f"embedding_next = NULL, embedding_next_model = NULL "
            f"WHERE t.{spec.pk} = CAST(:pk AS uuid) AND {stale}"
        )

    @staticmethod
    def episode_stale_copy_sql() -> str:
        return (
            "WITH pick AS ("
            " SELECT e.id, t.query_embedding, t.embedding_model, t.embedding_dim "
            " FROM trellion_episodes e "
            " JOIN trellion_traversal_records t ON t.video_frame_id = e.id "
            " WHERE e.embedding IS NOT NULL AND e.embedding_model IS NOT NULL "
            "   AND e.embedding_model <> :cur AND t.query_embedding IS NOT NULL "
            "   AND t.embedding_model = :cur "
            " LIMIT :limit FOR NO KEY UPDATE OF e SKIP LOCKED) "
            "UPDATE trellion_episodes e SET embedding = p.query_embedding, "
            "embedding_model = p.embedding_model, embedding_dim = p.embedding_dim, "
            "embedding_next = NULL, embedding_next_model = NULL "
            "FROM pick p WHERE e.id = p.id"
        )

    async def stale_sweep(self, rec: SpaceRecord) -> int:
        """Re-embed sourced rows written under a model that is not the space's.

        Text and entity tables are re-embedded with the plain space model;
        episodes copy from their (healed) traversal record. Cluster centroids
        are derived and follow their members on the owner's next run.
        """
        if self._embed_live is None:
            return 0
        cur = rec.model
        key = f"live:{cur}"
        total = 0
        for spec in TABLES:
            if spec.kind not in ("text", "entity", "episode"):
                continue
            for _ in range(LEGACY_MAX_BATCHES_PER_PASS):
                try:
                    handled, written = await self._stale_batch(spec, rec, cur, key)
                except Exception as exc:  # noqa: BLE001 - one table must not stall the rest
                    log.warning("reembed.stale_failed", table=spec.name, error=str(exc)[:200])
                    break
                total += written
                if handled == 0:
                    break
        if total:
            log.info("reembed.stale_healed", rows=total, model=cur)
        return total

    async def _stale_batch(
        self, spec: TableSpec, rec: SpaceRecord, cur: str, key: str
    ) -> tuple[int, int]:
        """One stale batch. Returns ``(rows handled, rows written)``."""
        if spec.kind == "episode":
            async with self._engine.begin() as conn:
                res = await conn.execute(
                    text(self.episode_stale_copy_sql()), {"cur": cur, "limit": self._batch}
                )
            n = res.rowcount or 0
            return n, n
        skip = [k[1] for k in self._skipped if k[0] == spec.name and k[2] == key]
        params: dict[str, Any] = {"cur": cur, "limit": self._batch}
        if skip:
            params["skip"] = skip
        # CLAIM: a short transaction, committed before anything is embedded.
        async with self._engine.connect() as conn:
            async with conn.begin():
                if spec.kind == "entity":
                    sql = (
                        "SELECT t.entity_id, NULL, NULL FROM entity_embeddings t WHERE "
                        "t.embedding IS NOT NULL AND t.embedding_model IS NOT NULL "
                        "AND t.embedding_model <> :cur"
                        + (" AND t.entity_id <> ALL(CAST(:skip AS uuid[]))" if skip else "")
                        + " ORDER BY t.entity_id LIMIT :limit FOR NO KEY UPDATE OF t SKIP LOCKED"
                    )
                    rows = (await conn.execute(text(sql), params)).fetchall()
                    ids = [str(r[0]) for r in rows]
                    desc = await self._describe(conn, ids) if ids else {}
                    pks = [i for i in ids if i in desc]
                    texts = [desc[i] for i in pks]
                    seen: dict[str, Optional[str]] = {}
                else:
                    rows = (
                        await conn.execute(text(self.stale_claim_sql(spec, skip=bool(skip))), params)
                    ).fetchall()
                    pks = [str(r[0]) for r in rows]
                    texts = [
                        (r[1][:ENTRY_CLIP_CHARS] if spec.name == "noo_entries" else r[1]) or ""
                        for r in rows
                    ]
                    seen = {str(r[0]): r[2] for r in rows}
        if not rows:
            return 0, 0
        written = 0
        if pks:
            # EMBED: no transaction and no connection held.
            vecs, served = await self._embed_rows(spec, pks, texts, self._embed_live, key)
            good = [(pk, v) for pk, v in zip(pks, vecs) if v is not None]
            if good:
                model = self._check_vectors(
                    [v for _, v in good], served, rec, len(good), live=True
                )
                # WRITE BACK: only rows that are still stale and unchanged.
                if spec.kind == "entity":
                    written = await self._write_entities(
                        spec, good, dict(zip(pks, texts)), model, cur, live=True
                    )
                else:
                    written = await self._write_back(
                        self.store_live_sql(spec),
                        [{"vec": _lit(v), "model": model, "dim": len(v), "pk": pk,
                          "seen": seen.get(pk), "cur": cur} for pk, v in good],
                    )
                self._log_writeback(spec, len(good), written)
        return len(rows), written
