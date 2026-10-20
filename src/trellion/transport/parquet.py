# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""ParquetShardPackager — pack/unpack memory slices as Parquet tarballs.

Tarball layout (matches proposal 06 §7(e.2)):

    trellion-shard-{persona_id}-{generation}.tar
    ├── manifest.json           # schema_version, slice_spec, hashes, table list
    ├── entities.parquet        # subgraph entities (id, name, type, metadata)
    ├── edges.parquet           # subgraph edges (src, dst, rel_type, valid_from, ...)
    ├── frames.parquet          # video_frame rows for the slice
    ├── traversals.parquet      # traversal_records for the slice
    └── faiss.index             # binary FAISS index sidecar (optional)

Compression: ZSTD by default (cold shards), Snappy if ``hot=True``
(faster decompression, larger files).

This implementation favours pyarrow + DuckDB as the read-side ecosystem.
Edge nodes that need to query a shard without trellion installed can do
``duckdb.sql("SELECT * FROM 'edges.parquet' WHERE ...")`` directly.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

logger = structlog.get_logger(__name__)

SHARD_SCHEMA_VERSION = 1

# Tables included in a shard
ENTITIES_TABLE = "entities.parquet"
EDGES_TABLE = "edges.parquet"
FRAMES_TABLE = "frames.parquet"
TRAVERSALS_TABLE = "traversals.parquet"
FAISS_SIDECAR = "faiss.index"
MANIFEST_FILE = "manifest.json"


@dataclass
class ShardSliceSpec:
    """Selects which rows go into a shard.

    At least one of ``persona_id`` or ``topic_entity_id`` must be set.
    """
    persona_id: str | None = None
    topic_entity_id: str | None = None  # str(uuid) of seed noo_entities_typed.id
    topic_hops: int = 2
    time_window: tuple[datetime, datetime] | None = None
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "topic_entity_id": self.topic_entity_id,
            "topic_hops": self.topic_hops,
            "time_window": (
                [self.time_window[0].isoformat(), self.time_window[1].isoformat()]
                if self.time_window else None
            ),
            "label": self.label,
        }


@dataclass
class ShardManifest:
    """Metadata travelling with each shard tarball."""
    schema_version: int = SHARD_SCHEMA_VERSION
    shard_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    persona_id: str | None = None
    slice_spec: dict[str, Any] = field(default_factory=dict)
    table_files: list[str] = field(default_factory=list)
    table_row_counts: dict[str, int] = field(default_factory=dict)
    table_hashes: dict[str, str] = field(default_factory=dict)
    has_faiss_sidecar: bool = False
    faiss_hash: str | None = None
    compression: str = "zstd"
    trellion_version: str = "0.0.2"

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "shard_id": self.shard_id,
                "created_at": self.created_at,
                "persona_id": self.persona_id,
                "slice_spec": self.slice_spec,
                "table_files": self.table_files,
                "table_row_counts": self.table_row_counts,
                "table_hashes": self.table_hashes,
                "has_faiss_sidecar": self.has_faiss_sidecar,
                "faiss_hash": self.faiss_hash,
                "compression": self.compression,
                "trellion_version": self.trellion_version,
            },
            indent=2,
            sort_keys=True,
        )


class ParquetShardPackager:
    """Pack live data into a shard tarball; unpack into a fresh instance.

    Both methods are async because they query trellion's AsyncEngine.
    File I/O is sync but small relative to the DB scan, so we don't
    bother offloading.
    """

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        faiss_base_path: str | None = None,
        compression: str = "zstd",
    ):
        self._engine = engine
        self._faiss_base_path = faiss_base_path
        self._compression = compression

    # ------------------------------------------------------------------
    # Pack
    # ------------------------------------------------------------------

    async def pack(
        self,
        spec: ShardSliceSpec,
        out_path: Path | str,
        *,
        include_faiss: bool = True,
    ) -> Path:
        """Build a shard tarball at ``out_path``. Returns the tarball path."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "pyarrow required for shard packaging. "
                "pip install pyarrow"
            ) from exc

        out_path = Path(out_path)
        manifest = ShardManifest(
            persona_id=spec.persona_id,
            slice_spec=spec.to_dict(),
            compression=self._compression,
        )

        with tempfile.TemporaryDirectory(prefix="trellion-shard-") as tmpdir:
            staging = Path(tmpdir)

            # 1. Entities + edges (live KG slice)
            entity_ids = await self._collect_entity_ids(spec)
            ent_rows = await self._fetch_entities(entity_ids)
            edge_rows = await self._fetch_edges(entity_ids, spec.time_window)
            self._write_parquet(staging / ENTITIES_TABLE, ent_rows, pa, pq)
            self._write_parquet(staging / EDGES_TABLE, edge_rows, pa, pq)
            manifest.table_files.append(ENTITIES_TABLE)
            manifest.table_files.append(EDGES_TABLE)
            manifest.table_row_counts[ENTITIES_TABLE] = len(ent_rows)
            manifest.table_row_counts[EDGES_TABLE] = len(edge_rows)

            # 2. Video frames + traversals (System 1 slice — persona-scoped)
            if spec.persona_id:
                frame_rows = await self._fetch_frames(spec.persona_id)
                trav_rows = await self._fetch_traversals(spec.persona_id)
                self._write_parquet(staging / FRAMES_TABLE, frame_rows, pa, pq)
                self._write_parquet(staging / TRAVERSALS_TABLE, trav_rows, pa, pq)
                manifest.table_files.append(FRAMES_TABLE)
                manifest.table_files.append(TRAVERSALS_TABLE)
                manifest.table_row_counts[FRAMES_TABLE] = len(frame_rows)
                manifest.table_row_counts[TRAVERSALS_TABLE] = len(trav_rows)

            # 3. FAISS sidecar (optional)
            if include_faiss and spec.persona_id and self._faiss_base_path:
                faiss_src = Path(self._faiss_base_path) / spec.persona_id / "faiss"
                if faiss_src.exists():
                    # Pack the persona's FAISS dir as a sub-tar so internal
                    # multi-file structure (conversations.index, meta.json, etc.) is preserved
                    faiss_tar_path = staging / FAISS_SIDECAR
                    with tarfile.open(faiss_tar_path, "w") as ftar:
                        for f in faiss_src.iterdir():
                            ftar.add(f, arcname=f.name)
                    manifest.has_faiss_sidecar = True
                    manifest.faiss_hash = _file_sha256(faiss_tar_path)

            # 4. Compute hashes
            for table_name in manifest.table_files:
                manifest.table_hashes[table_name] = _file_sha256(staging / table_name)

            # 5. Write manifest + bundle into final tar
            (staging / MANIFEST_FILE).write_text(manifest.to_json())

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(out_path, "w") as out_tar:
                out_tar.add(staging / MANIFEST_FILE, arcname=MANIFEST_FILE)
                for table_name in manifest.table_files:
                    out_tar.add(staging / table_name, arcname=table_name)
                if manifest.has_faiss_sidecar:
                    out_tar.add(staging / FAISS_SIDECAR, arcname=FAISS_SIDECAR)

        logger.info(
            "shard.packed",
            out=str(out_path),
            entities=manifest.table_row_counts.get(ENTITIES_TABLE, 0),
            edges=manifest.table_row_counts.get(EDGES_TABLE, 0),
            frames=manifest.table_row_counts.get(FRAMES_TABLE, 0),
            traversals=manifest.table_row_counts.get(TRAVERSALS_TABLE, 0),
            faiss=manifest.has_faiss_sidecar,
        )
        return out_path

    # ------------------------------------------------------------------
    # Unpack
    # ------------------------------------------------------------------

    async def unpack(
        self,
        tarball_path: Path | str,
        *,
        verify_hashes: bool = True,
    ) -> ShardManifest:
        """Extract a shard tarball and write its rows into the live DB.

        Idempotent: rows are inserted with ``ON CONFLICT DO NOTHING`` so
        re-applying the same shard is a no-op. FAISS sidecar is unpacked
        to ``<faiss_base_path>/<persona_id>/faiss/``.
        """
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("pyarrow required to unpack shards") from exc

        tarball_path = Path(tarball_path)
        with tempfile.TemporaryDirectory(prefix="trellion-unpack-") as tmpdir:
            staging = Path(tmpdir)
            with tarfile.open(tarball_path, "r") as tar:
                tar.extractall(staging)

            manifest_path = staging / MANIFEST_FILE
            if not manifest_path.exists():
                raise ValueError(f"Shard tarball missing {MANIFEST_FILE}")
            manifest_data = json.loads(manifest_path.read_text())
            manifest = ShardManifest(**manifest_data)

            if verify_hashes:
                for table_name, expected in manifest.table_hashes.items():
                    actual = _file_sha256(staging / table_name)
                    if actual != expected:
                        raise ValueError(
                            f"Shard hash mismatch for {table_name}: "
                            f"expected {expected}, got {actual}"
                        )

            # Apply rows table-by-table
            for table_name in manifest.table_files:
                table_path = staging / table_name
                rows = pq.read_table(str(table_path)).to_pylist()
                if not rows:
                    continue
                if table_name == ENTITIES_TABLE:
                    await self._apply_entities(rows)
                elif table_name == EDGES_TABLE:
                    await self._apply_edges(rows)
                elif table_name == FRAMES_TABLE:
                    await self._apply_frames(rows)
                elif table_name == TRAVERSALS_TABLE:
                    await self._apply_traversals(rows)
                else:
                    logger.warning("shard.unknown_table_skipped", name=table_name)

            # FAISS sidecar
            if manifest.has_faiss_sidecar and self._faiss_base_path:
                if verify_hashes and manifest.faiss_hash:
                    actual = _file_sha256(staging / FAISS_SIDECAR)
                    if actual != manifest.faiss_hash:
                        raise ValueError(
                            f"FAISS sidecar hash mismatch: "
                            f"expected {manifest.faiss_hash}, got {actual}"
                        )
                if manifest.persona_id:
                    target = Path(self._faiss_base_path) / manifest.persona_id / "faiss"
                    target.mkdir(parents=True, exist_ok=True)
                    with tarfile.open(staging / FAISS_SIDECAR, "r") as ftar:
                        ftar.extractall(target)

        logger.info(
            "shard.unpacked",
            shard_id=manifest.shard_id,
            persona=manifest.persona_id,
            tables=len(manifest.table_files),
        )
        return manifest

    # ------------------------------------------------------------------
    # Pack — DB reads
    # ------------------------------------------------------------------

    async def _collect_entity_ids(self, spec: ShardSliceSpec) -> list[str]:
        if spec.topic_entity_id:
            # BFS from seed
            visited: set[str] = {spec.topic_entity_id}
            frontier: set[str] = {spec.topic_entity_id}
            async with self._engine.connect() as conn:
                for _ in range(spec.topic_hops):
                    if not frontier:
                        break
                    rows = (await conn.execute(
                        text(
                            "SELECT subject_id::text, object_entity_id::text "
                            "FROM noo_temporal_edges "
                            "WHERE valid_to IS NULL "
                            "  AND object_entity_id IS NOT NULL "
                            "  AND (subject_id::text = ANY(:ids) OR object_entity_id::text = ANY(:ids))"
                        ),
                        {"ids": list(frontier)},
                    )).fetchall()
                    next_frontier: set[str] = set()
                    for src, dst in rows:
                        if src and src not in visited:
                            next_frontier.add(src)
                        if dst and dst not in visited:
                            next_frontier.add(dst)
                    visited.update(next_frontier)
                    frontier = next_frontier
            return list(visited)

        if spec.persona_id:
            # All entities the user has touched
            async with self._engine.connect() as conn:
                rows = (await conn.execute(
                    text(
                        "SELECT object_entity_id::text FROM noo_temporal_edges te "
                        "JOIN noo_entities_typed e ON e.id = te.subject_id "
                        "WHERE e.type = 'user' AND lower(e.canonical_name) = :p "
                        "  AND te.object_entity_id IS NOT NULL"
                    ),
                    {"p": spec.persona_id.lower()},
                )).fetchall()
            return [r[0] for r in rows if r[0]]

        return []

    async def _fetch_entities(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                text(
                    "SELECT id::text AS id, type, canonical_name, "
                    "       metadata::text AS metadata_json, created_at "
                    "FROM noo_entities_typed WHERE id::text = ANY(:ids)"
                ),
                {"ids": ids},
            )).mappings().fetchall()
        return [_row_to_dict(r) for r in rows]

    async def _fetch_edges(
        self,
        ids: list[str],
        time_window: tuple[datetime, datetime] | None,
    ) -> list[dict[str, Any]]:
        if not ids:
            return []
        sql = (
            "SELECT id::text AS id, subject_id::text AS subject_id, predicate, "
            "       object_entity_id::text AS object_entity_id, object_value, confidence, "
            "       valid_from, valid_to, observed_at "
            "FROM noo_temporal_edges "
            "WHERE subject_id::text = ANY(:ids) "
            "  AND (object_entity_id IS NULL OR object_entity_id::text = ANY(:ids))"
        )
        params: dict[str, Any] = {"ids": ids}
        if time_window:
            sql += " AND valid_from >= :f AND valid_from <= :t"
            params["f"] = time_window[0]
            params["t"] = time_window[1]
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text(sql), params)).mappings().fetchall()
        return [_row_to_dict(r) for r in rows]

    async def _fetch_frames(self, persona_id: str) -> list[dict[str, Any]]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                text(
                    "SELECT id::text AS id, persona_id, record_type, faiss_index_name, "
                    "       content_summary, prominence_score, access_count, "
                    "       first_accessed_at, last_accessed_at, avg_search_score, "
                    "       is_tombstoned, metadata::text AS metadata_json, "
                    "       created_at, updated_at "
                    "FROM trellion_video_frames WHERE persona_id = :p"
                ),
                {"p": persona_id},
            )).mappings().fetchall()
        return [_row_to_dict(r) for r in rows]

    async def _fetch_traversals(self, persona_id: str) -> list[dict[str, Any]]:
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                text(
                    "SELECT id::text AS id, persona_id, query_text, "
                    "       query_embedding::text AS query_embedding_text, "
                    "       embedding_model, intent, "
                    "       traversal_path::text AS traversal_path_json, outcome_score, "
                    "       exploration_rate, "
                    "       uncertainty_fingerprint::text AS uncertainty_fingerprint_json, "
                    "       archived_to_video, video_frame_id::text AS video_frame_id, "
                    "       created_at, updated_at "
                    "FROM trellion_traversal_records WHERE persona_id = :p"
                ),
                {"p": persona_id},
            )).mappings().fetchall()
        return [_row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Unpack — DB writes (idempotent)
    # ------------------------------------------------------------------

    async def _apply_entities(self, rows: list[dict[str, Any]]) -> None:
        async with self._engine.begin() as conn:
            for r in rows:
                await conn.execute(
                    text(
                        "INSERT INTO noo_entities_typed "
                        "(id, type, canonical_name, metadata, created_at) "
                        "VALUES (CAST(:id AS uuid), :type, :name, "
                        "        CAST(:meta AS jsonb), :created_at) "
                        "ON CONFLICT (type, canonical_name) DO NOTHING"
                    ),
                    {
                        "id": r["id"],
                        "type": r["type"],
                        "name": r["canonical_name"],
                        "meta": r.get("metadata_json") or "{}",
                        "created_at": r["created_at"],
                    },
                )

    async def _apply_edges(self, rows: list[dict[str, Any]]) -> None:
        async with self._engine.begin() as conn:
            for r in rows:
                await conn.execute(
                    text(
                        "INSERT INTO noo_temporal_edges "
                        "(id, subject_id, predicate, object_entity_id, object_value, "
                        " confidence, valid_from, valid_to, observed_at) "
                        "VALUES (CAST(:id AS uuid), CAST(:subj AS uuid), :pred, "
                        "        CASE WHEN :obj_id IS NULL THEN NULL::uuid ELSE CAST(:obj_id AS uuid) END, "
                        "        :obj_val, :conf, :vf, :vt, :obs) "
                        "ON CONFLICT (id) DO NOTHING"
                    ),
                    {
                        "id": r["id"], "subj": r["subject_id"], "pred": r["predicate"],
                        "obj_id": r.get("object_entity_id"), "obj_val": r.get("object_value"),
                        "conf": r["confidence"], "vf": r["valid_from"], "vt": r.get("valid_to"),
                        "obs": r["observed_at"],
                    },
                )

    async def _apply_frames(self, rows: list[dict[str, Any]]) -> None:
        async with self._engine.begin() as conn:
            for r in rows:
                await conn.execute(
                    text(
                        "INSERT INTO trellion_video_frames "
                        "(id, persona_id, record_type, faiss_index_name, content_summary, "
                        " prominence_score, access_count, first_accessed_at, last_accessed_at, "
                        " avg_search_score, is_tombstoned, metadata, created_at, updated_at) "
                        "VALUES (CAST(:id AS uuid), :p, :rt, :idx, :summary, "
                        "        :prom, :ac, :first, :last, :avg, :tomb, "
                        "        CAST(:meta AS jsonb), :created, :updated) "
                        "ON CONFLICT (id) DO NOTHING"
                    ),
                    {
                        "id": r["id"], "p": r["persona_id"], "rt": r["record_type"],
                        "idx": r["faiss_index_name"], "summary": r["content_summary"],
                        "prom": r["prominence_score"], "ac": r["access_count"],
                        "first": r.get("first_accessed_at"), "last": r.get("last_accessed_at"),
                        "avg": r.get("avg_search_score"), "tomb": r["is_tombstoned"],
                        "meta": r.get("metadata_json") or "{}",
                        "created": r["created_at"], "updated": r["updated_at"],
                    },
                )

    async def _apply_traversals(self, rows: list[dict[str, Any]]) -> None:
        async with self._engine.begin() as conn:
            for r in rows:
                await conn.execute(
                    text(
                        "INSERT INTO trellion_traversal_records "
                        "(id, persona_id, query_text, query_embedding, "
                        " embedding_model, embedding_dim, intent, "
                        " traversal_path, outcome_score, exploration_rate, "
                        " uncertainty_fingerprint, archived_to_video, video_frame_id, "
                        " created_at, updated_at) "
                        "VALUES (CAST(:id AS uuid), :p, :q, "
                        "        CASE WHEN :emb IS NULL THEN NULL::vector ELSE CAST(:emb AS vector) END, "
                        # The model travels with the shard (a shard from an older
                        # build carries none: NULL, labelled by the worker's legacy
                        # claim). The width is the vector's own.
                        "        CASE WHEN :emb IS NULL THEN NULL ELSE CAST(:emb_model AS text) END, "
                        "        CASE WHEN :emb IS NULL THEN NULL "
                        "             ELSE vector_dims(CAST(:emb AS vector)) END, "
                        "        :intent, "
                        "        CAST(:path AS jsonb), :outcome, :rate, "
                        "        CAST(:uf AS jsonb), :archived, "
                        "        CASE WHEN :fid IS NULL THEN NULL::uuid ELSE CAST(:fid AS uuid) END, "
                        "        :created, :updated) "
                        "ON CONFLICT (id) DO NOTHING"
                    ),
                    {
                        "id": r["id"], "p": r["persona_id"], "q": r["query_text"],
                        "emb": r.get("query_embedding_text"),
                        "emb_model": r.get("embedding_model"),
                        "intent": r.get("intent"),
                        "path": r.get("traversal_path_json") or "[]",
                        "outcome": r["outcome_score"], "rate": r.get("exploration_rate"),
                        "uf": r.get("uncertainty_fingerprint_json") or "{}",
                        "archived": r["archived_to_video"], "fid": r.get("video_frame_id"),
                        "created": r["created_at"], "updated": r["updated_at"],
                    },
                )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_parquet(self, path: Path, rows: list[dict[str, Any]], pa, pq) -> None:
        if not rows:
            # Write an empty file with no schema — readers just see 0 rows
            empty = pa.table({})
            pq.write_table(empty, str(path), compression=self._compression)
            return
        table = pa.Table.from_pylist(_serialise_rows(rows))
        pq.write_table(table, str(path), compression=self._compression)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _row_to_dict(row) -> dict[str, Any]:
    """Convert a SQLAlchemy mapping row to a plain dict (for pyarrow)."""
    return {k: v for k, v in row.items()}


def _serialise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce non-Arrow types (datetimes, UUIDs) to strings.

    Datetimes are kept as datetime instances — pyarrow handles them.
    UUIDs were already cast to ``::text`` in the SELECT, so they arrive
    as strings. Lists/dicts that came back as JSONB are already Python
    primitives.
    """
    return rows
