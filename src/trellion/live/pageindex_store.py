# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""PageIndex + Knowledge Graph CRUD store.

Uses the shared SQLAlchemy async engine from ``noomesh.db``
instead of creating its own psycopg async connection.
Schema is managed by Alembic.

Provides async operations for documents, pages, chunks,
embeddings, entities, mentions, and graph edges.
"""

from __future__ import annotations

from .. import json_fast as json
import asyncio
import os
import uuid
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine
from trellion_protocols.schema import ROLES_WHERE

from trellion import embedding_space

if TYPE_CHECKING:
    from trellion.live.recall.walk_store import WalkStore

log = structlog.get_logger(__name__)


class EmbedClaim:
    """Rows picked for embedding, and the recheck-guarded write-back.

    ``rows`` are the picked rows, read in a short transaction that has already
    committed: NO transaction or connection is held while the caller embeds
    (with ``NOO_EMBED_BULK`` that wait is minutes). ``store`` writes the
    vectors back in its own short transaction, and writes a row only if its
    text is still what was read (``seen``, an md5 taken at pick time) AND it
    still has no usable vector. A row that changed, vanished, was held, or was
    already embedded by another worker is skipped and left for the next pass;
    ``written`` lists the ids that were actually written.
    """

    def __init__(self, rows: list[dict], engine: Any, upsert_sql: str, key: str, check, vec) -> None:
        self.rows = rows
        self._engine = engine
        self._sql = upsert_sql
        self._key = key
        self._check = check
        self._vec = vec
        self._seen = {r[key]: r.get("seen") for r in rows}
        self.written: list[str] = []

    async def store(self, items: list[tuple[str, list[float]]]) -> int:
        if not items:
            return 0
        await self._check(items[0][1])
        async with self._engine.begin() as conn:
            for i, e in items:
                res = await conn.execute(
                    text(self._sql),
                    {self._key: i, "vec": self._vec(e), "seen": self._seen.get(i),
                     **_provenance(e)},
                )
                if (res.rowcount or 0) > 0:
                    self.written.append(i)
        skipped = len(items) - len(self.written)
        if skipped:
            log.info("embed_writeback_skipped", key=self._key, skipped=skipped,
                     note="source changed, held, or already embedded; next pass retries")
        return len(self.written)


def _provenance(vector: Any) -> dict:
    """The ``:model`` / ``:dim`` bind values recorded beside a vector.

    ``model`` is the model the embed RESPONSE named (carried on the vector by
    :mod:`trellion.embedding_space`); a bare list records NULL and is labelled
    by the worker's legacy claim. ``dim`` is the vector's own width.
    """
    return {"model": embedding_space.model_of(vector), "dim": len(vector) if vector else None}


def _get_embedding_dim() -> int:
    """Return the expected embedding dimension from config (preferred) or env.

    Used as the constructor-time fallback when ``PageIndexStore`` is built
    without an explicit ``embedding_dim``. Hot-path call sites use the
    ``self._vec`` method which reads the resolved dimension off the instance.
    """
    try:
        from ..config import get_config
        return get_config().services.embedding_dim
    except Exception:
        return int(os.environ.get("EMBEDDING_DIM", "768"))


class PageIndexStore:
    """Async CRUD operations for the PageIndex + Graph schema."""

    def __init__(
        self,
        async_engine: Optional[AsyncEngine] = None,
        dsn: Optional[str] = None,
        embedding_dim: Optional[int] = None,
    ) -> None:
        if async_engine is not None:
            self._engine = async_engine
        elif dsn is not None:
            from trellion.db.engine import TRELLION_SCHEMA, _async_dsn
            from sqlalchemy.ext.asyncio import create_async_engine
            # Pin search_path like every other engine builder (db/engine.py):
            # trellion's tables live in schema `trellion` and every query
            # here is unqualified — a raw engine sees only `public`, which
            # is exactly the fresh-install failure the temporal-graph test
            # fixture documents. Found live by the P1.5 bench harness.
            self._engine = create_async_engine(
                _async_dsn(dsn),
                pool_pre_ping=True,
                connect_args={"options": f"-csearch_path={TRELLION_SCHEMA},public"},
            )
        else:
            from trellion.db.engine import get_async_engine
            self._engine = get_async_engine()
        self._embedding_dim = (
            embedding_dim if embedding_dim is not None else _get_embedding_dim()
        )
        # ids this process has picked for embedding and not yet written back,
        # per id column: concurrent batches here must not pick the same rows
        # now that the pick holds no lock across the embed.
        self._inflight: dict[str, set[str]] = {}
        self._pick_lock = asyncio.Lock()

    # -- the add-on contract (trellion_protocols.GraphStore) -------------
    # Public names for what add-ons reach into
    # (``_engine``, ``_vec``). The private names stay: this module and its
    # callers use them.

    @property
    def engine(self) -> AsyncEngine:
        """The store's async engine (search_path pinned to trellion's schema)."""
        return self._engine

    @property
    def walk_store(self) -> "WalkStore":
        """A recallable-walks store over the same engine."""
        from trellion.live.recall.walk_store import WalkStore

        return WalkStore(self._engine)

    def vec(self, values: list[float]) -> str:
        """Format a float list as a pgvector literal, validating dimension."""
        return self._vec(values)

    def _vec(self, values: list[float]) -> str:
        """Format a float list as a pgvector literal, validating dimension."""
        expected = self._embedding_dim
        if len(values) != expected:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(values)}, "
                f"expected {expected} (services.embedding_dim). "
                f"Check your embedding model or set EMBEDDING_DIM={len(values)}."
            )
        return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"

    async def connect(self) -> None:
        """Validate connectivity (schema is managed by Alembic)."""
        async with self._engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        log.info("pageindex_store.connected")

    async def close(self) -> None:
        """No-op — engine lifecycle is managed by db.close_engines()."""
        pass

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    async def insert_doc(
        self,
        source_uri: str,
        title: str,
        content_hash: str,
        metadata: Optional[dict] = None,
    ) -> tuple[str, bool]:
        """Insert a document. Returns ``(doc_id, is_new)``.

        When the content hash already exists the existing ``doc_id`` is
        returned with ``is_new=False`` so the caller can skip page/chunk
        creation (deduplication). If *metadata* is provided on a dedup
        hit, the existing row's metadata is merged (new keys win) so a
        re-ingest from a different source can enrich existing entries.

        **Aliased URIs** — when a different `source_uri` collides on
        content hash with an existing row, the canonical `source_uri`
        column stays at the first-seen path (we don't want to flip-flop
        ownership). The new path is appended to ``metadata.aliased_uris``
        so dedup queries can still see it. Without this the same content
        at multiple paths would re-ingest forever (caller's dedup query
        on `source_uri` would never see the second/third paths).
        """
        import json as _json

        doc_id = str(uuid.uuid4())
        meta_json = _json.dumps(metadata) if metadata else None

        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                INSERT INTO docs (doc_id, source_uri, title, hash, metadata)
                VALUES (:doc_id, :source_uri, :title, :hash, CAST(:metadata AS jsonb))
                ON CONFLICT (hash) DO UPDATE
                  SET title = EXCLUDED.title,
                      metadata = COALESCE(docs.metadata, '{}'::jsonb)
                                 || COALESCE(EXCLUDED.metadata, '{}'::jsonb)
                                 || jsonb_build_object(
                                      'aliased_uris',
                                      CASE
                                        WHEN EXCLUDED.source_uri IS NULL
                                          OR EXCLUDED.source_uri = ''
                                          OR docs.source_uri = EXCLUDED.source_uri
                                          OR COALESCE(docs.metadata->'aliased_uris', '[]'::jsonb)
                                             @> to_jsonb(EXCLUDED.source_uri)
                                        THEN COALESCE(docs.metadata->'aliased_uris', '[]'::jsonb)
                                        ELSE COALESCE(docs.metadata->'aliased_uris', '[]'::jsonb)
                                             || jsonb_build_array(EXCLUDED.source_uri)
                                      END
                                    )
                RETURNING doc_id
                """),
                {
                    "doc_id": doc_id,
                    "source_uri": source_uri,
                    "title": title,
                    "hash": content_hash,
                    "metadata": meta_json,
                },
            )
            row = result.fetchone()
            await conn.commit()
            returned_id = str(row[0])
            return returned_id, (returned_id == doc_id)

    async def select_docs_for_retag(
        self,
        *,
        uri_pattern: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
        only_untagged: bool = True,
        limit: int = 1000,
    ) -> list[dict]:
        """Documents a retag would touch, WITHOUT touching them.

        The read half of :meth:`set_doc_roles`, so a caller can show an
        operator exactly what is about to change. Returns ``doc_id``,
        ``source_uri``, ``title`` and the CURRENT ``roles`` (``[]`` when the
        row carries none).

        *uri_pattern* is a SQL ``LIKE`` pattern, not a glob — the caller
        translates. ``only_untagged`` is the safe default: backfilling a
        document that nobody has classified is a different act from
        overwriting a classification somebody chose.
        """
        clauses = []
        params: dict = {"limit": max(1, min(int(limit), 100_000))}
        if uri_pattern:
            clauses.append("source_uri LIKE :uri")
            params["uri"] = uri_pattern
        if doc_ids:
            clauses.append("doc_id = ANY(CAST(:ids AS uuid[]))")
            params["ids"] = list(doc_ids)
        if only_untagged:
            # Absent key, JSON null, and empty array all mean "untagged".
            # Testing only for the absent key would silently skip the rows a
            # half-finished tagging run left behind.
            clauses.append(
                "(metadata->'roles' IS NULL "
                " OR jsonb_typeof(metadata->'roles') = 'null' "
                " OR jsonb_array_length(COALESCE(metadata->'roles', '[]'::jsonb)) = 0)"
            )
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT doc_id, source_uri, title, "
                    "       COALESCE(metadata->'roles', '[]'::jsonb) AS roles "
                    f"FROM docs{where} ORDER BY created_at LIMIT :limit"
                ),
                params,
            )
            return [
                {
                    "doc_id": str(r[0]),
                    "source_uri": r[1],
                    "title": r[2],
                    "roles": list(r[3] or []),
                }
                for r in result.fetchall()
            ]

    async def set_doc_roles(self, doc_ids: list[str], roles: list[str]) -> int:
        """Set ``metadata.roles`` on the named documents. Returns the count.

        A RELABEL, not a merge: the new list replaces whatever was there.
        Merging would make the operation impossible to undo — you could add
        an audience but never remove one — and "take this out of company"
        is the half of retagging that actually matters.

        Everything else in ``metadata`` is preserved: the update writes one
        key with ``||``, so ``aliased_uris`` and any ingest-time fields
        survive. An empty *roles* list clears the tag, returning the
        document to untagged rather than storing ``[]``, so the two states
        this repo already distinguishes stay distinguishable.
        """
        if not doc_ids:
            return 0
        import json as _json

        async with self._engine.connect() as conn:
            if roles:
                sql = (
                    "UPDATE docs SET metadata = "
                    "  COALESCE(metadata, '{}'::jsonb) "
                    "  || jsonb_build_object('roles', CAST(:roles AS jsonb)) "
                    "WHERE doc_id = ANY(CAST(:ids AS uuid[]))"
                )
                params = {"ids": list(doc_ids), "roles": _json.dumps(list(roles))}
            else:
                sql = (
                    "UPDATE docs SET metadata = "
                    "  COALESCE(metadata, '{}'::jsonb) - 'roles' "
                    "WHERE doc_id = ANY(CAST(:ids AS uuid[]))"
                )
                params = {"ids": list(doc_ids)}
            result = await conn.execute(text(sql), params)
            await conn.commit()
            return int(result.rowcount or 0)

    async def get_ingested_source_uris(self) -> set[str]:
        """Return all known source URIs (canonical + aliased) from docs.

        Includes ``metadata.aliased_uris`` entries so callers iterating
        for dedup see every path that maps to an existing doc, not just
        the first-ingested path.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT source_uri FROM docs
                WHERE source_uri IS NOT NULL AND source_uri != ''
                UNION
                SELECT jsonb_array_elements_text(metadata->'aliased_uris')
                FROM docs
                WHERE jsonb_typeof(metadata->'aliased_uris') = 'array'
                """),
            )
            return {str(row[0]) for row in result.fetchall()}

    async def delete_doc(self, doc_id: str) -> bool:
        """Delete a document and all its pages/chunks/embeddings/mentions.

        FK ``ON DELETE CASCADE`` handles the child rows.
        Returns True if the document existed and was deleted.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("DELETE FROM docs WHERE doc_id = :doc_id"),
                {"doc_id": doc_id},
            )
            await conn.commit()
            return result.rowcount > 0

    # ------------------------------------------------------------------
    # Document hold (migration 105)
    # ------------------------------------------------------------------

    @staticmethod
    def escape_like(prefix: str) -> str:
        """Make *prefix* a LITERAL for ``LIKE ... ESCAPE '\\'``: a title
        containing ``%``, ``_`` or ``\\`` must match itself, not a pattern."""
        return (
            prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        )

    @classmethod
    def _hold_selector(
        cls, title_prefix: Optional[str], doc_ids: Optional[list[str]]
    ) -> tuple[str, dict]:
        """WHERE clause + params for exactly one selector, else ValueError."""
        has_prefix = title_prefix is not None
        has_ids = doc_ids is not None
        if has_prefix == has_ids:
            raise ValueError("give exactly one of title_prefix or doc_ids")
        if has_prefix:
            if not title_prefix:
                raise ValueError("title_prefix must not be empty")
            return (
                "title LIKE :pat ESCAPE '\\'",
                {"pat": cls.escape_like(title_prefix) + "%"},
            )
        if not doc_ids:
            raise ValueError("doc_ids must not be empty")
        return "doc_id = ANY(CAST(:ids AS uuid[]))", {"ids": list(doc_ids)}

    async def hold_docs(
        self,
        title_prefix: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
        reason: str = "",
        apply: bool = False,
    ) -> dict:
        """Hold documents: keep them, take them out of embedding and search.

        Exactly one selector. Without *apply* nothing changes and the result
        says what WOULD be held. Re-holding a held document is a no-op (its
        original ``held_at`` and reason stand), so ``count`` and ``doc_ids``
        are the documents this call newly holds. Nothing is deleted.
        """
        if not reason or not reason.strip():
            raise ValueError("a hold needs a reason")
        sel, params = self._hold_selector(title_prefix, doc_ids)
        async with self._engine.connect() as conn:
            if not apply:
                res = await conn.execute(
                    text(f"SELECT doc_id FROM docs WHERE {sel} AND held_at IS NULL "
                         "ORDER BY created_at, doc_id"),
                    params,
                )
            else:
                res = await conn.execute(
                    text(f"UPDATE docs SET held_at = NOW(), hold_reason = :reason "
                         f"WHERE {sel} AND held_at IS NULL RETURNING doc_id"),
                    {**params, "reason": reason},
                )
                await conn.commit()
            ids = sorted(str(r[0]) for r in res.fetchall())
        return {"count": len(ids), "doc_ids": ids, "applied": bool(apply)}

    async def unhold_docs(
        self,
        title_prefix: Optional[str] = None,
        doc_ids: Optional[list[str]] = None,
        apply: bool = False,
    ) -> dict:
        """Release held documents: eligible for embedding and search again."""
        sel, params = self._hold_selector(title_prefix, doc_ids)
        async with self._engine.connect() as conn:
            if not apply:
                res = await conn.execute(
                    text(f"SELECT doc_id FROM docs WHERE {sel} AND held_at IS NOT NULL "
                         "ORDER BY created_at, doc_id"),
                    params,
                )
            else:
                res = await conn.execute(
                    text(f"UPDATE docs SET held_at = NULL, hold_reason = NULL "
                         f"WHERE {sel} AND held_at IS NOT NULL RETURNING doc_id"),
                    params,
                )
                await conn.commit()
            ids = sorted(str(r[0]) for r in res.fetchall())
        return {"count": len(ids), "doc_ids": ids, "applied": bool(apply)}

    async def list_held(
        self,
        reason: Optional[str] = None,
        limit: int = 1000,
        offset: int = 0,
    ) -> dict:
        """Held documents (id, title, reason, held_at) in a stable order,
        plus the ``total`` so a caller can page to the end."""
        where = "held_at IS NOT NULL"
        params: dict = {}
        if reason:
            where += " AND hold_reason = :reason"
            params["reason"] = reason
        limit = max(1, min(int(limit), 100_000))
        offset = max(0, int(offset))
        async with self._engine.connect() as conn:
            total = (
                await conn.execute(text(f"SELECT COUNT(*) FROM docs WHERE {where}"), params)
            ).scalar() or 0
            res = await conn.execute(
                text(f"SELECT doc_id, title, hold_reason, held_at FROM docs "
                     f"WHERE {where} ORDER BY held_at, doc_id LIMIT :limit OFFSET :offset"),
                {**params, "limit": limit, "offset": offset},
            )
            docs = [
                {
                    "doc_id": str(r[0]),
                    "title": r[1],
                    "reason": r[2],
                    "held_at": r[3].isoformat() if r[3] else None,
                }
                for r in res.fetchall()
            ]
        return {"total": int(total), "docs": docs, "limit": limit, "offset": offset}

    async def purge_held(
        self, reason: str, apply: bool = False, batch: int = 500
    ) -> dict:
        """DELETE documents already held under exactly *reason*.

        The final, irreversible step after a hold. A document that is not
        held, or is held under another reason, is never touched; there is no
        "purge everything held". Deletes in batches, each its own
        transaction, through the same cascade as :meth:`delete_doc`, so a
        large purge never holds one long lock. Without *apply* nothing
        changes and ``count`` is what would go.
        """
        if not reason or not reason.strip():
            raise ValueError("a purge needs the reason the documents are held under")
        batch = max(1, int(batch))
        async with self._engine.connect() as conn:
            res = await conn.execute(
                text("SELECT doc_id FROM docs WHERE held_at IS NOT NULL "
                     "AND hold_reason = :reason ORDER BY doc_id"),
                {"reason": reason},
            )
            ids = [str(r[0]) for r in res.fetchall()]
        if not apply:
            return {"count": len(ids), "doc_ids": ids, "deleted": 0}
        deleted = 0
        for i in range(0, len(ids), batch):
            chunk = ids[i : i + batch]
            async with self._engine.connect() as conn:
                # Re-checks the hold in the DELETE itself: a doc released
                # between the listing and this batch is not deleted.
                r = await conn.execute(
                    text("DELETE FROM docs WHERE doc_id = ANY(CAST(:ids AS uuid[])) "
                         "AND held_at IS NOT NULL AND hold_reason = :reason"),
                    {"ids": chunk, "reason": reason},
                )
                await conn.commit()
                deleted += int(r.rowcount or 0)
        return {"count": len(ids), "doc_ids": ids, "deleted": deleted}

    async def deduplicate_pages(self) -> int:
        """Remove duplicate pages that share the same (doc_id, page_no).

        Keeps the most recently inserted row (highest page_id) for each
        pair and deletes the rest.  Returns the number of rows removed.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(text("""
                DELETE FROM pages
                WHERE page_id IN (
                    SELECT page_id FROM (
                        SELECT page_id,
                               ROW_NUMBER() OVER (
                                   PARTITION BY doc_id, page_no
                                   ORDER BY page_id DESC
                               ) AS rn
                        FROM pages
                    ) sub
                    WHERE rn > 1
                )
            """))
            removed = result.rowcount
            await conn.commit()
            return removed

    async def count_duplicate_pages(self) -> int:
        """Return the number of duplicate page rows (same doc_id + page_no)."""
        async with self._engine.connect() as conn:
            result = await conn.execute(text("""
                SELECT COALESCE(SUM(cnt - 1), 0) FROM (
                    SELECT COUNT(*) AS cnt
                    FROM pages
                    GROUP BY doc_id, page_no
                    HAVING COUNT(*) > 1
                ) sub
            """))
            return result.scalar() or 0

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------

    async def insert_page(
        self,
        doc_id: str,
        page_no: int,
        text_content: str,
        meta: Optional[dict] = None,
    ) -> str:
        """Insert a page. Returns page_id.

        Uses ON CONFLICT on ``(doc_id, page_no)`` so re-ingestion of
        the same document with updated content replaces the old page
        rather than creating duplicates.
        """
        page_id = str(uuid.uuid4())
        meta_json = json.dumps(meta) if meta else None
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                INSERT INTO pages (page_id, doc_id, page_no, text, meta_json)
                VALUES (:page_id, :doc_id, :page_no, :text, CAST(:meta_json AS jsonb))
                ON CONFLICT (doc_id, page_no) DO UPDATE
                    SET text = EXCLUDED.text, meta_json = EXCLUDED.meta_json
                RETURNING page_id
                """),
                {"page_id": page_id, "doc_id": doc_id, "page_no": page_no,
                 "text": text_content, "meta_json": meta_json},
            )
            row = result.fetchone()
            await conn.commit()
        return str(row[0])

    async def get_page(self, page_id: str) -> Optional[dict]:
        """Return a single page as a dict, or None."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT page_id, doc_id, page_no, text, meta_json
                FROM pages WHERE page_id = :page_id
                """),
                {"page_id": page_id},
            )
            row = result.fetchone()
            if not row:
                return None
            return {
                "page_id": str(row[0]),
                "doc_id": str(row[1]),
                "page_no": row[2],
                "text": row[3],
                "meta": row[4],
            }

    async def get_doc_pages(self, doc_id: str) -> list[dict]:
        """Return all pages for a document, ordered by page_no."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT page_id, doc_id, page_no, text, meta_json
                FROM pages WHERE doc_id = :doc_id
                ORDER BY page_no
                """),
                {"doc_id": doc_id},
            )
            rows = result.fetchall()
            return [
                {
                    "page_id": str(r[0]),
                    "doc_id": str(r[1]),
                    "page_no": r[2],
                    "text": r[3],
                    "meta": r[4],
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    async def insert_chunks(self, page_id: str, chunks: list[str]) -> list[str]:
        """Insert multiple chunks for a page. Returns list of chunk_ids."""
        chunk_ids: list[str] = []
        async with self._engine.connect() as conn:
            for idx, chunk_text in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                await conn.execute(
                    text("""
                    INSERT INTO page_chunks (chunk_id, page_id, chunk_no, text)
                    VALUES (:chunk_id, :page_id, :chunk_no, :text)
                    """),
                    {"chunk_id": chunk_id, "page_id": page_id, "chunk_no": idx, "text": chunk_text},
                )
                chunk_ids.append(chunk_id)
            await conn.commit()
        return chunk_ids

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------


    # ── dimension guard ────────────────────────────────────────────────
    #
    # The pgvector column type IS the authoritative declaration of the
    # store's embedding space: `vector(1024)` means 1024, and atttypmod
    # carries it. Anything else claiming a dimension — env vars, config,
    # a model's reputation — is a second declaration that can go stale.
    #
    # Why this guard exists: on 2026-07-29 the fleet moved its embedder
    # from bge (768) to harrier (1024). Before the schema migration, a
    # mis-routed mesh call returned 768-dim vectors that were written
    # without complaint; after it, 768-dim writes failed with a raw
    # pgvector DataError repeated hundreds of times with no explanation of
    # the cause. Neither is acceptable: the first corrupts the space
    # silently, the second is unreadable.
    #
    # A mismatch refuses WRITES only. Searches over existing vectors keep
    # working, because the stored space is still internally consistent —
    # it is the incoming vector that is foreign.

    _dims_cache: dict[str, int] = {}

    async def vector_dims(self, table: str, column: str = "embedding") -> int:
        """Declared dimension of a pgvector column, or 0 if unknown.

        Cached per table — a column's type does not change without a
        migration, and this sits on the embedding write path.
        """
        key = f"{table}.{column}"
        if key in self._dims_cache:
            return self._dims_cache[key]
        try:
            async with self._engine.connect() as conn:
                row = await conn.execute(
                    text(
                        "SELECT a.atttypmod FROM pg_attribute a "
                        "JOIN pg_class c ON c.oid = a.attrelid "
                        "WHERE c.relname = :t AND a.attname = :col "
                        "AND a.attnum > 0 AND NOT a.attisdropped"
                    ),
                    {"t": table, "col": column},
                )
                dims = int(row.scalar() or 0)
        except Exception as exc:  # noqa: BLE001
            log.warning("vector_dims_probe_failed", table=table, error=str(exc))
            return 0
        if dims > 0:
            self._dims_cache[key] = dims
        return dims

    async def _check_dims(self, table: str, embedding: list[float]) -> None:
        """Raise before writing a vector whose width the column cannot hold.

        Raising beats letting pgvector reject it: the error names the
        expected and actual dimensions and what to do, instead of a bare
        "expected 1024 dimensions, not 768" with no indication of which
        model produced it or why.
        """
        expected = await self.vector_dims(table)
        if expected <= 0 or len(embedding) == expected:
            return
        raise ValueError(
            f"refusing to write a {len(embedding)}-dim vector into "
            f"{table}.embedding, which is vector({expected}). The embedder is "
            f"serving a different model than the store was built for. Either "
            f"point the role=embed placement back at a {expected}-dim model, "
            f"or migrate the vector columns and re-embed. Writes are refused; "
            f"searches over existing vectors still work."
        )

    async def store_page_embedding(
        self, page_id: str, embedding: list[float]
    ) -> None:
        """Upsert embedding for a page."""
        await self._check_dims("page_embeddings", embedding)
        vec = self._vec(embedding)
        async with self._engine.connect() as conn:
            await conn.execute(
                text("""
                INSERT INTO page_embeddings (page_id, embedding, embedding_model, embedding_dim)
                VALUES (:page_id, CAST(:vec AS vector), :model, :dim)
                ON CONFLICT (page_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    embedding_next = NULL,
                    embedding_next_model = NULL
                """),
                {"page_id": page_id, "vec": vec, **_provenance(embedding)},
            )
            await conn.commit()

    async def store_chunk_embedding(
        self, chunk_id: str, embedding: list[float]
    ) -> None:
        """Upsert embedding for a chunk."""
        await self._check_dims("chunk_embeddings", embedding)
        vec = self._vec(embedding)
        async with self._engine.connect() as conn:
            await conn.execute(
                text("""
                INSERT INTO chunk_embeddings (chunk_id, embedding, embedding_model, embedding_dim)
                VALUES (:chunk_id, CAST(:vec AS vector), :model, :dim)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    embedding_next = NULL,
                    embedding_next_model = NULL
                """),
                {"chunk_id": chunk_id, "vec": vec, **_provenance(embedding)},
            )
            await conn.commit()

    async def store_page_embeddings_batch(
        self, items: list[tuple[str, list[float]]]
    ) -> int:
        """Bulk upsert page embeddings via executemany (pipelined in asyncpg)."""
        if not items:
            return 0
        # Checked on the FIRST item only: a batch comes from one embedder
        # call, so a width mismatch is uniform. This is the path the
        # re-embed actually uses — guarding only the single-write methods
        # would leave bulk traffic, i.e. all of it, unchecked.
        await self._check_dims("page_embeddings", items[0][1])
        rows = [{"page_id": pid, "vec": self._vec(emb), **_provenance(emb)} for pid, emb in items]
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                INSERT INTO page_embeddings (page_id, embedding, embedding_model, embedding_dim)
                VALUES (:page_id, CAST(:vec AS vector), :model, :dim)
                ON CONFLICT (page_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    embedding_next = NULL,
                    embedding_next_model = NULL
                """),
                rows,
            )
        return len(items)

    async def store_chunk_embeddings_batch(
        self, items: list[tuple[str, list[float]]]
    ) -> int:
        """Bulk upsert chunk embeddings via executemany (pipelined in asyncpg)."""
        if not items:
            return 0
        await self._check_dims("chunk_embeddings", items[0][1])
        rows = [{"chunk_id": cid, "vec": self._vec(emb), **_provenance(emb)} for cid, emb in items]
        async with self._engine.begin() as conn:
            await conn.execute(
                text("""
                INSERT INTO chunk_embeddings (chunk_id, embedding, embedding_model, embedding_dim)
                VALUES (:chunk_id, CAST(:vec AS vector), :model, :dim)
                ON CONFLICT (chunk_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dim = EXCLUDED.embedding_dim,
                    embedding_next = NULL,
                    embedding_next_model = NULL
                """),
                rows,
            )
        return len(items)

    # Role-based KB segmentation: docs carry audience tags in
    # metadata.roles (e.g. ["system"], ["user:alice"], ["company"]).
    # A query carries a role SET; a doc matches when the sets overlap.
    # Untagged docs are treated as ["user"] (backward compatibility —
    # everything ingested before roles existed is personal-tier).
    # roles=None means unrestricted (operator/internal callers).
    # jsonb_exists* functions instead of ?/?| operators — no driver
    # paramstyle escaping concerns.
    # The predicate itself is part of the add-on contract
    # (trellion_protocols.schema.ROLES_WHERE): add-on audience
    # filters and the gateway's scope.rs apply the same one. Public name for
    # add-ons, the historical private name for this module's own queries.
    ROLES_WHERE = ROLES_WHERE
    _ROLES_WHERE = ROLES_WHERE

    # A held document (migration 105) is kept but invisible: not embedded,
    # not searched, not counted. `d` is the docs row.
    _NOT_HELD = "d.held_at IS NULL"

    async def search_pages(
        self,
        embedding: list[float],
        top_k: int = 10,
        doc_ids: Optional[list[str]] = None,
        roles: Optional[list[str]] = None,
    ) -> list[dict]:
        """Vector similarity search over page embeddings.

        ``roles`` restricts results to documents whose ``metadata.roles``
        overlaps the given set (see ``_ROLES_WHERE``). ``None`` = no
        role restriction.
        """
        vec = self._vec(embedding)
        params: dict = {"vec": vec, "vec2": vec, "top_k": top_k}
        # A held document is never returned (migration 105): the join to
        # docs is unconditional so the hold applies with or without roles.
        join_docs = "JOIN docs d ON d.doc_id = p.doc_id"
        where: list[str] = [self._NOT_HELD]
        if doc_ids:
            where.append("p.doc_id = ANY(:doc_ids)")
            params["doc_ids"] = doc_ids
        if roles is not None:
            where.append(self._ROLES_WHERE)
            params["roles"] = list(roles)
        # Only vectors from the space's current model are comparable to the
        # query vector (dropped, never emptied, when the model is unknown).
        model_sql, model_params = await embedding_space.model_clause(
            "pe.embedding_model", query_vector=embedding
        )
        if model_sql:
            where.append(model_sql)
            params.update(model_params)
        where_sql = "WHERE " + " AND ".join(where)
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"""
                SELECT p.page_id, p.doc_id, p.page_no, p.text,
                       1 - (pe.embedding <=> CAST(:vec AS vector)) AS score
                FROM page_embeddings pe
                JOIN pages p ON p.page_id = pe.page_id
                {join_docs}
                {where_sql}
                ORDER BY pe.embedding <=> CAST(:vec2 AS vector)
                LIMIT :top_k
                """),
                params,
            )
            rows = result.fetchall()
            return [
                {
                    "page_id": str(r[0]),
                    "doc_id": str(r[1]),
                    "page_no": r[2],
                    "text": r[3],
                    "score": float(r[4]),
                }
                for r in rows
            ]

    async def search_chunks(
        self,
        embedding: list[float],
        top_k: int = 10,
        page_ids: Optional[list[str]] = None,
        roles: Optional[list[str]] = None,
    ) -> list[dict]:
        """Vector similarity search over chunk embeddings.

        ``roles`` applies the same audience filter as ``search_pages``
        (joined through pages → docs). The retriever always scopes chunk
        search to already-role-filtered ``page_ids``, so this is
        defence-in-depth for any future caller that searches globally.
        """
        vec = self._vec(embedding)
        params: dict = {"vec": vec, "vec2": vec, "top_k": top_k}
        # A held document is never returned (migration 105): the join to
        # pages and docs is unconditional so the hold applies with or
        # without roles.
        join_docs = (
            "JOIN pages p ON p.page_id = c.page_id "
            "JOIN docs d ON d.doc_id = p.doc_id"
        )
        where: list[str] = [self._NOT_HELD]
        if page_ids:
            where.append("c.page_id = ANY(:page_ids)")
            params["page_ids"] = page_ids
        if roles is not None:
            where.append(self._ROLES_WHERE)
            params["roles"] = roles
        model_sql, model_params = await embedding_space.model_clause(
            "ce.embedding_model", query_vector=embedding
        )
        if model_sql:
            where.append(model_sql)
            params.update(model_params)
        where_sql = f"WHERE {' AND '.join(where)}"
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"""
                SELECT c.chunk_id, c.page_id, c.chunk_no, c.text,
                       1 - (ce.embedding <=> CAST(:vec AS vector)) AS score
                FROM chunk_embeddings ce
                JOIN page_chunks c ON c.chunk_id = ce.chunk_id
                {join_docs}
                {where_sql}
                ORDER BY ce.embedding <=> CAST(:vec2 AS vector)
                LIMIT :top_k
                """),
                params,
            )
            rows = result.fetchall()
            return [
                {
                    "chunk_id": str(r[0]),
                    "page_id": str(r[1]),
                    "chunk_no": r[2],
                    "text": r[3],
                    "score": float(r[4]),
                }
                for r in rows
            ]

    async def get_unembedded_pages(self, limit: int = 100) -> list[dict]:
        """Pages with no usable embedding — no row OR a NULL vector.

        The NULL case is not hypothetical. A dimension migration keeps the
        rows (they carry ids and metadata worth preserving) and blanks the
        vectors, so every page needing re-embedding has a row with a NULL
        embedding. Matching only on a MISSING row reported "nothing to do"
        for an entire corpus that was in fact completely unembedded.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT p.page_id, p.text
                FROM pages p
                JOIN docs d ON d.doc_id = p.doc_id
                WHERE d.held_at IS NULL
                  AND NOT EXISTS (SELECT 1 FROM page_embeddings pe
                                  WHERE pe.page_id = p.page_id AND pe.embedding IS NOT NULL)
                ORDER BY p.page_id
                LIMIT :limit
                """),
                {"limit": limit},
            )
            rows = result.fetchall()
            return [{"page_id": str(r[0]), "text": r[1]} for r in rows]

    async def get_unembedded_chunks(self, limit: int = 100) -> list[dict]:
        """Chunks with no usable embedding — no row OR a NULL vector.

        See :meth:`get_unembedded_pages`; a dimension migration blanks
        vectors in place, so "no row" alone misses the entire corpus.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT c.chunk_id, c.text
                FROM page_chunks c
                JOIN pages p ON p.page_id = c.page_id
                JOIN docs d ON d.doc_id = p.doc_id
                WHERE d.held_at IS NULL
                  AND NOT EXISTS (SELECT 1 FROM chunk_embeddings ce
                                  WHERE ce.chunk_id = c.chunk_id AND ce.embedding IS NOT NULL)
                ORDER BY c.chunk_id
                LIMIT :limit
                """),
                {"limit": limit},
            )
            rows = result.fetchall()
            return [{"chunk_id": str(r[0]), "text": r[1]} for r in rows]

    # -- claimed embedding backfill ---------------------------------------
    #
    # "Needs an embedding" is NOT EXISTS (a row with a non-NULL vector), not
    # LEFT JOIN ... WHERE ce.embedding IS NULL. With the filter above the join
    # the planner carries every vector through the hash join: 974 chunks made
    # a ~136 MB hash that spilled to a temp file on EVERY idle poll (pop-os,
    # 2026-10-26: ~130 MB/s of writes for a week). Inside NOT EXISTS the
    # IS NOT NULL test reads only the null bitmap and the join keeps ids.
    #
    # Two processes (and several batches within one) drain the same backlog.
    # A plain "SELECT ... LIMIT n" hands every one of them the SAME first
    # rows. The pick locks the child rows with FOR NO KEY UPDATE SKIP LOCKED
    # so concurrent pickers get disjoint rows, reads each row's id and an md5
    # of its text, and COMMITS: nothing is held while the caller embeds. NOT
    # FOR UPDATE: the vector insert's foreign-key check takes FOR KEY SHARE on
    # the parent row, which FOR UPDATE conflicts with.
    #
    # With no lock held across the embed, a row can be picked twice. Batches
    # in this process never are (``_inflight`` excludes them from the pick);
    # another process can, and that is harmless: the write-back inserts only
    # where the text still hashes to what was read and the row still has no
    # vector, so the second write is a no-op (rowcount 0, not counted).

    @asynccontextmanager
    async def _claim(
        self, select_sql: str, limit: int, upsert_sql: str, table: str, key: str
    ) -> AsyncIterator[EmbedClaim]:
        inflight = self._inflight.setdefault(key, set())
        # Picks in this process take turns: a pick that commits before the next
        # one starts, and has registered its ids, cannot be handed out twice.
        async with self._pick_lock:
            async with self._engine.connect() as conn:
                async with conn.begin():
                    result = await conn.execute(
                        text(select_sql), {"limit": limit, "skip": sorted(inflight)}
                    )
                    rows = [
                        {k: (str(v) if k.endswith("_id") else v) for k, v in r._asdict().items()}
                        for r in result.fetchall()
                    ]
            ids = {r[key] for r in rows}
            inflight |= ids

        async def check(vec: list[float]) -> None:
            await self._check_dims(table, vec)

        try:
            yield EmbedClaim(rows, self._engine, upsert_sql, key, check, self._vec)
        finally:
            inflight -= ids

    def claim_unembedded_pages(self, limit: int = 32) -> AbstractAsyncContextManager[EmbedClaim]:
        """Claim pages with no usable embedding (see :meth:`get_unembedded_pages`)."""
        return self._claim(
            """
            SELECT p.page_id, p.text, md5(p.text) AS seen
            FROM pages p
            JOIN docs d ON d.doc_id = p.doc_id
            WHERE d.held_at IS NULL
              AND NOT EXISTS (SELECT 1 FROM page_embeddings pe
                              WHERE pe.page_id = p.page_id AND pe.embedding IS NOT NULL)
              AND p.page_id <> ALL(CAST(:skip AS uuid[]))
            ORDER BY p.page_id
            LIMIT :limit
            FOR NO KEY UPDATE OF p SKIP LOCKED
            """,
            limit,
            """
            INSERT INTO page_embeddings (page_id, embedding, embedding_model, embedding_dim)
            SELECT p.page_id, CAST(:vec AS vector), CAST(:model AS text), CAST(:dim AS integer)
            FROM pages p JOIN docs d ON d.doc_id = p.doc_id
            WHERE p.page_id = CAST(:page_id AS uuid)
              AND md5(p.text) = :seen AND d.held_at IS NULL
            ON CONFLICT (page_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dim = EXCLUDED.embedding_dim,
                embedding_next = NULL,
                embedding_next_model = NULL
            WHERE page_embeddings.embedding IS NULL
            """,
            "page_embeddings",
            "page_id",
        )

    def claim_unembedded_chunks(self, limit: int = 32) -> AbstractAsyncContextManager[EmbedClaim]:
        """Claim chunks with no usable embedding (see :meth:`get_unembedded_chunks`)."""
        return self._claim(
            """
            SELECT c.chunk_id, c.page_id, c.text, md5(c.text) AS seen
            FROM page_chunks c
            JOIN pages p ON p.page_id = c.page_id
            JOIN docs d ON d.doc_id = p.doc_id
            WHERE d.held_at IS NULL
              AND NOT EXISTS (SELECT 1 FROM chunk_embeddings ce
                              WHERE ce.chunk_id = c.chunk_id AND ce.embedding IS NOT NULL)
              AND c.chunk_id <> ALL(CAST(:skip AS uuid[]))
            ORDER BY c.chunk_id
            LIMIT :limit
            FOR NO KEY UPDATE OF c SKIP LOCKED
            """,
            limit,
            """
            INSERT INTO chunk_embeddings (chunk_id, embedding, embedding_model, embedding_dim)
            SELECT c.chunk_id, CAST(:vec AS vector), CAST(:model AS text), CAST(:dim AS integer)
            FROM page_chunks c
            JOIN pages p ON p.page_id = c.page_id
            JOIN docs d ON d.doc_id = p.doc_id
            WHERE c.chunk_id = CAST(:chunk_id AS uuid)
              AND md5(c.text) = :seen AND d.held_at IS NULL
            ON CONFLICT (chunk_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                embedding_dim = EXCLUDED.embedding_dim,
                embedding_next = NULL,
                embedding_next_model = NULL
            WHERE chunk_embeddings.embedding IS NULL
            """,
            "chunk_embeddings",
            "chunk_id",
        )

    # ------------------------------------------------------------------
    # Graph: entities
    # ------------------------------------------------------------------

    async def upsert_entity(
        self,
        canonical_name: str,
        type: str,
        aliases: Optional[list[str]] = None,
    ) -> str:
        """Insert or update an entity. Returns entity_id."""
        entity_id = str(uuid.uuid4())
        aliases_json = json.dumps(aliases) if aliases else None
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                INSERT INTO entities (entity_id, canonical_name, type, aliases_json)
                VALUES (:entity_id, :canonical_name, :type, CAST(:aliases_json AS jsonb))
                ON CONFLICT (canonical_name, type) DO UPDATE
                    SET aliases_json = COALESCE(EXCLUDED.aliases_json, entities.aliases_json)
                RETURNING entity_id
                """),
                {"entity_id": entity_id, "canonical_name": canonical_name,
                 "type": type, "aliases_json": aliases_json},
            )
            row = result.fetchone()
            await conn.commit()
            return str(row[0])

    async def search_entities(self, name_pattern: str) -> list[dict]:
        """Search entities by name pattern (case-insensitive ILIKE)."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT entity_id, canonical_name, type, aliases_json, created_at
                FROM entities
                WHERE canonical_name ILIKE :pattern
                ORDER BY canonical_name
                LIMIT 50
                """),
                {"pattern": f"%{name_pattern}%"},
            )
            rows = result.fetchall()
            return [
                {
                    "entity_id": str(r[0]),
                    "canonical_name": r[1],
                    "type": r[2],
                    "aliases": r[3],
                    "created_at": r[4].isoformat() if r[4] else None,
                }
                for r in rows
            ]

    async def get_entities_by_ids(self, entity_ids: list[str]) -> dict[str, dict]:
        """Batch-lookup entities by UUID.  Returns ``{entity_id: {canonical_name, type}}``."""
        if not entity_ids:
            return {}
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT entity_id, canonical_name, type
                FROM entities
                WHERE entity_id = ANY(:ids)
                """),
                {"ids": entity_ids},
            )
            rows = result.fetchall()
            return {
                str(r[0]): {"canonical_name": r[1], "type": r[2]}
                for r in rows
            }

    # ------------------------------------------------------------------
    # Graph: mentions
    # ------------------------------------------------------------------

    async def insert_mention(
        self,
        entity_id: str,
        page_id: str,
        span: Optional[dict] = None,
        quote: Optional[str] = None,
    ) -> str:
        """Record that an entity was mentioned on a page. Returns mention_id."""
        mention_id = str(uuid.uuid4())
        span_json = json.dumps(span) if span else None
        async with self._engine.connect() as conn:
            # One mention per (entity, page): a second extraction of the
            # same page — two containers, or a re-run — reads the existing
            # row back instead of doubling it (migration 103).
            result = await conn.execute(
                text("""
                INSERT INTO mentions (mention_id, entity_id, page_id, span_json, quote)
                VALUES (:mention_id, :entity_id, :page_id, CAST(:span_json AS jsonb), :quote)
                ON CONFLICT (entity_id, page_id) DO UPDATE SET entity_id = mentions.entity_id
                RETURNING mention_id
                """),
                {"mention_id": mention_id, "entity_id": entity_id,
                 "page_id": page_id, "span_json": span_json, "quote": quote},
            )
            row = result.fetchone()
            await conn.commit()
        return str(row[0]) if row else mention_id

    # ------------------------------------------------------------------
    # Graph: extraction bookkeeping (migration 103)
    # ------------------------------------------------------------------

    async def claim_page_for_extraction(
        self, page_id: str, stale_after_minutes: int = 15
    ) -> bool:
        """Atomically claim a page for extraction.

        False when the page is already extracted or another extractor holds
        a live claim on it. A claim older than ``stale_after_minutes`` is
        taken over — the worker that held it is gone.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                UPDATE pages SET extraction_claimed_at = NOW()
                WHERE page_id = :page_id
                  AND extracted_at IS NULL
                  AND NOT EXISTS (SELECT 1 FROM docs d
                                  WHERE d.doc_id = pages.doc_id
                                    AND d.held_at IS NOT NULL)
                  AND (extraction_claimed_at IS NULL
                       OR extraction_claimed_at < NOW() - make_interval(mins => :stale))
                RETURNING page_id
                """),
                {"page_id": page_id, "stale": stale_after_minutes},
            )
            row = result.fetchone()
            await conn.commit()
            return row is not None

    async def mark_page_extracted(self, page_id: str) -> None:
        """Extraction completed for this page — with or without results."""
        async with self._engine.connect() as conn:
            await conn.execute(
                text("""
                UPDATE pages SET extracted_at = NOW(), extraction_claimed_at = NULL
                WHERE page_id = :page_id
                """),
                {"page_id": page_id},
            )
            await conn.commit()

    async def release_extraction_claim(self, page_id: str) -> None:
        """Extraction did NOT complete (the model was unreachable): hand the
        page back so the next cycle retries it."""
        async with self._engine.connect() as conn:
            await conn.execute(
                text("""
                UPDATE pages SET extraction_claimed_at = NULL
                WHERE page_id = :page_id AND extracted_at IS NULL
                """),
                {"page_id": page_id},
            )
            await conn.commit()

    # ------------------------------------------------------------------
    # Graph: edges
    # ------------------------------------------------------------------

    async def insert_edge(
        self,
        src_entity_id: str,
        rel_type: str,
        dst_entity_id: str,
        evidence: Optional[dict] = None,
        reinforce_factor: float = 0.2,
    ) -> str:
        """Insert a directed relationship edge, or REINFORCE an existing one.

        Reinforcement (edge metabolism):
        if the same ``(src, rel, dst)`` triple already exists, we don't insert a
        duplicate — we corroborate the existing edge. Its confidence moves a
        fraction ``reinforce_factor`` of the remaining distance toward 1.0
        (diminishing returns), ``created_at`` is refreshed so the age-gated
        prune restarts its clock, and the new evidence is appended. This makes
        edge survival Darwinian: relationships that keep being re-observed
        strengthen; one-off noise fades and is snipped by the prune pass.

        Returns the edge_id (existing on reinforcement, new otherwise).
        """
        evidence_json = json.dumps(evidence) if evidence else None
        async with self._engine.connect() as conn:
            # Reinforce an existing identical triple if present (newest first).
            existing = await conn.execute(
                text("""
                    SELECT edge_id::text, confidence FROM edges
                    WHERE src_entity_id = :src AND rel_type = :rel
                      AND dst_entity_id = :dst
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"src": src_entity_id, "rel": rel_type, "dst": dst_entity_id},
            )
            row = existing.first()
            if row is not None:
                edge_id, conf = row[0], float(row[1] if row[1] is not None else 1.0)
                # Move a fraction of the way to 1.0; corroboration only
                # strengthens, never weakens, an edge.
                new_conf = min(1.0, conf + (1.0 - conf) * max(0.0, min(1.0, reinforce_factor)))
                await conn.execute(
                    text("""
                        UPDATE edges
                        SET confidence = :conf,
                            created_at = NOW(),
                            evidence_json = CASE
                                WHEN CAST(:ev AS text) IS NULL THEN evidence_json
                                ELSE COALESCE(evidence_json, '{}'::jsonb)
                                     || jsonb_build_object(
                                         'reinforced_at', to_jsonb(NOW()),
                                         'last_evidence', CAST(:ev AS jsonb))
                            END
                        WHERE edge_id = :edge_id
                    """),
                    {"conf": new_conf, "ev": evidence_json, "edge_id": edge_id},
                )
                await conn.commit()
                return edge_id

            edge_id = str(uuid.uuid4())
            # The existence check above is not atomic across two builders;
            # the unique index (migration 103) is, and a loser reads the
            # winner's edge back.
            result = await conn.execute(
                text("""
                INSERT INTO edges (edge_id, src_entity_id, rel_type, dst_entity_id, evidence_json)
                VALUES (:edge_id, :src_entity_id, :rel_type, :dst_entity_id, CAST(:evidence_json AS jsonb))
                ON CONFLICT (src_entity_id, rel_type, dst_entity_id)
                    DO UPDATE SET rel_type = edges.rel_type
                RETURNING edge_id
                """),
                {"edge_id": edge_id, "src_entity_id": src_entity_id,
                 "rel_type": rel_type, "dst_entity_id": dst_entity_id,
                 "evidence_json": evidence_json},
            )
            row = result.fetchone()
            await conn.commit()
        return str(row[0]) if row else edge_id

    async def get_entity_neighbors(
        self,
        entity_id: str,
        rel_types: Optional[list[str]] = None,
        hops: int = 1,
    ) -> list[dict]:
        """Multi-hop graph traversal from an entity."""
        if hops < 1:
            return []

        # Build the recursive CTE
        if rel_types:
            query = """
                WITH RECURSIVE neighbors AS (
                    SELECT e.edge_id, e.src_entity_id, e.rel_type,
                           e.dst_entity_id, e.evidence_json,
                           1 AS depth,
                           ARRAY[e.src_entity_id, e.dst_entity_id] AS visited
                    FROM edges e
                    WHERE (e.src_entity_id = :eid OR e.dst_entity_id = :eid2)
                      AND e.rel_type = ANY(:rel_types)

                    UNION ALL

                    SELECT e.edge_id, e.src_entity_id, e.rel_type,
                           e.dst_entity_id, e.evidence_json,
                           n.depth + 1,
                           n.visited || CASE
                               WHEN e.src_entity_id = ANY(n.visited)
                               THEN ARRAY[e.dst_entity_id]
                               ELSE ARRAY[e.src_entity_id]
                           END
                    FROM edges e
                    JOIN neighbors n ON (
                        (e.src_entity_id = n.dst_entity_id AND NOT e.dst_entity_id = ANY(n.visited))
                        OR
                        (e.dst_entity_id = n.src_entity_id AND NOT e.src_entity_id = ANY(n.visited))
                    )
                    WHERE n.depth < :hops
                      AND e.rel_type = ANY(:rel_types2)
                )
                SELECT DISTINCT edge_id, src_entity_id, rel_type, dst_entity_id,
                       evidence_json, depth
                FROM neighbors
                ORDER BY depth, edge_id
            """
            params: dict[str, Any] = {
                "eid": entity_id, "eid2": entity_id,
                "rel_types": rel_types, "hops": hops, "rel_types2": rel_types,
            }
        else:
            query = """
                WITH RECURSIVE neighbors AS (
                    SELECT e.edge_id, e.src_entity_id, e.rel_type,
                           e.dst_entity_id, e.evidence_json,
                           1 AS depth,
                           ARRAY[e.src_entity_id, e.dst_entity_id] AS visited
                    FROM edges e
                    WHERE (e.src_entity_id = :eid OR e.dst_entity_id = :eid2)

                    UNION ALL

                    SELECT e.edge_id, e.src_entity_id, e.rel_type,
                           e.dst_entity_id, e.evidence_json,
                           n.depth + 1,
                           n.visited || CASE
                               WHEN e.src_entity_id = ANY(n.visited)
                               THEN ARRAY[e.dst_entity_id]
                               ELSE ARRAY[e.src_entity_id]
                           END
                    FROM edges e
                    JOIN neighbors n ON (
                        (e.src_entity_id = n.dst_entity_id AND NOT e.dst_entity_id = ANY(n.visited))
                        OR
                        (e.dst_entity_id = n.src_entity_id AND NOT e.src_entity_id = ANY(n.visited))
                    )
                    WHERE n.depth < :hops
                )
                SELECT DISTINCT edge_id, src_entity_id, rel_type, dst_entity_id,
                       evidence_json, depth
                FROM neighbors
                ORDER BY depth, edge_id
            """
            params = {"eid": entity_id, "eid2": entity_id, "hops": hops}

        async with self._engine.connect() as conn:
            result = await conn.execute(text(query), params)
            rows = result.fetchall()
            return [
                {
                    "edge_id": str(r[0]),
                    "src_entity_id": str(r[1]),
                    "rel_type": r[2],
                    "dst_entity_id": str(r[3]),
                    "evidence": r[4],
                    "depth": r[5],
                }
                for r in rows
            ]

    async def get_neighbors_batch(self, entity_ids: list[str]) -> list[dict]:
        """Get all edges connected to *any* of the given entities in one query."""
        if not entity_ids:
            return []
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT DISTINCT e.edge_id, e.src_entity_id, e.rel_type,
                       e.dst_entity_id, e.evidence_json,
                       COALESCE(e.confidence, 1.0) AS confidence
                FROM edges e
                WHERE e.src_entity_id = ANY(:ids)
                   OR e.dst_entity_id = ANY(:ids2)
                """),
                {"ids": entity_ids, "ids2": entity_ids},
            )
            rows = result.fetchall()
            return [
                {
                    "edge_id": str(r[0]),
                    "src_entity_id": str(r[1]),
                    "rel_type": r[2],
                    "dst_entity_id": str(r[3]),
                    "evidence": r[4],
                    "confidence": float(r[5]),
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Listing / export methods (used by heartbeat actions)
    # ------------------------------------------------------------------

    async def list_docs(self, limit: int = 200) -> list[dict]:
        """List all documents with their source URIs and hashes."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT doc_id, source_uri, title, hash, created_at
                FROM docs
                ORDER BY created_at DESC
                LIMIT :limit
                """),
                {"limit": limit},
            )
            rows = result.fetchall()
            return [
                {
                    "doc_id": str(r[0]),
                    "source_uri": r[1],
                    "title": r[2],
                    "hash": r[3],
                    "created_at": r[4].isoformat() if r[4] else None,
                }
                for r in rows
            ]

    async def list_entities(self, limit: int = 500) -> list[dict]:
        """List all entities for graph operations."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT entity_id, canonical_name, type, aliases_json, created_at
                FROM entities
                ORDER BY canonical_name
                LIMIT :limit
                """),
                {"limit": limit},
            )
            rows = result.fetchall()
            return [
                {
                    "entity_id": str(r[0]),
                    "canonical_name": r[1],
                    "type": r[2],
                    "aliases": r[3],
                    "created_at": r[4].isoformat() if r[4] else None,
                }
                for r in rows
            ]

    async def merge_entities(self, keep_id: str, merge_id: str) -> bool:
        """Merge merge_id entity into keep_id: reassign mentions/edges, delete merge_id."""
        async with self._engine.connect() as conn:
            # Reassign mentions
            await conn.execute(
                text("UPDATE mentions SET entity_id = :keep WHERE entity_id = :merge"),
                {"keep": keep_id, "merge": merge_id},
            )
            # Reassign edges (src)
            await conn.execute(
                text("UPDATE edges SET src_entity_id = :keep WHERE src_entity_id = :merge"),
                {"keep": keep_id, "merge": merge_id},
            )
            # Reassign edges (dst)
            await conn.execute(
                text("UPDATE edges SET dst_entity_id = :keep WHERE dst_entity_id = :merge"),
                {"keep": keep_id, "merge": merge_id},
            )
            # Delete the merged entity
            await conn.execute(
                text("DELETE FROM entities WHERE entity_id = :merge"),
                {"merge": merge_id},
            )
            await conn.commit()
        return True

    async def export_all_docs(self) -> list[dict]:
        """Export all documents for backup."""
        async with self._engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT doc_id, source_uri, title, hash, created_at FROM docs ORDER BY created_at"
            ))
            return [
                {
                    "doc_id": str(r[0]), "source_uri": r[1], "title": r[2],
                    "hash": r[3], "created_at": r[4].isoformat() if r[4] else None,
                }
                for r in result.fetchall()
            ]

    async def export_all_entities(self) -> list[dict]:
        """Export all entities for backup."""
        async with self._engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT entity_id, canonical_name, type, aliases_json, created_at FROM entities ORDER BY canonical_name"
            ))
            return [
                {
                    "entity_id": str(r[0]), "canonical_name": r[1], "type": r[2],
                    "aliases": r[3], "created_at": r[4].isoformat() if r[4] else None,
                }
                for r in result.fetchall()
            ]

    async def export_all_edges(self) -> list[dict]:
        """Export all edges for backup."""
        async with self._engine.connect() as conn:
            result = await conn.execute(text(
                "SELECT edge_id, src_entity_id, rel_type, dst_entity_id, evidence_json, created_at "
                "FROM edges ORDER BY created_at"
            ))
            return [
                {
                    "edge_id": str(r[0]), "src_entity_id": str(r[1]),
                    "rel_type": r[2], "dst_entity_id": str(r[3]),
                    "evidence": r[4], "created_at": r[5].isoformat() if r[5] else None,
                }
                for r in result.fetchall()
            ]

    async def get_entity_pages(self, entity_id: str) -> list[str]:
        """Get all page_ids where an entity is mentioned."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT DISTINCT m.page_id
                FROM mentions m
                JOIN pages p ON p.page_id = m.page_id
                JOIN docs d ON d.doc_id = p.doc_id
                WHERE m.entity_id = :entity_id AND d.held_at IS NULL
                """),
                {"entity_id": entity_id},
            )
            rows = result.fetchall()
            return [str(r[0]) for r in rows]

    async def get_entity_pages_batch(self, entity_ids: list[str]) -> list[str]:
        """Get all page_ids where *any* of the given entities are mentioned."""
        if not entity_ids:
            return []
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT DISTINCT m.page_id
                FROM mentions m
                JOIN pages p ON p.page_id = m.page_id
                JOIN docs d ON d.doc_id = p.doc_id
                WHERE m.entity_id = ANY(:ids) AND d.held_at IS NULL
                """),
                {"ids": entity_ids},
            )
            return [str(r[0]) for r in result.fetchall()]

    # ------------------------------------------------------------------
    # Semantic edge classification
    # ------------------------------------------------------------------

    async def update_edge_classification(
        self,
        edge_id: str,
        rel_type: str,
        confidence: float,
        classifier_model: str,
    ) -> None:
        """Classify an edge with a semantic relationship type and confidence."""
        async with self._engine.connect() as conn:
            await conn.execute(
                text("""
                UPDATE edges
                SET rel_type = :rel_type,
                    confidence = :confidence,
                    classified_at = NOW(),
                    classifier_model = :classifier_model
                WHERE edge_id = :edge_id
                """),
                {
                    "edge_id": edge_id,
                    "rel_type": rel_type,
                    "confidence": confidence,
                    "classifier_model": classifier_model,
                },
            )
            await conn.commit()

    async def get_unclassified_edges(self, limit: int = 50) -> list[dict]:
        """Return edges that have not yet been semantically classified."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT edge_id, src_entity_id, dst_entity_id, rel_type, evidence_json
                FROM edges
                WHERE classified_at IS NULL
                ORDER BY created_at
                LIMIT :limit
                """),
                {"limit": limit},
            )
            rows = result.fetchall()
            return [
                {
                    "edge_id": str(r[0]),
                    "src_entity_id": str(r[1]),
                    "dst_entity_id": str(r[2]),
                    "rel_type": r[3],
                    "evidence_json": r[4],
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Entity embeddings (graph-aware)
    # ------------------------------------------------------------------

    async def upsert_entity_embedding(
        self,
        entity_id: str,
        embedding: list[float],
        generation: int,
    ) -> None:
        """Insert or update a graph-aware entity embedding.

        The signature is the ``GraphStore`` protocol's, so the model cannot be
        a parameter: it is the model the embed response named, carried on
        ``embedding`` (:mod:`trellion.embedding_space`).
        """
        vec = self._vec(embedding)
        prov = _provenance(embedding)
        async with self._engine.connect() as conn:
            await conn.execute(
                text("""
                INSERT INTO entity_embeddings
                    (entity_id, embedding, computed_at, generation,
                     embedding_model, embedding_dim)
                VALUES (:entity_id, CAST(:vec AS vector), NOW(), :generation,
                        :model, :dim)
                ON CONFLICT (entity_id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        computed_at = EXCLUDED.computed_at,
                        generation = EXCLUDED.generation,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dim = EXCLUDED.embedding_dim,
                        embedding_next = NULL,
                        embedding_next_model = NULL
                """),
                {"entity_id": entity_id, "vec": vec, "generation": generation, **prov},
            )
            await conn.commit()

    async def search_entity_embeddings(
        self,
        embedding: list[float],
        top_k: int = 10,
    ) -> list[dict]:
        """Vector similarity search over entity embeddings.

        Returns entity_ids and cosine similarity scores, ordered by relevance.
        """
        vec = self._vec(embedding)
        model_sql, model_params = await embedding_space.model_clause(
            "ee.embedding_model", query_vector=embedding
        )
        where_sql = f"WHERE {model_sql}" if model_sql else ""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"""
                SELECT ee.entity_id,
                       1 - (ee.embedding <=> CAST(:vec AS vector)) AS score,
                       e.canonical_name,
                       e.type
                FROM entity_embeddings ee
                JOIN entities e ON e.entity_id = ee.entity_id
                {where_sql}
                ORDER BY ee.embedding <=> CAST(:vec2 AS vector)
                LIMIT :top_k
                """),
                {"vec": vec, "vec2": vec, "top_k": top_k, **model_params},
            )
            rows = result.fetchall()
            return [
                {
                    "entity_id": str(r[0]),
                    "score": float(r[1]),
                    "canonical_name": r[2],
                    "type": r[3],
                }
                for r in rows
            ]

    async def get_stale_entity_embeddings(
        self,
        current_generation: int,
        limit: int = 100,
    ) -> list[dict]:
        """Return entities whose embeddings are stale or missing.

        An entity is stale if its embedding generation is below
        *current_generation* or if it has no embedding row at all.
        """
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT e.entity_id, e.canonical_name, e.type,
                       ee.generation AS current_gen
                FROM entities e
                LEFT JOIN entity_embeddings ee ON ee.entity_id = e.entity_id
                WHERE ee.entity_id IS NULL
                   OR ee.generation < :current_generation
                ORDER BY e.entity_id
                LIMIT :limit
                """),
                {"current_generation": current_generation, "limit": limit},
            )
            rows = result.fetchall()
            return [
                {
                    "entity_id": str(r[0]),
                    "canonical_name": r[1],
                    "type": r[2],
                    "current_generation": r[3],
                }
                for r in rows
            ]

    # ------------------------------------------------------------------
    # Entity clusters
    # ------------------------------------------------------------------

    async def upsert_cluster(
        self,
        cluster_id: str,
        name: str,
        level: int,
        parent_id: Optional[str],
        summary: Optional[str],
        centroid: Optional[list[float]],
        member_count: int,
        generation: int,
    ) -> None:
        """Insert or update a hierarchical entity cluster.

        A centroid is computed from entity vectors, so it has no embed
        response of its own. A caller that knows the model of the vectors it
        averaged passes the centroid as an
        :class:`~trellion.embedding_space.Vector` carrying it; a bare list is
        recorded under the space's current model (the entity vectors it
        averaged are in that space).
        """
        centroid_val = self._vec(centroid) if centroid else None
        centroid_model: Optional[str] = None
        if centroid:
            centroid_model = await embedding_space.write_model(centroid)
        async with self._engine.connect() as conn:
            await conn.execute(
                text("""
                INSERT INTO entity_clusters
                    (cluster_id, name, level, parent_cluster_id, summary,
                     centroid, member_count, computed_at, generation,
                     embedding_model, embedding_dim)
                VALUES
                    (:cluster_id, :name, :level, :parent_id, :summary,
                     CAST(:centroid AS vector), :member_count, NOW(), :generation,
                     :model, :dim)
                ON CONFLICT (cluster_id) DO UPDATE
                    SET name = EXCLUDED.name,
                        level = EXCLUDED.level,
                        parent_cluster_id = EXCLUDED.parent_cluster_id,
                        summary = EXCLUDED.summary,
                        centroid = EXCLUDED.centroid,
                        member_count = EXCLUDED.member_count,
                        computed_at = EXCLUDED.computed_at,
                        generation = EXCLUDED.generation,
                        embedding_model = EXCLUDED.embedding_model,
                        embedding_dim = EXCLUDED.embedding_dim,
                        embedding_next = NULL,
                        embedding_next_model = NULL
                """),
                {
                    "cluster_id": cluster_id,
                    "name": name,
                    "level": level,
                    "parent_id": parent_id,
                    "summary": summary,
                    "centroid": centroid_val,
                    "member_count": member_count,
                    "generation": generation,
                    "model": centroid_model,
                    "dim": len(centroid) if centroid else None,
                },
            )
            await conn.commit()

    async def set_cluster_members(
        self,
        cluster_id: str,
        members: list[dict],
    ) -> None:
        """Replace all members of a cluster.

        Each dict in *members* must have ``entity_id`` and optionally
        ``membership_score`` (defaults to 1.0).
        """
        async with self._engine.connect() as conn:
            # Remove existing members
            await conn.execute(
                text("DELETE FROM entity_cluster_members WHERE cluster_id = :cluster_id"),
                {"cluster_id": cluster_id},
            )
            # Insert new members
            for m in members:
                await conn.execute(
                    text("""
                    INSERT INTO entity_cluster_members (entity_id, cluster_id, membership_score)
                    VALUES (:entity_id, :cluster_id, :score)
                    """),
                    {
                        "entity_id": m["entity_id"],
                        "cluster_id": cluster_id,
                        "score": m.get("membership_score", 1.0),
                    },
                )
            await conn.commit()

    async def get_entity_clusters(self, entity_id: str) -> list[dict]:
        """Return all clusters that an entity belongs to."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT c.cluster_id, c.name, c.level, c.parent_cluster_id,
                       c.summary, c.member_count, c.generation,
                       ecm.membership_score
                FROM entity_cluster_members ecm
                JOIN entity_clusters c ON c.cluster_id = ecm.cluster_id
                WHERE ecm.entity_id = :entity_id
                ORDER BY c.level, c.name
                """),
                {"entity_id": entity_id},
            )
            rows = result.fetchall()
            return [
                {
                    "cluster_id": str(r[0]),
                    "name": r[1],
                    "level": r[2],
                    "parent_cluster_id": str(r[3]) if r[3] else None,
                    "summary": r[4],
                    "member_count": r[5],
                    "generation": r[6],
                    "membership_score": float(r[7]),
                }
                for r in rows
            ]

    async def search_clusters(
        self,
        embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """Vector similarity search over cluster centroids."""
        vec = self._vec(embedding)
        model_sql, model_params = await embedding_space.model_clause(
            "c.embedding_model", query_vector=embedding
        )
        model_and = f"AND {model_sql}" if model_sql else ""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"""
                SELECT c.cluster_id, c.name, c.level, c.parent_cluster_id,
                       c.summary, c.member_count, c.generation,
                       1 - (c.centroid <=> CAST(:vec AS vector)) AS score
                FROM entity_clusters c
                WHERE c.centroid IS NOT NULL
                {model_and}
                ORDER BY c.centroid <=> CAST(:vec2 AS vector)
                LIMIT :top_k
                """),
                {"vec": vec, "vec2": vec, "top_k": top_k, **model_params},
            )
            rows = result.fetchall()
            return [
                {
                    "cluster_id": str(r[0]),
                    "name": r[1],
                    "level": r[2],
                    "parent_cluster_id": str(r[3]) if r[3] else None,
                    "summary": r[4],
                    "member_count": r[5],
                    "generation": r[6],
                    "score": float(r[7]),
                }
                for r in rows
            ]

    async def list_clusters(
        self,
        level: Optional[int] = None,
        parent_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict]:
        """List entity clusters with optional level and parent filters."""
        conditions: list[str] = []
        params: dict[str, Any] = {"limit": limit}

        if level is not None:
            conditions.append("c.level = :level")
            params["level"] = level
        if parent_id is not None:
            conditions.append("c.parent_cluster_id = :parent_id")
            params["parent_id"] = parent_id

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        async with self._engine.connect() as conn:
            result = await conn.execute(
                text(f"""
                SELECT c.cluster_id, c.name, c.level, c.parent_cluster_id,
                       c.summary, c.member_count, c.generation, c.computed_at
                FROM entity_clusters c
                {where}
                ORDER BY c.level, c.name
                LIMIT :limit
                """),
                params,
            )
            rows = result.fetchall()
            return [
                {
                    "cluster_id": str(r[0]),
                    "name": r[1],
                    "level": r[2],
                    "parent_cluster_id": str(r[3]) if r[3] else None,
                    "summary": r[4],
                    "member_count": r[5],
                    "generation": r[6],
                    "computed_at": r[7].isoformat() if r[7] else None,
                }
                for r in rows
            ]

    async def get_cluster_members(
        self,
        cluster_id: str,
        limit: int = 200,
    ) -> list[dict]:
        """Return entity info for all members of a cluster."""
        async with self._engine.connect() as conn:
            result = await conn.execute(
                text("""
                SELECT e.entity_id, e.canonical_name, e.type, e.aliases_json,
                       ecm.membership_score
                FROM entity_cluster_members ecm
                JOIN entities e ON e.entity_id = ecm.entity_id
                WHERE ecm.cluster_id = :cluster_id
                ORDER BY ecm.membership_score DESC
                LIMIT :limit
                """),
                {"cluster_id": cluster_id, "limit": limit},
            )
            rows = result.fetchall()
            return [
                {
                    "entity_id": str(r[0]),
                    "canonical_name": r[1],
                    "type": r[2],
                    "aliases": r[3],
                    "membership_score": float(r[4]),
                }
                for r in rows
            ]

    async def delete_clusters(
        self,
        generation_below: Optional[int] = None,
    ) -> int:
        """Delete clusters, optionally filtering by generation.

        When *generation_below* is provided, only clusters with
        ``generation < generation_below`` are removed (stale cleanup).
        Otherwise all clusters are deleted.

        Returns the number of clusters removed.
        """
        async with self._engine.connect() as conn:
            if generation_below is not None:
                result = await conn.execute(
                    text("""
                    DELETE FROM entity_clusters
                    WHERE generation < :gen
                    """),
                    {"gen": generation_below},
                )
            else:
                result = await conn.execute(
                    text("DELETE FROM entity_clusters"),
                )
            removed = result.rowcount
            await conn.commit()
            return removed
