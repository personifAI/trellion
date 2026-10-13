# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""PostgreSQL + pgvector store for semantic memory.

Uses the shared SQLAlchemy engine from ``noomesh.db`` instead of
creating its own psycopg connection.  Schema is managed by Alembic.

Supports:
- Storing conversations with or without embeddings
- Semantic search (when embeddings available)
- Text search fallback (when embeddings not available)
- Batch embedding of unindexed records
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .store import MemoryRecord


def _get_embedding_dim() -> int:
    """Return the expected embedding dimension from config (preferred) or env."""
    try:
        from ..config import get_config
        return get_config().services.embedding_dim
    except Exception:
        return int(os.environ.get("EMBEDDING_DIM", "768"))


@dataclass
class PGVectorConfig:
    dsn: str = ""
    table: str = "noo_conversations"
    dim: int = 768  # overridden at runtime; see _get_embedding_dim()
    create_extension: bool = True
    create_index: bool = True

    def __post_init__(self) -> None:
        if self.dim == 768:  # only override if caller didn't set explicitly
            self.dim = _get_embedding_dim()


class PGVectorStore:
    """PostgreSQL store with optional vector embeddings.

    Can store conversations without embeddings and add them later.
    Falls back to text search when embeddings aren't available.
    """

    def __init__(
        self,
        config: PGVectorConfig,
        engine: Optional[Engine] = None,
    ) -> None:
        self.config = config
        if engine is not None:
            self._engine = engine
        elif config.dsn:
            from trellion.db.engine import _sync_dsn
            from sqlalchemy import create_engine
            self._engine = create_engine(_sync_dsn(config.dsn), pool_pre_ping=True)
        else:
            from trellion.db.engine import get_engine
            self._engine = get_engine()

    def _format_vector(self, values: List[float]) -> str:
        """Format a float list as a pgvector literal, validating dimension."""
        expected = self.config.dim
        if len(values) != expected:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(values)}, "
                f"expected {expected} (services.embedding_dim). "
                f"Check your embedding model or set EMBEDDING_DIM={len(values)}."
            )
        return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"

    def add(
        self,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "global",
        persona_id: Optional[str] = None,
    ) -> MemoryRecord:
        """Add a record, optionally with embedding.

        Args:
            role: Message role (user/assistant/system)
            content: Message content
            embedding: Optional vector embedding (can add later)
            user_id: Optional user identifier
            session_id: Optional session identifier
            scope: Memory scope (default: global)

        Returns:
            Created MemoryRecord
        """
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            scope=scope,
            role=role,
            content=content,
            created_at=time.time(),
        )

        with self._engine.connect() as conn:
            if embedding:
                vector = self._format_vector(embedding)
                conn.execute(
                    text(f"""
                    INSERT INTO {self.config.table}
                    (id, scope, user_id, session_id, persona_id, role, content, embedding, created_at, indexed_at)
                    VALUES (:id, :scope, :user_id, :session_id, :persona_id, :role, :content, :vec::vector, to_timestamp(:ts), NOW())
                    """),
                    {
                        "id": record.id, "scope": record.scope,
                        "user_id": user_id, "session_id": session_id,
                        "persona_id": persona_id,
                        "role": record.role, "content": record.content,
                        "vec": vector, "ts": record.created_at,
                    },
                )
            else:
                conn.execute(
                    text(f"""
                    INSERT INTO {self.config.table}
                    (id, scope, user_id, session_id, persona_id, role, content, created_at)
                    VALUES (:id, :scope, :user_id, :session_id, :persona_id, :role, :content, to_timestamp(:ts))
                    """),
                    {
                        "id": record.id, "scope": record.scope,
                        "user_id": user_id, "session_id": session_id,
                        "persona_id": persona_id,
                        "role": record.role, "content": record.content,
                        "ts": record.created_at,
                    },
                )
            conn.commit()
        return record

    def search(
        self,
        embedding: List[float],
        top_k: int = 5,
        user_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Semantic search using vector similarity.

        Only searches records that have embeddings.
        When ``persona_id`` is provided, results include persona-scoped
        records plus shared (persona_id IS NULL).
        """
        vector = self._format_vector(embedding)

        where_parts = ["embedding IS NOT NULL"]
        params = {"vec": vector, "vec2": vector, "top_k": top_k}

        if user_id:
            where_parts.append("(user_id = :user_id OR user_id IS NULL)")
            params["user_id"] = user_id

        if persona_id is not None:
            where_parts.append("(persona_id = :persona_id OR persona_id IS NULL)")
            params["persona_id"] = persona_id

        where_sql = " AND ".join(where_parts)

        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                SELECT id, scope, role, content, EXTRACT(EPOCH FROM created_at) as created_at,
                       1 - (embedding <=> :vec::vector) as score
                FROM {self.config.table}
                WHERE {where_sql}
                ORDER BY embedding <=> :vec2::vector
                LIMIT :top_k
                """),
                params,
            ).fetchall()

        results: List[Tuple[MemoryRecord, float]] = []
        for row in rows:
            record = MemoryRecord(
                id=str(row[0]),
                scope=row[1],
                role=row[2],
                content=row[3],
                created_at=float(row[4]),
            )
            score = float(row[5])
            results.append((record, score))
        return results

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Text search using trigram similarity.

        Fallback when embeddings aren't available.
        """
        where_parts = ["content %% :query2"]
        params = {"query": query, "query2": query, "query3": query, "top_k": top_k}

        if user_id:
            where_parts.append("(user_id = :user_id OR user_id IS NULL)")
            params["user_id"] = user_id

        if persona_id is not None:
            where_parts.append("(persona_id = :persona_id OR persona_id IS NULL)")
            params["persona_id"] = persona_id

        where_sql = " AND ".join(where_parts)

        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                SELECT id, scope, role, content, EXTRACT(EPOCH FROM created_at) as created_at,
                       similarity(content, :query) as score
                FROM {self.config.table}
                WHERE {where_sql}
                ORDER BY similarity(content, :query3) DESC
                LIMIT :top_k
                """),
                params,
            ).fetchall()

        results: List[Tuple[MemoryRecord, float]] = []
        for row in rows:
            record = MemoryRecord(
                id=str(row[0]),
                scope=row[1],
                role=row[2],
                content=row[3],
                created_at=float(row[4]),
            )
            score = float(row[5])
            results.append((record, score))
        return results

    def get_unembedded(self, limit: int = 100) -> List[Tuple[str, str]]:
        """Get records that don't have embeddings yet.

        Returns:
            List of (id, content) tuples for batch embedding
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                SELECT id, content
                FROM {self.config.table}
                WHERE embedding IS NULL
                ORDER BY created_at ASC
                LIMIT :limit
                """),
                {"limit": limit},
            ).fetchall()
            return [(str(row[0]), row[1]) for row in rows]

    def update_embedding(self, record_id: str, embedding: List[float]) -> bool:
        """Add embedding to an existing record.

        Args:
            record_id: UUID of record to update
            embedding: Vector embedding to add

        Returns:
            True if updated, False if not found
        """
        vector = self._format_vector(embedding)

        with self._engine.connect() as conn:
            result = conn.execute(
                text(f"""
                UPDATE {self.config.table}
                SET embedding = :vec::vector, indexed_at = NOW()
                WHERE id = :id
                """),
                {"vec": vector, "id": record_id},
            )
            conn.commit()
            return result.rowcount > 0

    def batch_update_embeddings(
        self,
        updates: List[Tuple[str, List[float]]]
    ) -> int:
        """Batch update embeddings for multiple records.

        Args:
            updates: List of (record_id, embedding) tuples

        Returns:
            Number of records updated
        """
        if not updates:
            return 0

        count = 0
        with self._engine.connect() as conn:
            for record_id, embedding in updates:
                vector = self._format_vector(embedding)
                result = conn.execute(
                    text(f"""
                    UPDATE {self.config.table}
                    SET embedding = :vec::vector, indexed_at = NOW()
                    WHERE id = :id
                    """),
                    {"vec": vector, "id": record_id},
                )
                count += result.rowcount
            conn.commit()
        return count

    def get_user_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[MemoryRecord]:
        """Get recent conversation history for a user."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(f"""
                SELECT id, scope, role, content, EXTRACT(EPOCH FROM created_at) as created_at
                FROM {self.config.table}
                WHERE user_id = :user_id
                ORDER BY created_at DESC
                LIMIT :limit
                """),
                {"user_id": user_id, "limit": limit},
            ).fetchall()

        return [
            MemoryRecord(
                id=str(row[0]),
                scope=row[1],
                role=row[2],
                content=row[3],
                created_at=float(row[4]),
            )
            for row in reversed(rows)  # Return in chronological order
        ]

    def get_stats(self) -> dict:
        """Get storage statistics."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text(f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(embedding) as embedded,
                    COUNT(*) - COUNT(embedding) as unembedded,
                    COUNT(DISTINCT user_id) as unique_users,
                    MIN(created_at) as oldest,
                    MAX(created_at) as newest
                FROM {self.config.table}
                """)
            ).fetchone()

            return {
                "total_records": row[0],
                "embedded_records": row[1],
                "unembedded_records": row[2],
                "unique_users": row[3],
                "oldest_record": row[4].isoformat() if row[4] else None,
                "newest_record": row[5].isoformat() if row[5] else None,
            }


