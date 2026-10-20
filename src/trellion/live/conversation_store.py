# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Enhanced conversation storage with full interaction tracking.

Uses SQLAlchemy Core expressions against ORM-defined tables from
``noomesh.db.models``.  Schema is managed by Alembic.

Stores:
- Messages (user, assistant, system)
- Tool calls and results
- Reasoning traces
- Retrieved context
- Performance metrics
- Feedback signals
"""

from __future__ import annotations

from .. import json_fast as json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy import (
    and_,
    cast,
    case,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert, JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from .. import embedding_space
from ..db.models import (
    PaiConversation,
    PaiEntry,
    PaiToolExecution,
    PaiContextRetrieval,
    PaiConversationFeedback,
)

logger = structlog.get_logger()

# Table references for Core expressions
_conv = PaiConversation.__table__
_entry = PaiEntry.__table__
_tool = PaiToolExecution.__table__
_ctx = PaiContextRetrieval.__table__
_fb = PaiConversationFeedback.__table__


def _get_embedding_dim() -> int:
    """Return the expected embedding dimension from config (preferred) or env.

    Used as the constructor-time fallback when ``ConversationStore`` is built
    without an explicit ``embedding_dim``. Hot-path call sites use the
    ``self._format_vector`` method which reads the resolved dimension off the
    instance.
    """
    try:
        from ..config import get_config
        return get_config().services.embedding_dim
    except Exception:
        return int(os.environ.get("EMBEDDING_DIM", "768"))


class MessageType(str, Enum):
    """Types of conversation entries."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    CONTEXT_RETRIEVAL = "context_retrieval"
    FEEDBACK = "feedback"
    METADATA = "metadata"


@dataclass
class ConversationEntry:
    """A single entry in a conversation."""
    id: str
    conversation_id: str
    type: MessageType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None  # Links tool results to tool calls
    created_at: float = field(default_factory=time.time)


class ConversationStore:
    """PostgreSQL store for rich conversation data (async)."""

    def __init__(
        self,
        async_engine: Optional[AsyncEngine] = None,
        dsn: Optional[str] = None,
        table_prefix: str = "pai",
        embedding_dim: Optional[int] = None,
    ):
        self.table_prefix = table_prefix
        if async_engine is not None:
            self._engine = async_engine
        elif dsn is not None:
            from trellion.db.engine import _async_dsn
            self._engine = create_async_engine(_async_dsn(dsn), pool_pre_ping=True)
        else:
            from trellion.db.engine import get_async_engine
            self._engine = get_async_engine()
        self._embedding_dim = (
            embedding_dim if embedding_dim is not None else _get_embedding_dim()
        )

    def _format_vector(self, values: List[float]) -> str:
        """Format a float list as a pgvector literal, validating dimension."""
        expected = self._embedding_dim
        if len(values) != expected:
            raise ValueError(
                f"Embedding dimension mismatch: got {len(values)}, "
                f"expected {expected} (services.embedding_dim). "
                f"Check your embedding model or set EMBEDDING_DIM={len(values)}."
            )
        return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"

    # ==================== CONVERSATION MANAGEMENT ====================

    async def create_conversation(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict] = None,
        persona_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> str:
        """Create a new conversation and return its ID.

        ``org_id`` is the tenant boundary (PR5b): set it so the conversation is
        readable only within that org. ``None`` leaves it unassigned (single-org
        deployments never set it and see everything).
        """
        conv_id = str(uuid.uuid4())

        async with self._engine.connect() as conn:
            await conn.execute(
                _conv.insert().values(
                    id=conv_id,
                    user_id=user_id,
                    org_id=org_id,
                    session_id=session_id,
                    persona_id=persona_id,
                    backend=backend,
                    model=model,
                    total_messages=0,
                    metadata=metadata or {},
                )
            )
            await conn.commit()

        logger.debug("conversation_created", id=conv_id, user_id=user_id, org_id=org_id, persona_id=persona_id)
        return conv_id

    async def ensure_conversation(
        self,
        conversation_id: str,
        user_id: Optional[str] = None,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict] = None,
        persona_id: Optional[str] = None,
    ) -> str:
        """Ensure a conversation row exists (upsert). Returns conversation_id."""
        async with self._engine.connect() as conn:
            stmt = pg_insert(_conv).values(
                id=conversation_id,
                user_id=user_id,
                persona_id=persona_id,
                backend=backend,
                model=model,
                total_messages=0,
                metadata=metadata or {},
            ).on_conflict_do_nothing(index_elements=["id"])
            await conn.execute(stmt)
            await conn.commit()
        return conversation_id

    async def end_conversation(self, conversation_id: str) -> None:
        """Mark conversation as ended."""
        async with self._engine.connect() as conn:
            await conn.execute(
                _conv.update()
                .where(_conv.c.id == conversation_id)
                .values(ended_at=func.now())
            )
            await conn.commit()

    async def list_unarchived_conversations(self, limit: int = 100, min_messages: int = 2) -> List[Dict]:
        """Return conversations not yet archived to video memory.

        Filters by metadata->>'video_archived' IS NULL and total_messages >= min_messages.
        """
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(
                    _conv.c.id,
                    _conv.c.user_id,
                    _conv.c.started_at,
                    _conv.c.total_messages,
                    _conv.c.model,
                    _conv.c.metadata,
                )
                .where(
                    and_(
                        _conv.c.metadata["video_archived"].astext.is_(None),
                        _conv.c.total_messages >= min_messages,
                    )
                )
                .order_by(_conv.c.started_at.asc())
                .limit(limit)
            )).fetchall()

        return [
            {
                "id": str(r[0]),
                "user_id": r[1],
                "started_at": r[2].isoformat() if r[2] else None,
                "total_messages": r[3],
                "model": r[4],
                "metadata": r[5] or {},
            }
            for r in rows
        ]

    async def count_unarchived_conversations(self, min_messages: int = 2) -> int:
        """Count conversations not yet archived to video memory."""
        async with self._engine.connect() as conn:
            return (await conn.execute(
                select(func.count())
                .select_from(_conv)
                .where(
                    and_(
                        _conv.c.metadata["video_archived"].astext.is_(None),
                        _conv.c.total_messages >= min_messages,
                    )
                )
            )).scalar() or 0

    async def mark_conversations_archived(self, conversation_ids: List[str]) -> int:
        """Set video_archived=true in metadata for the given conversations."""
        if not conversation_ids:
            return 0
        async with self._engine.connect() as conn:
            result = await conn.execute(
                _conv.update()
                .where(_conv.c.id.in_(conversation_ids))
                .values(
                    metadata=func.coalesce(_conv.c.metadata, cast({}, JSONB)).concat(
                        cast({"video_archived": True}, JSONB)
                    )
                )
            )
            await conn.commit()
            return result.rowcount

    # ==================== ENTRY RECORDING ====================

    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a user/assistant/system message."""
        entry_id = str(uuid.uuid4())
        msg_type = MessageType(role) if role in [e.value for e in MessageType] else MessageType.USER

        async with self._engine.connect() as conn:
            await conn.execute(
                _entry.insert().values(
                    id=entry_id,
                    conversation_id=conversation_id,
                    type=msg_type.value,
                    content=content,
                    metadata=metadata or {},
                )
            )

            # Update conversation stats
            await conn.execute(
                _conv.update()
                .where(_conv.c.id == conversation_id)
                .values(total_messages=func.coalesce(_conv.c.total_messages, 0) + 1)
            )
            await conn.commit()

        return entry_id

    async def add_tool_call(
        self,
        conversation_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
        parent_message_id: Optional[str] = None
    ) -> str:
        """Record a tool being called."""
        entry_id = str(uuid.uuid4())
        content = f"Tool: {tool_name}\nArguments: {json.dumps(arguments, indent=2)}"

        async with self._engine.connect() as conn:
            # Add entry
            await conn.execute(
                _entry.insert().values(
                    id=entry_id,
                    conversation_id=conversation_id,
                    parent_id=parent_message_id,
                    type=MessageType.TOOL_CALL.value,
                    content=content,
                    metadata={"tool_name": tool_name, "arguments": arguments},
                )
            )

            # Add to tool executions
            await conn.execute(
                _tool.insert().values(
                    id=entry_id,
                    conversation_id=conversation_id,
                    entry_id=entry_id,
                    tool_name=tool_name,
                    arguments=arguments,
                )
            )

            # Update conversation stats
            await conn.execute(
                _conv.update()
                .where(_conv.c.id == conversation_id)
                .values(total_tool_calls=func.coalesce(_conv.c.total_tool_calls, 0) + 1)
            )
            await conn.commit()

        return entry_id

    async def add_tool_result(
        self,
        conversation_id: str,
        tool_call_id: str,
        success: bool,
        output: Optional[str] = None,
        error: Optional[str] = None,
        execution_ms: Optional[int] = None
    ) -> str:
        """Record a tool execution result."""
        entry_id = str(uuid.uuid4())

        content = output if success else f"Error: {error}"
        if content and len(content) > 10000:
            content = content[:10000] + "... (truncated)"

        async with self._engine.connect() as conn:
            # Add result entry
            await conn.execute(
                _entry.insert().values(
                    id=entry_id,
                    conversation_id=conversation_id,
                    parent_id=tool_call_id,
                    type=MessageType.TOOL_RESULT.value,
                    content=content or "",
                    metadata={"success": success, "execution_ms": execution_ms},
                )
            )

            # Update tool execution record
            await conn.execute(
                _tool.update()
                .where(_tool.c.id == tool_call_id)
                .values(
                    success=success,
                    output=output[:50000] if output else None,
                    error=error,
                    execution_ms=execution_ms,
                )
            )
            await conn.commit()

        return entry_id

    async def add_reasoning(
        self,
        conversation_id: str,
        reasoning: str,
        reasoning_type: str = "chain_of_thought",
        parent_id: Optional[str] = None
    ) -> str:
        """Record reasoning/thinking trace."""
        entry_id = str(uuid.uuid4())

        async with self._engine.connect() as conn:
            await conn.execute(
                _entry.insert().values(
                    id=entry_id,
                    conversation_id=conversation_id,
                    parent_id=parent_id,
                    type=MessageType.REASONING.value,
                    content=reasoning,
                    metadata={"reasoning_type": reasoning_type},
                )
            )
            await conn.commit()

        return entry_id

    async def add_context_retrieval(
        self,
        conversation_id: str,
        query: str,
        scope: str,
        results: List[Dict],
        top_score: Optional[float] = None,
        retrieval_ms: Optional[int] = None
    ) -> str:
        """Record what context was retrieved from memory."""
        entry_id = str(uuid.uuid4())

        async with self._engine.connect() as conn:
            # Add entry
            await conn.execute(
                _entry.insert().values(
                    id=entry_id,
                    conversation_id=conversation_id,
                    type=MessageType.CONTEXT_RETRIEVAL.value,
                    content=f"Retrieved {len(results)} results from {scope}",
                    metadata={"query": query, "scope": scope, "count": len(results)},
                )
            )

            # Add detailed retrieval record
            await conn.execute(
                _ctx.insert().values(
                    id=entry_id,
                    conversation_id=conversation_id,
                    entry_id=entry_id,
                    query=query,
                    scope=scope,
                    results=results,
                    top_score=top_score,
                    retrieval_ms=retrieval_ms,
                )
            )
            await conn.commit()

        return entry_id

    async def add_feedback(
        self,
        conversation_id: str,
        feedback_type: str,
        value: Any,
        entry_id: Optional[str] = None
    ) -> str:
        """Record user feedback (thumbs up/down, correction, regeneration, etc.)."""
        feedback_id = str(uuid.uuid4())

        async with self._engine.connect() as conn:
            await conn.execute(
                _fb.insert().values(
                    id=feedback_id,
                    conversation_id=conversation_id,
                    entry_id=entry_id,
                    feedback_type=feedback_type,
                    value=value,
                )
            )
            await conn.commit()

        logger.debug("feedback_recorded", type=feedback_type, conversation_id=conversation_id)
        return feedback_id

    async def get_feedback_for_dpo(self, min_rating: int = 3) -> List[Dict]:
        """Return DPO-ready pairs from conversation feedback.

        Joins noo_feedback → noo_entries → noo_conversations to build
        (prompt, chosen, rejected) tuples without touching the standalone
        feedback module's tables.
        """
        async with self._engine.connect() as conn:
            # Get all rated feedback with their entry content + conversation_id
            rows = (await conn.execute(
                select(
                    _fb.c.conversation_id,
                    _fb.c.entry_id,
                    _fb.c.feedback_type,
                    _fb.c.value,
                    _entry.c.content.label("entry_content"),
                )
                .join(_entry, _fb.c.entry_id == _entry.c.id)
                .where(_fb.c.feedback_type == "rating")
            )).fetchall()

            if not rows:
                return []

            # Group by conversation
            from collections import defaultdict
            conv_feedback: Dict[str, list] = defaultdict(list)
            for r in rows:
                conv_feedback[str(r[0])].append({
                    "entry_id": str(r[1]),
                    "rating": r[3] if isinstance(r[3], (int, float)) else None,
                    "content": r[4],
                })

            dpo_pairs = []
            for conv_id, items in conv_feedback.items():
                rated = [i for i in items if i["rating"] is not None]
                if not rated:
                    continue

                good = [i for i in rated if i["rating"] >= min_rating]
                bad = [i for i in rated if i["rating"] < min_rating]
                if not good or not bad:
                    continue

                # Get first user message as prompt
                first_user = (await conn.execute(
                    select(_entry.c.content)
                    .where(_entry.c.conversation_id == conv_id)
                    .where(_entry.c.type == "user")
                    .order_by(_entry.c.created_at.asc())
                    .limit(1)
                )).scalar()
                if not first_user:
                    continue

                for g in good:
                    for b in bad:
                        if g["entry_id"] != b["entry_id"]:
                            dpo_pairs.append({
                                "prompt": first_user,
                                "chosen": g["content"],
                                "rejected": b["content"],
                            })

            return dpo_pairs

    async def get_feedback_stats(self) -> Dict:
        """Aggregate feedback stats from noo_feedback table."""
        async with self._engine.connect() as conn:
            total = (await conn.execute(
                select(func.count()).select_from(_fb)
            )).scalar() or 0

            thumbs_up = (await conn.execute(
                select(func.count()).select_from(_fb)
                .where(_fb.c.feedback_type == "thumbs_up")
            )).scalar() or 0

            thumbs_down = (await conn.execute(
                select(func.count()).select_from(_fb)
                .where(_fb.c.feedback_type == "thumbs_down")
            )).scalar() or 0

            ratings = (await conn.execute(
                select(_fb.c.value)
                .where(_fb.c.feedback_type == "rating")
            )).fetchall()

            avg_rating = None
            if ratings:
                nums = [r[0] for r in ratings if isinstance(r[0], (int, float))]
                if nums:
                    avg_rating = sum(nums) / len(nums)

            corrections = (await conn.execute(
                select(func.count()).select_from(_fb)
                .where(_fb.c.feedback_type == "correction")
            )).scalar() or 0

            return {
                "total_feedback": total,
                "thumbs_up": thumbs_up,
                "thumbs_down": thumbs_down,
                "thumbs_up_ratio": thumbs_up / (thumbs_up + thumbs_down) if (thumbs_up + thumbs_down) > 0 else 0.0,
                "avg_rating": avg_rating,
                "corrections_count": corrections,
            }

    async def update_metrics(
        self,
        conversation_id: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: int = 0
    ) -> None:
        """Update conversation performance metrics."""
        async with self._engine.connect() as conn:
            await conn.execute(
                _conv.update()
                .where(_conv.c.id == conversation_id)
                .values(
                    total_tokens_in=func.coalesce(_conv.c.total_tokens_in, 0) + tokens_in,
                    total_tokens_out=func.coalesce(_conv.c.total_tokens_out, 0) + tokens_out,
                    total_latency_ms=func.coalesce(_conv.c.total_latency_ms, 0) + latency_ms,
                )
            )
            await conn.commit()

    # ==================== RETRIEVAL ====================

    async def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation with all entries."""
        async with self._engine.connect() as conn:
            # Get conversation
            row = (await conn.execute(
                select(
                    _conv.c.id, _conv.c.user_id, _conv.c.session_id,
                    _conv.c.backend, _conv.c.model,
                    _conv.c.started_at, _conv.c.ended_at,
                    _conv.c.total_messages, _conv.c.total_tool_calls,
                    _conv.c.total_tokens_in, _conv.c.total_tokens_out,
                    _conv.c.total_latency_ms, _conv.c.metadata,
                )
                .where(_conv.c.id == conversation_id)
            )).fetchone()
            if not row:
                return None

            conv = {
                "id": str(row[0]),
                "user_id": row[1],
                "session_id": row[2],
                "backend": row[3],
                "model": row[4],
                "started_at": row[5].isoformat() if row[5] else None,
                "ended_at": row[6].isoformat() if row[6] else None,
                "total_messages": row[7],
                "total_tool_calls": row[8],
                "total_tokens_in": row[9],
                "total_tokens_out": row[10],
                "total_latency_ms": row[11],
                "metadata": row[12],
            }

            # Get entries
            rows = (await conn.execute(
                select(
                    _entry.c.id, _entry.c.parent_id, _entry.c.type,
                    _entry.c.content, _entry.c.metadata, _entry.c.created_at,
                )
                .where(_entry.c.conversation_id == conversation_id)
                .order_by(_entry.c.created_at.asc())
            )).fetchall()

            conv["entries"] = [
                {
                    "id": str(r[0]),
                    "parent_id": str(r[1]) if r[1] else None,
                    "type": r[2],
                    "content": r[3],
                    "metadata": r[4],
                    "created_at": r[5].isoformat() if r[5] else None,
                }
                for r in rows
            ]

            return conv

    async def get_tool_stats(self, days: int = 30) -> Dict:
        """Get tool usage statistics."""
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(
                    _tool.c.tool_name,
                    func.count().label("total_calls"),
                    func.sum(case((_tool.c.success == True, 1), else_=0)).label("successful"),  # noqa: E712
                    func.avg(_tool.c.execution_ms).label("avg_execution_ms"),
                )
                .where(_tool.c.created_at > func.now() - text(f"INTERVAL '{int(days)} days'"))
                .group_by(_tool.c.tool_name)
                .order_by(func.count().desc())
            )).fetchall()

            return {
                "period_days": days,
                "tools": [
                    {
                        "name": r[0],
                        "total_calls": r[1],
                        "successful": r[2],
                        "success_rate": r[2] / r[1] if r[1] > 0 else 0,
                        "avg_execution_ms": float(r[3]) if r[3] else None,
                    }
                    for r in rows
                ]
            }

    # ==================== CHANNEL CONVERSATION QUERIES ====================

    async def find_active_conversation(
        self,
        user_id: str,
        source_channel: str,
    ) -> Optional[str]:
        """Find an open conversation for a user on a specific channel."""
        async with self._engine.connect() as conn:
            row = (await conn.execute(
                select(_conv.c.id)
                .where(
                    and_(
                        _conv.c.user_id == user_id,
                        _conv.c.metadata["source_channel"].astext == source_channel,
                        _conv.c.ended_at.is_(None),
                    )
                )
                .order_by(_conv.c.started_at.desc())
                .limit(1)
            )).fetchone()
            return str(row[0]) if row else None

    async def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10,
        types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch the last N entries ordered chronologically (oldest first)."""
        if types is None:
            types = ["user", "assistant"]

        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(
                    _entry.c.type, _entry.c.content,
                    _entry.c.metadata, _entry.c.created_at,
                )
                .where(
                    and_(
                        _entry.c.conversation_id == conversation_id,
                        _entry.c.type.in_(types),
                    )
                )
                .order_by(_entry.c.created_at.desc())
                .limit(limit)
            )).fetchall()

        # Reverse to get oldest-first order
        rows = list(reversed(rows))
        return [
            {
                "type": r[0],
                "content": r[1],
                "metadata": r[2] or {},
                "created_at": r[3].isoformat() if r[3] else None,
            }
            for r in rows
        ]

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        source_channel: Optional[str] = None,
        active_only: bool = False,
        limit: int = 20,
        offset: int = 0,
        persona_id: Optional[str] = None,
        org: Optional[str] = None,
        teams: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Paginated conversation list with optional filters.

        Tenant scoping (PR5): ``org`` bounds the list to that org; ``teams``
        widens the ``user_id`` match to the caller's team-owned conversations.
        """
        from sqlalchemy import or_

        conditions = []
        if org:
            conditions.append(_conv.c.org_id == org)
        if user_id:
            if teams:
                conditions.append(or_(_conv.c.user_id == user_id, _conv.c.user_id.in_(list(teams))))
            else:
                conditions.append(_conv.c.user_id == user_id)
        if source_channel:
            conditions.append(_conv.c.metadata["source_channel"].astext == source_channel)
        if active_only:
            conditions.append(_conv.c.ended_at.is_(None))
        if persona_id is not None:
            conditions.append(_conv.c.persona_id == persona_id)

        where_clause = and_(*conditions) if conditions else True

        async with self._engine.connect() as conn:
            # Total count
            total = (await conn.execute(
                select(func.count()).select_from(_conv).where(where_clause)
            )).scalar()

            # Paginated results
            rows = (await conn.execute(
                select(
                    _conv.c.id, _conv.c.user_id, _conv.c.session_id,
                    _conv.c.backend, _conv.c.model,
                    _conv.c.started_at, _conv.c.ended_at,
                    _conv.c.total_messages, _conv.c.total_tool_calls,
                    _conv.c.metadata,
                )
                .where(where_clause)
                .order_by(_conv.c.started_at.desc())
                .limit(limit)
                .offset(offset)
            )).fetchall()

            conversations = [
                {
                    "id": str(r[0]),
                    "user_id": r[1],
                    "session_id": r[2],
                    "backend": r[3],
                    "model": r[4],
                    "started_at": r[5].isoformat() if r[5] else None,
                    "ended_at": r[6].isoformat() if r[6] else None,
                    "total_messages": r[7],
                    "total_tool_calls": r[8],
                    "metadata": r[9] or {},
                }
                for r in rows
            ]

        return {"conversations": conversations, "total": total}

    async def list_conversations_with_preview(
        self,
        user_id: Optional[str] = None,
        source_channel: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Paginated conversation list with first-user-message preview.

        Uses a LATERAL join which requires raw SQL since SQLAlchemy Core
        doesn't natively support LATERAL subqueries cleanly.
        """
        conditions: List[str] = []
        params: Dict[str, Any] = {}

        if user_id:
            conditions.append("c.user_id = :user_id")
            params["user_id"] = user_id
        if source_channel:
            conditions.append("c.metadata->>'source_channel' = :channel")
            params["channel"] = source_channel

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        async with self._engine.connect() as conn:
            total = (await conn.execute(
                text(f"SELECT COUNT(*) FROM {_conv.name} c {where}"),
                params,
            )).scalar()

            params["limit"] = limit
            params["offset"] = offset
            rows = (await conn.execute(text(f"""
                SELECT c.id, c.started_at, c.total_messages, c.model,
                       LEFT(p.content, 120) AS preview
                FROM {_conv.name} c
                LEFT JOIN LATERAL (
                    SELECT content FROM {_entry.name}
                    WHERE conversation_id = c.id AND type = 'user'
                    ORDER BY created_at ASC
                    LIMIT 1
                ) p ON TRUE
                {where}
                ORDER BY c.started_at DESC
                LIMIT :limit OFFSET :offset
            """), params)).fetchall()

            conversations = [
                {
                    "id": str(r[0]),
                    "started_at": r[1].isoformat() if r[1] else None,
                    "total_messages": r[2],
                    "model": r[3],
                    "preview": r[4] or "",
                }
                for r in rows
            ]

        return {"conversations": conversations, "total": total}

    @staticmethod
    def _routing_from_metadata(meta: Dict[str, Any], uid: str) -> Dict[str, Any]:
        """Resolve a reply address from conversation metadata.

        Prefers the normalized ``noo_routing`` block the harness writes. That
        block is the explicit contract: the harness knows what a "space" or a
        "chat id" means, this layer does not, so it states the answer once and
        we store it opaquely.

        The ``legacy`` branch reads the per-channel keys directly, which is
        what this method used to do unconditionally. It stays ONLY for
        conversations created before the contract landed — those rows have no
        ``noo_routing`` and would otherwise lose their reply address. It is a
        migration tail, not a fallback to keep extending: a new channel that
        needs a different address shape must set ``noo_routing``, not add
        another key here. (Guessing across four keys is precisely how this
        coupling drifted and why the Rust channel port would have broken it
        silently — see gateway followup 20.)
        """
        routing = meta.get("noo_routing")
        if isinstance(routing, dict) and routing.get("recipient"):
            return {
                "recipient": routing["recipient"],
                "chat_type": routing.get("chat_type", "private"),
                "display_name": routing.get("display_name") or uid,
            }
        # legacy (pre-contract rows)
        return {
            "recipient": meta.get("space_name") or meta.get("chat_id") or uid.split(":", 1)[-1],
            "chat_type": meta.get("chat_type", "private"),
            "display_name": meta.get("chat_title") or meta.get("space_display_name") or uid,
        }

    async def list_known_recipients(
        self,
        source_channel: str,
    ) -> List[Dict[str, Any]]:
        """Return distinct chat contexts seen on a channel."""
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(
                    _conv.c.user_id, _conv.c.metadata,
                    _conv.c.total_messages, _conv.c.started_at,
                )
                .where(_conv.c.metadata["source_channel"].astext == source_channel)
                .order_by(_conv.c.started_at.desc())
            )).fetchall()

            seen: dict[str, dict] = {}
            for r in rows:
                uid = r[0]
                if uid in seen:
                    seen[uid]["total_messages"] += r[2] or 0
                    continue
                meta = r[1] or {}
                seen[uid] = {
                    "user_id": uid,
                    **self._routing_from_metadata(meta, uid),
                    "total_messages": r[2] or 0,
                    "last_active": r[3].isoformat() if r[3] else None,
                }

        return list(seen.values())

    # ==================== VECTOR / SEMANTIC SEARCH ====================

    async def search_semantic(
        self,
        embedding: List[float],
        top_k: int = 5,
        user_id: Optional[str] = None,
        persona_id: Optional[str] = None,
    ) -> List[Tuple[dict, float]]:
        """Cosine similarity search on noo_entries.embedding.

        Uses raw SQL for pgvector's ``<=>`` operator which SQLAlchemy Core
        doesn't expose natively.

        When ``persona_id`` is provided, results are scoped to that persona's
        conversations (persona_id match) plus shared conversations (persona_id IS NULL).
        """
        vector = self._format_vector(embedding)

        async with self._engine.connect() as conn:
            # Build dynamic WHERE clauses
            where_parts = ["e.embedding IS NOT NULL"]
            params: Dict[str, Any] = {"vec": vector, "vec2": vector, "top_k": top_k}

            needs_join = user_id is not None or persona_id is not None
            join_clause = f"JOIN {_conv.name} c ON e.conversation_id = c.id" if needs_join else ""

            if user_id:
                where_parts.append("(c.user_id = :user_id OR c.user_id IS NULL)")
                params["user_id"] = user_id

            if persona_id is not None:
                where_parts.append("(c.persona_id = :persona_id OR c.persona_id IS NULL)")
                params["persona_id"] = persona_id

            # Compare only vectors from the space's current model (dropped,
            # never emptied, when the model cannot be determined).
            model_sql, model_params = await embedding_space.model_clause(
                "e.embedding_model", query_vector=embedding
            )
            if model_sql:
                where_parts.append(model_sql)
                params.update(model_params)

            where_sql = " AND ".join(where_parts)

            rows = (await conn.execute(text(f"""
                SELECT e.id, e.type, e.content, e.created_at, e.metadata,
                       1 - (e.embedding <=> CAST(:vec AS vector)) AS score
                FROM {_entry.name} e
                {join_clause}
                WHERE {where_sql}
                ORDER BY e.embedding <=> CAST(:vec2 AS vector)
                LIMIT :top_k
            """), params)).fetchall()

            return [
                (
                    {
                        "id": str(r[0]),
                        "type": r[1],
                        "content": r[2],
                        "created_at": r[3].timestamp() if r[3] else 0.0,
                        "metadata": r[4] or {},
                    },
                    float(r[5]),
                )
                for r in rows
            ]

    async def search_text(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        persona_id: Optional[str] = None,
        org: Optional[str] = None,
        teams: Optional[List[str]] = None,
    ) -> List[Tuple[dict, float]]:
        """Trigram similarity search on noo_entries.content.

        Uses raw SQL for pg_trgm's ``similarity()`` ranking. We avoid the
        threshold operator so short probe strings are not dropped by the
        database's global pg_trgm similarity threshold.

        Tenant scoping (PR5): ``org`` bounds results to that org (hard
        boundary). Within it, ``user_id`` + ``teams`` select the personal,
        team, and org-shared (``user_id IS NULL``) tiers. All optional; a
        single-org / single-user caller passes none and sees everything.
        """
        if not query:
            return []

        async with self._engine.connect() as conn:
            where_parts = ["similarity(e.content, :query2) > 0"]
            params: Dict[str, Any] = {"query": query, "query2": query, "query3": query, "top_k": top_k}

            needs_join = user_id is not None or persona_id is not None or org is not None
            join_clause = f"JOIN {_conv.name} c ON e.conversation_id = c.id" if needs_join else ""

            if org:
                # Hard org boundary — no cross-org rows, not even org-shared.
                where_parts.append("c.org_id = :org")
                params["org"] = org

            if user_id:
                if teams:
                    where_parts.append(
                        "(c.user_id = :user_id OR c.user_id = ANY(:teams) OR c.user_id IS NULL)"
                    )
                    params["teams"] = list(teams)
                else:
                    where_parts.append("(c.user_id = :user_id OR c.user_id IS NULL)")
                params["user_id"] = user_id

            if persona_id is not None:
                where_parts.append("(c.persona_id = :persona_id OR c.persona_id IS NULL)")
                params["persona_id"] = persona_id

            where_sql = " AND ".join(where_parts)

            rows = (await conn.execute(text(f"""
                SELECT e.id, e.type, e.content, e.created_at, e.metadata,
                       similarity(e.content, :query) AS score
                FROM {_entry.name} e
                {join_clause}
                WHERE {where_sql}
                ORDER BY similarity(e.content, :query3) DESC
                LIMIT :top_k
            """), params)).fetchall()

            return [
                (
                    {
                        "id": str(r[0]),
                        "type": r[1],
                        "content": r[2],
                        "created_at": r[3].timestamp() if r[3] else 0.0,
                        "metadata": r[4] or {},
                    },
                    float(r[5]),
                )
                for r in rows
            ]

    async def get_unembedded(self, limit: int = 100) -> List[Tuple[str, str]]:
        """Get entries without embeddings."""
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(_entry.c.id, _entry.c.content)
                .where(_entry.c.embedding.is_(None))
                .order_by(_entry.c.created_at.asc())
                .limit(limit)
            )).fetchall()
            return [(str(r[0]), r[1]) for r in rows]

    async def update_embedding(self, entry_id: str, embedding: List[float]) -> bool:
        """Set the embedding vector for an entry.

        Uses raw SQL for CAST to pgvector type.
        """
        vector = self._format_vector(embedding)

        async with self._engine.connect() as conn:
            result = await conn.execute(text(f"""
                UPDATE {_entry.name}
                SET embedding = CAST(:vec AS vector), indexed_at = NOW(),
                    embedding_model = :model, embedding_dim = :dim,
                    embedding_next = NULL, embedding_next_model = NULL
                WHERE id = :id
            """), {
                "vec": vector, "id": entry_id,
                "model": embedding_space.model_of(embedding),
                "dim": len(embedding),
            })
            await conn.commit()
            return result.rowcount > 0

    async def batch_update_embeddings(
        self, updates: List[Tuple[str, List[float]]]
    ) -> int:
        """Batch-set embeddings via executemany (pipelined in asyncpg).

        Uses raw SQL for CAST to pgvector type.
        """
        if not updates:
            return 0

        rows = [
            {
                "id": eid, "vec": self._format_vector(emb),
                "model": embedding_space.model_of(emb), "dim": len(emb),
            }
            for eid, emb in updates
        ]
        async with self._engine.connect() as conn:
            await conn.execute(text(f"""
                UPDATE {_entry.name}
                SET embedding = CAST(:vec AS vector), indexed_at = NOW(),
                    embedding_model = :model, embedding_dim = :dim,
                    embedding_next = NULL, embedding_next_model = NULL
                WHERE id = :id
            """), rows)
            await conn.commit()
        return len(updates)

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get conversation entries for a user across all conversations."""
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(
                    _entry.c.id, _entry.c.type, _entry.c.content,
                    _entry.c.created_at, _entry.c.metadata,
                )
                .select_from(_entry.join(_conv, _entry.c.conversation_id == _conv.c.id))
                .where(
                    and_(
                        _conv.c.user_id == user_id,
                        _entry.c.type.in_(["user", "assistant"]),
                    )
                )
                .order_by(_entry.c.created_at.desc())
                .limit(limit)
            )).fetchall()

        # Return oldest-first
        rows = list(reversed(rows))
        return [
            {
                "id": str(r[0]),
                "role": r[1],
                "content": r[2],
                "created_at": r[3].isoformat() if r[3] else None,
                "metadata": r[4] or {},
            }
            for r in rows
        ]

    async def get_stats(self) -> Dict[str, Any]:
        """Embedding statistics from noo_entries."""
        async with self._engine.connect() as conn:
            row = (await conn.execute(
                select(
                    func.count().label("total"),
                    func.count(_entry.c.embedding).label("embedded"),
                    (func.count() - func.count(_entry.c.embedding)).label("unembedded"),
                    func.min(_entry.c.created_at).label("oldest"),
                    func.max(_entry.c.created_at).label("newest"),
                )
                .select_from(_entry)
            )).fetchone()

            return {
                "total_records": row[0],
                "embedded_records": row[1],
                "unembedded_records": row[2],
                "unique_users": 0,
                "oldest_record": row[3].isoformat() if row[3] else None,
                "newest_record": row[4].isoformat() if row[4] else None,
            }

    # ==================== STATISTICS ====================

    async def get_conversation_stats(self, user_id: Optional[str] = None, days: int = 30) -> Dict:
        """Get conversation statistics."""
        conditions = [_conv.c.started_at > func.now() - text(f"INTERVAL '{int(days)} days'")]
        if user_id:
            conditions.append(_conv.c.user_id == user_id)

        async with self._engine.connect() as conn:
            row = (await conn.execute(
                select(
                    func.count().label("total_conversations"),
                    func.sum(_conv.c.total_messages).label("total_messages"),
                    func.sum(_conv.c.total_tool_calls).label("total_tool_calls"),
                    func.avg(_conv.c.total_messages).label("avg_messages_per_conv"),
                    func.sum(_conv.c.total_tokens_in).label("total_tokens_in"),
                    func.sum(_conv.c.total_tokens_out).label("total_tokens_out"),
                    func.avg(_conv.c.total_latency_ms).label("avg_latency_ms"),
                )
                .select_from(_conv)
                .where(and_(*conditions))
            )).fetchone()

            return {
                "period_days": days,
                "user_id": user_id,
                "total_conversations": row[0] or 0,
                "total_messages": row[1] or 0,
                "total_tool_calls": row[2] or 0,
                "avg_messages_per_conversation": float(row[3]) if row[3] else 0,
                "total_tokens_in": row[4] or 0,
                "total_tokens_out": row[5] or 0,
                "avg_latency_ms": float(row[6]) if row[6] else 0,
            }

    # ==================== CLEANUP & MAINTENANCE ====================

    async def get_old_conversations(
        self,
        older_than_days: int = 30,
        min_messages: int = 5,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get old conversations with enough messages, candidates for summarization."""
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(
                    _conv.c.id, _conv.c.user_id,
                    _conv.c.started_at, _conv.c.ended_at,
                    _conv.c.total_messages, _conv.c.metadata,
                )
                .where(
                    and_(
                        _conv.c.started_at < func.now() - text(f"INTERVAL '{int(older_than_days)} days'"),
                        _conv.c.total_messages >= min_messages,
                    )
                )
                .order_by(_conv.c.started_at.asc())
                .limit(limit)
            )).fetchall()

            return [
                {
                    "id": str(r[0]),
                    "user_id": r[1],
                    "started_at": r[2].isoformat() if r[2] else None,
                    "ended_at": r[3].isoformat() if r[3] else None,
                    "total_messages": r[4],
                    "metadata": r[5] or {},
                }
                for r in rows
            ]

    async def cleanup_old(self, max_age_days: int = 90) -> int:
        """Delete conversations older than max_age_days. Returns count removed."""
        cutoff = func.now() - text(f"INTERVAL '{int(max_age_days)} days'")

        async with self._engine.connect() as conn:
            # Delete entries first (FK), then conversations
            old_ids = select(_conv.c.id).where(_conv.c.started_at < cutoff)
            await conn.execute(
                _entry.delete().where(_entry.c.conversation_id.in_(old_ids))
            )
            result = await conn.execute(
                _conv.delete().where(_conv.c.started_at < cutoff)
            )
            await conn.commit()
            return result.rowcount

    # ==================== RELATIONSHIP METHODS ====================
    # (Used by EmbeddingWorker for semantic relationship building)
    # These use raw SQL because noo_semantic_relationships may not have an ORM model.

    async def get_recent_embedded(self, limit: int = 1000) -> List[Tuple[str, List[float]]]:
        """Get recently embedded entries for relationship building."""
        async with self._engine.connect() as conn:
            rows = (await conn.execute(
                select(_entry.c.id, _entry.c.embedding)
                .where(
                    and_(
                        _entry.c.embedding.isnot(None),
                        _entry.c.indexed_at > func.now() - text("INTERVAL '1 day'"),
                    )
                )
                .order_by(_entry.c.indexed_at.desc())
                .limit(limit)
            )).fetchall()

            results = []
            for row in rows:
                record_id = str(row[0])
                embedding = _parse_vector(row[1])
                if embedding:
                    results.append((record_id, embedding))
            return results

    async def store_relationship(self, source_id: str, target_id: str, similarity: float) -> None:
        """Store a semantic relationship between two entries."""
        async with self._engine.connect() as conn:
            await conn.execute(text("""
                INSERT INTO noo_semantic_relationships (source_id, target_id, similarity)
                VALUES (:source_id, :target_id, :similarity)
                ON CONFLICT (source_id, target_id, relationship_type)
                DO UPDATE SET similarity = EXCLUDED.similarity, created_at = NOW()
            """), {"source_id": source_id, "target_id": target_id, "similarity": similarity})
            await conn.commit()

    async def get_semantic_relationships(self, entry_id: str) -> List[Dict]:
        """Get semantic relationships for an entry."""
        async with self._engine.connect() as conn:
            rows = (await conn.execute(text("""
                SELECT target_id, similarity
                FROM noo_semantic_relationships
                WHERE source_id = :id
                ORDER BY similarity DESC
            """), {"id": entry_id})).fetchall()
            return [{"target_id": str(r[0]), "similarity": float(r[1])} for r in rows]


def _parse_vector(pg_vector: Any) -> Optional[List[float]]:
    """Parse PostgreSQL vector to list of floats."""
    if pg_vector is None:
        return None
    if isinstance(pg_vector, list):
        return pg_vector
    # Handle string format [0.1,0.2,...]
    if isinstance(pg_vector, str):
        try:
            return [float(x) for x in pg_vector.strip('[]').split(',')]
        except Exception:
            return None
    return None
