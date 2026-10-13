# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

from trellion.live.manager import MemoryConfig, MemorySystem


class FakeConversationStore:
    def __init__(self):
        self.conversations = []
        self.messages = []
        self.embeddings = []
        self.semantic_error = None
        self.semantic_results = None

    async def create_conversation(self, **kwargs):
        conversation_id = f"conv-{len(self.conversations) + 1}"
        self.conversations.append((conversation_id, kwargs))
        return conversation_id

    async def ensure_conversation(self, conversation_id, **kwargs):
        self.conversations.append((conversation_id, kwargs))
        return conversation_id

    async def add_message(self, conversation_id, role, content, metadata=None):
        self.messages.append((conversation_id, role, content, metadata))
        return "entry-1"

    async def update_embedding(self, entry_id, embedding):
        self.embeddings.append((entry_id, embedding))
        return True

    async def search_semantic(self, embedding, top_k=5, user_id=None, persona_id=None):
        if self.semantic_error:
            raise self.semantic_error
        if self.semantic_results is not None:
            return self.semantic_results
        return [
            (
                {
                    "id": "entry-1",
                    "type": "metadata",
                    "content": "Setup: smoke. Lesson: use semantic memory.",
                    "created_at": 0,
                    "metadata": {},
                },
                0.8,
            )
        ]

    async def search_text(self, query, top_k=5, user_id=None, persona_id=None):
        return [
            (
                {
                    "id": "entry-1",
                    "type": "metadata",
                    "content": "Setup: smoke. Lesson: use Postgres text retrieval.",
                    "created_at": 0,
                    "metadata": {},
                },
                0.7,
            )
        ]


def test_memory_system_sync_path_uses_conversation_store():
    store = FakeConversationStore()
    memory = MemorySystem(
        MemoryConfig(enabled=True, include_session=False, include_global=True, min_score=0.02),
        store=store,
    )

    memory.record_sync(
        None,
        "metadata",
        "Setup: smoke. Lesson: use Postgres text retrieval.",
        conversation_id="exp-not-a-uuid",
        metadata={"record_id": "exp-not-a-uuid"},
    )

    retrieved = memory.retrieve("smoke postgres", None)

    assert len(store.messages) == 1
    assert store.conversations[0][1]["metadata"]["external_conversation_id"] == "exp-not-a-uuid"
    assert "Postgres text retrieval" in retrieved


def test_memory_system_embeds_conversation_store_records_on_write():
    store = FakeConversationStore()
    memory = MemorySystem(
        MemoryConfig(enabled=True, include_session=False, include_global=True, min_score=0.02),
        store=store,
        embedder=lambda text: [0.1, 0.2, 0.3],
    )

    memory.record_sync(None, "metadata", "embed this record", conversation_id="exp-embed")

    assert store.embeddings == [("entry-1", [0.1, 0.2, 0.3])]


def test_memory_system_falls_back_to_text_when_semantic_search_fails():
    store = FakeConversationStore()
    store.semantic_error = RuntimeError("embedding service down")
    memory = MemorySystem(
        MemoryConfig(enabled=True, include_session=False, include_global=True, min_score=0.02),
        store=store,
        embedder=lambda text: [0.1, 0.2, 0.3],
    )

    retrieved = memory.retrieve("smoke postgres", None)

    assert "Postgres text retrieval" in retrieved


def test_memory_system_falls_back_to_text_when_semantic_search_is_empty():
    store = FakeConversationStore()
    store.semantic_results = []
    memory = MemorySystem(
        MemoryConfig(enabled=True, include_session=False, include_global=True, min_score=0.02),
        store=store,
        embedder=lambda text: [0.1, 0.2, 0.3],
    )

    retrieved = memory.retrieve("smoke postgres", None)

    assert "Postgres text retrieval" in retrieved
