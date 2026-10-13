# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Contract tests for the harness→Trellion conversation routing block.

Nothing guarded this before. `list_known_recipients` reconstructed a reply
address by guessing across channel-specific metadata keys, which meant the
memory layer encoded channel semantics and the contract was implicit,
unschema'd and untested — exactly the coupling that would have broken
silently once the Rust channel port started writing a different key shape
(gateway followup 20).

These lock the resolution order: the normalized `noo_routing` block wins,
and the legacy per-channel keys remain readable for rows created before the
contract landed.
"""

from trellion.live.conversation_store import ConversationStore

_resolve = ConversationStore._routing_from_metadata


class TestNormalizedBlockWins:
    def test_uses_noo_routing_when_present(self):
        meta = {
            "noo_routing": {
                "v": 1,
                "recipient": "spaces/AAQA123",
                "chat_type": "space",
                "display_name": "Ops Room",
            },
            # Legacy keys deliberately disagree — the block must win.
            "space_name": "spaces/WRONG",
            "chat_id": "users/wrong",
            "chat_title": "Wrong Title",
        }
        got = _resolve(meta, "google_chat:spaces/AAQA123")
        assert got["recipient"] == "spaces/AAQA123"
        assert got["chat_type"] == "space"
        assert got["display_name"] == "Ops Room"

    def test_display_name_falls_back_to_uid_but_recipient_never_does(self):
        meta = {"noo_routing": {"v": 1, "recipient": "users/42", "chat_type": "private"}}
        got = _resolve(meta, "telegram:42")
        assert got["recipient"] == "users/42"
        assert got["display_name"] == "telegram:42"

    def test_block_without_recipient_is_ignored(self):
        # A malformed block must not shadow a usable legacy address —
        # delivering nowhere is worse than delivering via the old path.
        meta = {"noo_routing": {"v": 1, "chat_type": "space"}, "chat_id": "users/7"}
        assert _resolve(meta, "telegram:7")["recipient"] == "users/7"


class TestLegacyRowsStillResolve:
    """Conversations created before the contract have no noo_routing."""

    def test_space_prefers_space_name(self):
        meta = {"space_name": "spaces/OLD", "chat_id": "users/9", "chat_type": "space"}
        got = _resolve(meta, "google_chat:spaces/OLD")
        assert got["recipient"] == "spaces/OLD"
        assert got["chat_type"] == "space"

    def test_dm_uses_chat_id(self):
        meta = {"chat_id": "users/9", "chat_type": "private"}
        assert _resolve(meta, "google_chat:users/9")["recipient"] == "users/9"

    def test_bare_metadata_derives_from_uid(self):
        assert _resolve({}, "telegram:12345")["recipient"] == "12345"

    def test_chat_type_defaults_to_private(self):
        # A missing chat_type must not make a DM look like a room.
        assert _resolve({"chat_id": "u"}, "telegram:u")["chat_type"] == "private"
