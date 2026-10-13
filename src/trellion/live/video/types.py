# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Lightweight dataclasses shared across the video memory layer.

Kept in their own module so callers can import without pulling in the
async manager + FAISS deps.

Defined in ``trellion_protocols.types`` (the contract package an add-on
imports) and re-exported here: one class object for each.
"""
from __future__ import annotations

from trellion_protocols.types import TraversalSnapshot, TraversalStep

__all__ = ["TraversalSnapshot", "TraversalStep"]
