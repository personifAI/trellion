# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""trellion.transport — wire formats for shipping memory across nodes.

The live storage layer (``trellion.live`` + ``trellion.db``) is rich and
chatty — fine when memory is local, but expensive when it has to travel
between controller, edge nodes, and companions in the noomesh cluster.

The transport layer is the cold/wire format. ``ParquetShardPackager``
takes a slice of the live store + the matching FAISS index files and
produces a single tarball that can be:

- rsync'd between controller and peer cluster for replication
- shipped to an edge node that lacks full DB access (DuckDB queries
  Parquet directly without trellion installed)
- exported as a backup
- applied to a clean instance via ``unpack()``

Public API:
    from trellion.transport import ParquetShardPackager, ShardManifest
"""
from __future__ import annotations

from .parquet import ParquetShardPackager, ShardManifest, ShardSliceSpec

__all__ = [
    "ParquetShardPackager",
    "ShardManifest",
    "ShardSliceSpec",
]
