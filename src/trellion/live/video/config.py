# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Video memory configuration model."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from .paths import video_memory_dir


class VideoMemoryConfig(BaseModel):
    """Configuration for the MP4 compressed semantic memory system (System 1)."""

    enabled: bool = Field(default=False, description="Enable video memory system")
    storage_dir: str = Field(
        default_factory=video_memory_dir,
        description="Directory for FAISS indices and replay data; NOO_VIDEO_MEMORY_PATH decides it "
        "(on the noo-data volume)",
    )
    codec: str = Field(default="ffv1", description="Video codec (ffv1=lossless, mp4v=lossy fallback)")
    fps: int = Field(default=1, description="Frames per second (data storage, not playback)")
    frame_size: int = Field(default=800, description="Frame dimensions in pixels (square)")
    qr_version: int = Field(
        default=40,
        ge=1,
        le=40,
        description="QR code version (40 = max capacity ~2953 bytes binary)",
    )
    qr_error_correction: str = Field(
        default="M",
        description="QR error correction level: L (7%), M (15%), Q (25%), H (30%)",
    )
    tile_layout: str = Field(
        default="1x1",
        description="QR tile layout per frame: '1x1' or '2x2' (4x capacity)",
    )
    embedding_dim: int = Field(
        default_factory=lambda: int(os.environ.get("EMBEDDING_DIM", "768")),
        description="Embedding dimension (must match TEI/PageIndex)",
    )
    archive_interval_minutes: int = Field(
        default=10,
        description="Minutes between automatic archival runs",
    )
    archive_batch_size: int = Field(
        default=100,
        description="Max conversations to archive per batch",
    )
    max_video_size_mb: int = Field(
        default=500,
        description="Max single video file size in MB before rolling",
    )
    promotion_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Min score for promoting video results to System 2",
    )

    # -- Lifecycle management --
    prune_after_days: int = Field(
        default=90,
        ge=1,
        description="Min age (days) before a low-prominence record can be pruned",
    )
    prune_prominence_floor: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Records below this prominence are pruned (truly worthless)",
    )
    elevate_prominence_ceiling: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description=(
            "Records with prominence below this ceiling (but above prune floor) "
            "are consolidated into System 2 and removed from System 1"
        ),
    )
    elevate_min_access: int = Field(
        default=1,
        ge=0,
        description="Record must have been accessed at least this many times to be worth elevating",
    )
    lifecycle_interval_hours: int = Field(
        default=6,
        description="Hours between lifecycle (prune + elevate + compact) runs",
    )
    compact_tombstone_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Trigger video compaction when tombstoned fraction exceeds this",
    )
    max_evolution_snapshots: int = Field(
        default=50,
        description="Number of lifecycle evolution snapshots to retain",
    )

    # -- Hard retention caps (independent of prominence-based pruning above) --
    max_age_days: int = Field(
        default=365,
        ge=1,
        description=(
            "Hard age cap — records older than this are tombstoned regardless of "
            "prominence. Acts as the outer envelope; prominence-based pruning "
            "(prune_after_days) handles low-value records sooner."
        ),
    )
    max_size_gb: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Hard size cap on total MP4 storage. When exceeded, the lifecycle "
            "pass tombstones records oldest-first until under the cap. Set high "
            "(e.g. 100.0) to effectively disable size-based eviction."
        ),
    )

    # -- Storage backend --
    storage_backend: str = Field(
        default="local",
        description="Storage backend type: 'local' (filesystem) or 'azure' (Azure Blob Storage)",
    )
    azure_connection_string: str = Field(
        default="",
        description="Azure Blob Storage connection string (required when storage_backend='azure')",
    )
    azure_container: str = Field(
        default="video-memory",
        description="Azure Blob Storage container name",
    )
