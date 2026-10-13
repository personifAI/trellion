# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Document ingest + chunking.

Working implementation (txt/md). Hook in PDF/HTML loaders as needed later.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

from trellion.common.config import ChunkingConfig
from trellion.common.schema import Chunk

logger = logging.getLogger(__name__)

_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst"}


def load_documents(input_dir: Path) -> Iterable[tuple[str, str]]:
    """Yield (doc_id, text) pairs for every text file under input_dir."""
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in _TEXT_EXTS:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("skipping %s: %s", p, e)
            continue
        doc_id = _short_hash(str(p.resolve()))
        yield doc_id, text


def chunk_document(
    doc_id: str, text: str, cfg: ChunkingConfig
) -> list[Chunk]:
    """Simple word-based chunker with overlap.

    Not token-accurate, but adequate for extraction prompts where chunks feed
    into vLLM which applies its own tokenizer server-side.
    """
    words = text.split()
    if not words:
        return []

    # Approximate: treat 1 word ~ 1.3 tokens. Close enough for layout purposes.
    words_per_chunk = max(1, int(cfg.target_tokens / 1.3))
    overlap_words = max(0, int(cfg.overlap_tokens / 1.3))
    step = max(1, words_per_chunk - overlap_words)

    chunks: list[Chunk] = []
    # Precompute cumulative char offsets so we can report source ranges.
    # (Cheap; only run once per doc.)
    char_offset = 0
    offsets: list[int] = []
    for w in words:
        offsets.append(char_offset)
        char_offset += len(w) + 1  # +1 for the space we split on

    for i in range(0, len(words), step):
        window = words[i : i + words_per_chunk]
        if not window:
            break
        chunk_text = " ".join(window)
        chunk_id = f"{doc_id}:{i:06d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                token_count=int(len(window) * 1.3),
                offset=offsets[i] if i < len(offsets) else char_offset,
            )
        )
        if i + words_per_chunk >= len(words):
            break
    return chunks


def _short_hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
