# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

from __future__ import annotations

from pathlib import Path
import glob
from typing import List


def _expand_paths(patterns: List[str]) -> List[Path]:
    expanded: List[Path] = []
    for raw in patterns:
        matches = [Path(p) for p in glob.glob(raw, recursive=True)]
        expanded.extend(matches or [Path(raw)])
    return expanded


class ContextLoader:
    def __init__(self, files: List[str]) -> None:
        self.files = files

    def load(self, max_bytes: int = 16000) -> str:
        if not self.files:
            return ""
        chunks = []
        for path in _expand_paths(self.files):
            if not path.exists() or path.is_dir():
                continue
            data = path.read_bytes()
            if len(data) > max_bytes:
                data = data[:max_bytes]
            text = data.decode("utf-8", errors="replace")
            chunks.append(f"[context] {path}\n{text}")
        return "\n\n".join(chunks)
