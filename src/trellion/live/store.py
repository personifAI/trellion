# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

from __future__ import annotations

from .. import json_fast as json
import math
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import List, Tuple

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}

def _tokenize_list(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _bm25_scores(query_tokens: List[str], docs: List[List[str]]) -> List[float]:
    if not docs:
        return []
    if not query_tokens:
        return [0.0 for _ in docs]

    N = len(docs)
    avgdl = sum(len(doc) for doc in docs) / N if N else 0.0
    df: Counter[str] = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1

    idf = {
        term: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
        for term, freq in df.items()
    }

    k1 = 1.5
    b = 0.75
    scores: List[float] = []
    for doc in docs:
        tf = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            denom = tf[term] + k1 * (1 - b + b * (dl / avgdl if avgdl else 0.0))
            score += idf.get(term, 0.0) * (tf[term] * (k1 + 1)) / denom
        scores.append(score)

    return scores


@dataclass
class MemoryRecord:
    id: str
    scope: str
    role: str
    content: str
    created_at: float


class InMemoryStore:
    def __init__(self, scope: str, max_items: int = 200) -> None:
        self.scope = scope
        self.max_items = max_items
        self._items: List[MemoryRecord] = []

    def add(self, role: str, content: str) -> MemoryRecord:
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            scope=self.scope,
            role=role,
            content=content,
            created_at=time.time(),
        )
        self._items.append(record)
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]
        return record

    def items(self) -> List[MemoryRecord]:
        return list(self._items)

    def search(
        self, query: str, top_k: int, min_score: float, scoring: str = "jaccard"
    ) -> List[Tuple[MemoryRecord, float]]:
        if not query:
            return []
        if scoring == "bm25":
            docs = [_tokenize_list(item.content) for item in self._items]
            scores = _bm25_scores(_tokenize_list(query), docs)
            scored = [
                (item, score)
                for item, score in zip(self._items, scores)
                if score >= min_score
            ]
        else:
            q = _tokenize(query)
            scored = []
            for item in self._items:
                score = _jaccard(q, _tokenize(item.content))
                if score >= min_score:
                    scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class FileMemoryStore:
    def __init__(self, path: Path, scope: str = "global", max_items: int = 2000) -> None:
        self.path = path
        self.scope = scope
        self.max_items = max_items
        self._items: List[MemoryRecord] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return
        for line in data:
            try:
                payload = json.loads(line)
                self._items.append(
                    MemoryRecord(
                        id=payload.get("id", str(uuid.uuid4())),
                        scope=payload.get("scope", self.scope),
                        role=payload.get("role", "assistant"),
                        content=payload.get("content", ""),
                        created_at=float(payload.get("created_at", time.time())),
                    )
                )
            except Exception:
                continue

    def _persist_append(self, record: MemoryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "id": record.id,
                "scope": record.scope,
                "role": record.role,
                "content": record.content,
                "created_at": record.created_at,
            }
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _trim(self) -> None:
        if len(self._items) <= self.max_items:
            return
        self._items = self._items[-self.max_items :]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in self._items:
                handle.write(
                    json.dumps(
                        {
                            "id": record.id,
                            "scope": record.scope,
                            "role": record.role,
                            "content": record.content,
                            "created_at": record.created_at,
                        }
                    )
                    + "\n"
                )

    def add(self, role: str, content: str) -> MemoryRecord:
        record = MemoryRecord(
            id=str(uuid.uuid4()),
            scope=self.scope,
            role=role,
            content=content,
            created_at=time.time(),
        )
        self._items.append(record)
        self._persist_append(record)
        self._trim()
        return record

    def items(self) -> List[MemoryRecord]:
        return list(self._items)

    def search(
        self, query: str, top_k: int, min_score: float, scoring: str = "jaccard"
    ) -> List[Tuple[MemoryRecord, float]]:
        if not query:
            return []
        if scoring == "bm25":
            docs = [_tokenize_list(item.content) for item in self._items]
            scores = _bm25_scores(_tokenize_list(query), docs)
            scored = [
                (item, score)
                for item, score in zip(self._items, scores)
                if score >= min_score
            ]
        else:
            q = _tokenize(query)
            scored = []
            for item in self._items:
                score = _jaccard(q, _tokenize(item.content))
                if score >= min_score:
                    scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
