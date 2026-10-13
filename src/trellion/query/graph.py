# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""In-memory CKG representation loaded from the 5-column CSV.

Loads once, answers many queries. All lookups are O(1) dict hits or O(V+E)
graph traversals — zero LLM calls, zero vector store, zero network.
"""

from __future__ import annotations

import csv
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Concept:
    cid: int
    label: str
    dependencies: list
    taxonomy: str
    relation_type: str
    source_chunks: list = field(default_factory=list)


@dataclass
class CKG:
    concepts: dict = field(default_factory=dict)
    label_to_id: dict = field(default_factory=dict)
    forward: dict = field(default_factory=lambda: defaultdict(list))
    backward: dict = field(default_factory=lambda: defaultdict(list))
    taxonomy_members: dict = field(default_factory=lambda: defaultdict(list))

    @classmethod
    def load(cls, path):
        """Load a CKG CSV. Accepts str or Path."""
        path = Path(path)
        g = cls()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["ConceptID"])
                label = row["ConceptLabel"].strip()
                raw_deps = (row.get("Dependencies") or "").strip()
                deps = [int(x) for x in raw_deps.split("|") if x]
                tax = (row.get("TaxonomyID") or "UNASSIGNED").strip()
                rel = (row.get("RelationType") or "PREREQ").strip()
                raw_chunks = (row.get("SourceChunks") or "").strip()
                chunks = [c for c in raw_chunks.split("|") if c]

                concept = Concept(
                    cid=cid,
                    label=label,
                    dependencies=deps,
                    taxonomy=tax,
                    relation_type=rel,
                    source_chunks=chunks,
                )
                g.concepts[cid] = concept
                g.label_to_id[_norm(label)] = cid
                g.taxonomy_members[tax].append(cid)
                for d in deps:
                    g.forward[cid].append(d)
                    g.backward[d].append(cid)
        return g

    def find_by_label(self, label):
        cid = self.label_to_id.get(_norm(label))
        return self.concepts.get(cid) if cid is not None else None

    def direct_prerequisites(self, cid):
        return [self.concepts[d] for d in self.forward.get(cid, []) if d in self.concepts]

    def direct_dependents(self, cid):
        return [self.concepts[d] for d in self.backward.get(cid, []) if d in self.concepts]

    def bfs_path(self, src_cid, dst_cid, max_hops=5):
        """Shortest directed path from src -> dst (via forward edges).

        If no forward path, tries the reverse and flips the result so the
        returned list still reads src -> ... -> dst.
        """
        if src_cid == dst_cid:
            return [src_cid]
        parent = {src_cid: None}
        q = deque([(src_cid, 0)])
        while q:
            node, depth = q.popleft()
            if depth >= max_hops:
                continue
            for nxt in self.forward.get(node, []):
                if nxt in parent:
                    continue
                parent[nxt] = node
                if nxt == dst_cid:
                    return _reconstruct(parent, dst_cid)
                q.append((nxt, depth + 1))

        parent = {dst_cid: None}
        q = deque([(dst_cid, 0)])
        while q:
            node, depth = q.popleft()
            if depth >= max_hops:
                continue
            for nxt in self.forward.get(node, []):
                if nxt in parent:
                    continue
                parent[nxt] = node
                if nxt == src_cid:
                    return list(reversed(_reconstruct(parent, src_cid)))
                q.append((nxt, depth + 1))
        return None

    def dfs_ancestors(self, cid, max_hops=5):
        seen = set()
        stack = [(cid, 0)]
        while stack:
            node, depth = stack.pop()
            if depth >= max_hops:
                continue
            for nxt in self.forward.get(node, []):
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append((nxt, depth + 1))
        return sorted(seen)

    def by_taxonomy(self, tax):
        return [self.concepts[c] for c in self.taxonomy_members.get(tax, [])]

    def sources_for(self, cid):
        """Return the list of source chunk ids that produced this concept.

        Empty list if the CSV had no SourceChunks column or if the concept
        was loaded from an older 5-column file.
        """
        c = self.concepts.get(cid)
        return list(c.source_chunks) if c is not None else []


def _reconstruct(parent, target):
    out = [target]
    cur = parent.get(target)
    while cur is not None:
        out.append(cur)
        cur = parent.get(cur)
    return list(reversed(out))


def _norm(s):
    return " ".join(s.strip().lower().split())
