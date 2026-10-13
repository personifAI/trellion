# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Query-side answer routing for the T1-T5 taxonomy."""

from __future__ import annotations

import re
from dataclasses import dataclass, field



@dataclass
class Answer:
    query_type: str
    content: str
    concept_ids: list = field(default_factory=list)
    retrieved_tokens: int = 0


def answer_query(ckg, query, *, max_hops=5):
    qt = classify(query)
    if qt == "T1":
        return _t1(ckg, query)
    if qt == "T2":
        return _t2(ckg, query)
    if qt == "T3":
        return _t3(ckg, query, max_hops)
    if qt == "T4":
        return _t4(ckg, query)
    if qt == "T5":
        return _t5(ckg, query, max_hops)
    return Answer(query_type="UNKNOWN", content="Could not classify query.")


_T2_PAT = re.compile(r"(prereq|depend|require|need|before)", re.IGNORECASE)
_T3_PAT = re.compile(r"\b(chain|path|from .+ to |route)\b", re.IGNORECASE)
_T4_PAT = re.compile(r"\b(list|all .+|every)\b", re.IGNORECASE)
_T5_PAT = re.compile(r"\b(relate|relation|connection|compare)\b", re.IGNORECASE)


def classify(query):
    if _T3_PAT.search(query):
        return "T3"
    if _T2_PAT.search(query):
        return "T2"
    if _T4_PAT.search(query):
        return "T4"
    if _T5_PAT.search(query):
        return "T5"
    return "T1"


def _t1(ckg, query):
    concept = _best_label_match(ckg, query)
    if not concept:
        return Answer(query_type="T1", content="No matching concept found.")
    deps = ckg.direct_prerequisites(concept.cid)
    tokens = _tok_cost(concept.label) + _tok_cost(concept.taxonomy) + sum(_tok_cost(d.label) for d in deps)
    return Answer(
        query_type="T1",
        content=(
            f"{concept.label} (taxonomy={concept.taxonomy}, relation={concept.relation_type}). "
            f"Direct dependencies: {[d.label for d in deps] or 'none'}."
        ),
        concept_ids=[concept.cid] + [d.cid for d in deps],
        retrieved_tokens=tokens,
    )


def _t2(ckg, query):
    concept = _best_label_match(ckg, query)
    if not concept:
        return Answer(query_type="T2", content="No matching concept found.")
    deps = ckg.direct_prerequisites(concept.cid)
    tokens = sum(_tok_cost(d.label) for d in deps)
    return Answer(
        query_type="T2",
        content="; ".join(d.label for d in deps) or "none",
        concept_ids=[d.cid for d in deps],
        retrieved_tokens=tokens,
    )


def _t3(ckg, query, max_hops):
    pair = _two_concepts(ckg, query)
    if not pair:
        return Answer(query_type="T3", content="Could not identify both endpoints.")
    src, dst = pair
    path = ckg.bfs_path(src.cid, dst.cid, max_hops=max_hops)
    if not path:
        return Answer(
            query_type="T3",
            content=f"No path found between {src.label} and {dst.label}.",
            concept_ids=[src.cid, dst.cid],
        )
    labels = [ckg.concepts[c].label for c in path]
    return Answer(
        query_type="T3",
        content=" -> ".join(labels),
        concept_ids=path,
        retrieved_tokens=sum(_tok_cost(lbl) for lbl in labels),
    )


def _t4(ckg, query):
    tax = _extract_taxonomy(ckg, query)
    if not tax:
        return Answer(query_type="T4", content="Could not identify a taxonomy label.")
    members = ckg.by_taxonomy(tax)
    labels = [m.label for m in members]
    return Answer(
        query_type="T4",
        content=", ".join(labels),
        concept_ids=[m.cid for m in members],
        retrieved_tokens=sum(_tok_cost(lbl) for lbl in labels),
    )


def _t5(ckg, query, max_hops):
    pair = _two_concepts(ckg, query)
    if not pair:
        return Answer(query_type="T5", content="Could not identify both endpoints.")
    a, b = pair
    path = ckg.bfs_path(a.cid, b.cid, max_hops=max_hops) or []
    a_nei = {c.cid for c in ckg.direct_prerequisites(a.cid) + ckg.direct_dependents(a.cid)}
    b_nei = {c.cid for c in ckg.direct_prerequisites(b.cid) + ckg.direct_dependents(b.cid)}
    shared = a_nei & b_nei
    path_labels = [ckg.concepts[c].label for c in path]
    shared_labels = [ckg.concepts[c].label for c in sorted(shared)]
    content = []
    if path_labels:
        content.append("Path: " + " -> ".join(path_labels))
    if shared_labels:
        content.append("Shared neighbors: " + ", ".join(shared_labels))
    return Answer(
        query_type="T5",
        content="\n".join(content) or "No structural relation found.",
        concept_ids=list({*path, *shared, a.cid, b.cid}),
        retrieved_tokens=sum(_tok_cost(lbl) for lbl in path_labels + shared_labels),
    )


def _best_label_match(ckg, query):
    norm_labels = sorted(ckg.label_to_id.keys(), key=len, reverse=True)
    q = " ".join(query.lower().split())
    for label in norm_labels:
        if label and label in q:
            cid = ckg.label_to_id[label]
            return ckg.concepts[cid]
    return None


def _two_concepts(ckg, query):
    q = " ".join(query.lower().split())
    norm_labels = sorted(ckg.label_to_id.keys(), key=len, reverse=True)
    hits = []
    taken = []
    cursor = 0
    while cursor < len(q):
        best = None
        for label in norm_labels:
            idx = q.find(label, cursor)
            if idx < 0:
                continue
            if any(not (idx + len(label) <= ts or idx >= te) for ts, te in taken):
                continue
            if best is None or idx < best[0] or (idx == best[0] and len(label) > best[1] - best[0]):
                best = (idx, idx + len(label), ckg.label_to_id[label])
        if best is None:
            break
        hits.append(best)
        taken.append((best[0], best[1]))
        cursor = best[1]
        if len(hits) == 2:
            break
    if len(hits) < 2:
        return None
    first = ckg.concepts[hits[0][2]]
    second = ckg.concepts[hits[1][2]]
    if first.cid == second.cid:
        return None
    return first, second


def _extract_taxonomy(ckg, query):
    q_upper = query.upper()
    for tax in sorted(ckg.taxonomy_members.keys(), key=len, reverse=True):
        if tax.upper() in q_upper:
            return tax
    return None


def _tok_cost(s):
    return max(1, len(s.split()))
