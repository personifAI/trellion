# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""One topic may not take every evidence slot.

The trading agents write a document per trade idea, all titled
``pai-trading / experience / exp-<id>``, and all with the same boilerplate
opening. By 2026-10-16 they were most of the knowledge base, so a general
question ("best practices") came back with five of them and nothing else:
the other topics were in the corpus but never reached the answer.

A *family* is a document's title without its last `` / `` segment
(``pai-trading / experience``); a title without one is its own family. At
most ``cap`` pages of a family are taken while other families still have
candidates. A page only displaces a better-scored one if it is close to the
best score (``floor``), so a question that really is about one topic still
gets that topic. Empty slots are backfilled from the capped pages, and the
result keeps relevance order.
"""

from __future__ import annotations

from typing import Callable, Optional


def topic_family(title: str, doc_id: str) -> str:
    """The family a document belongs to, from its title."""
    t = (title or "").strip()
    if " / " in t:
        return t.rsplit(" / ", 1)[0].strip().lower()
    return f"doc:{doc_id}"


def diversify(
    pages: list[dict],
    family_of: Callable[[dict], str],
    top_k: int,
    cap: Optional[int] = None,
    floor: float = 0.85,
) -> list[dict]:
    """Pick ``top_k`` of ``pages`` (sorted best first) with at most ``cap``
    per family while other families have candidates near the top score."""
    if top_k <= 0:
        return []
    if len(pages) <= top_k:
        return list(pages)
    cap = cap if cap is not None else max(2, (top_k + 1) // 2)
    best = float(pages[0].get("score", 0.0) or 0.0)
    counts: dict[str, int] = {}
    chosen: list[dict] = []
    held: list[dict] = []  # over their family's cap
    weak: list[dict] = []  # would displace a better page but too far below the best
    for page in pages:
        if len(chosen) == top_k:
            break
        fam = family_of(page)
        if counts.get(fam, 0) >= cap:
            held.append(page)
            continue
        score = float(page.get("score", 0.0) or 0.0)
        if held and score < floor * best:
            weak.append(page)
            continue
        counts[fam] = counts.get(fam, 0) + 1
        chosen.append(page)
    for page in held + weak:
        if len(chosen) == top_k:
            break
        chosen.append(page)
    order = {id(p): i for i, p in enumerate(pages)}
    chosen.sort(key=lambda p: order[id(p)])
    return chosen
