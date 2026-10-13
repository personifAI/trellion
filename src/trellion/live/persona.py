# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Persona adapter -- thin layer over persona-service data for RAG biasing.

The canonical persona lives in the persona service (SQLAlchemy + Pydantic).  This module
derives a ``PersonaLens`` (retrieval-biasing view) from the persona service fields
without duplicating schema or storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import structlog

log = structlog.get_logger(__name__)

# Entity types recognized by the graph extractor
_ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "TECHNOLOGY", "CONCEPT",
    "LOCATION", "EVENT", "PRODUCT", "METRIC", "PROCESS",
]

# Keywords in traits/goals that map to entity-type affinity
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "TECHNOLOGY": ["technical", "engineering", "software", "hardware", "code", "system"],
    "ORGANIZATION": ["business", "corporate", "team", "management", "leadership"],
    "CONCEPT": ["research", "theory", "academic", "philosophy", "abstract"],
    "PERSON": ["social", "people", "relationship", "networking", "interpersonal"],
    "PRODUCT": ["product", "design", "user", "customer", "market"],
    "METRIC": ["data", "analytics", "quantitative", "measurement", "performance"],
    "PROCESS": ["process", "workflow", "methodology", "agile", "operational"],
}

# Communication style to strategy mapping
_STYLE_STRATEGY: dict[str, str] = {
    "analytical": "graph_first",
    "technical": "graph_first",
    "precise": "graph_first",
    "conversational": "lookup",
    "casual": "lookup",
    "friendly": "lookup",
}


@dataclass
class PersonaLens:
    """Retrieval-biasing view derived from a persona service persona."""

    persona_id: str
    name: str
    domain_weights: dict[str, float] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    preferred_strategy: str = "hybrid"
    system_prompt_context: str = ""


class PersonaAdapter:
    """Converts persona-service data into a PersonaLens for RAG use."""

    @staticmethod
    def from_noo_core(persona_dict: dict) -> PersonaLens:
        """Derive a PersonaLens from a persona service persona row/dict.

        Expected keys (all optional except persona_id, name):
            persona_id, name, personality_traits (list[str]),
            goals (list[str]), challenges (list[str]),
            persona_background (list[str]), communication_style (str),
            prompt_set (list[dict with role/content])
        """
        pid = str(persona_dict.get("persona_id", ""))
        name = persona_dict.get("name", "")

        # --- domain_weights: scan traits + goals for keyword matches ---
        traits = persona_dict.get("personality_traits") or []
        goals = persona_dict.get("goals") or []
        if isinstance(goals, str):
            goals = [goals]
        if isinstance(traits, str):
            traits = [traits]

        combined_text = " ".join(traits + goals).lower()
        weights: dict[str, float] = {}
        for etype, keywords in _DOMAIN_KEYWORDS.items():
            score = sum(1.0 for kw in keywords if kw in combined_text)
            if score > 0:
                weights[etype] = min(score / len(keywords), 1.0)

        # --- constraints: from challenges + background ---
        challenges = persona_dict.get("challenges") or []
        background = persona_dict.get("persona_background") or []
        if isinstance(challenges, str):
            challenges = [challenges]
        if isinstance(background, str):
            background = [background]
        constraints = [c for c in challenges + background if isinstance(c, str) and c.strip()]

        # --- preferred_strategy: from communication_style ---
        style = (persona_dict.get("communication_style") or "").lower().strip()
        strategy = "hybrid"
        for keyword, strat in _STYLE_STRATEGY.items():
            if keyword in style:
                strategy = strat
                break

        # --- system_prompt_context: from prompt_set ---
        prompt_set = persona_dict.get("prompt_set") or []
        prompt_parts: list[str] = []
        for entry in prompt_set:
            if isinstance(entry, dict):
                role = entry.get("role", "")
                content = entry.get("content", "")
                if content:
                    prompt_parts.append(f"[{role}] {content}" if role else content)
        system_prompt_context = "\n".join(prompt_parts)

        return PersonaLens(
            persona_id=pid,
            name=name,
            domain_weights=weights,
            constraints=constraints,
            preferred_strategy=strategy,
            system_prompt_context=system_prompt_context,
        )

    @staticmethod
    async def load_from_db(persona_id: str) -> Optional[PersonaLens]:
        """Load a persona directly from the persona service database.

        Imports noo_core as a Python module (must be installed in the venv).
        Returns None if the persona service is not installed or the persona is not found.
        """
        try:
            from noo_core.db import get_session_factory
            from noo_core.db.repositories import PersonaRepository
        except ImportError:
            log.warning("persona_adapter.noo_core_not_installed")
            return None

        try:
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = PersonaRepository(session)
                persona = await repo.get_by_uuid(persona_id)
                if not persona:
                    log.warning(
                        "persona_adapter.not_found",
                        persona_id=persona_id,
                    )
                    return None

                # Convert ORM object to dict for from_noo_core()
                data = {
                    "persona_id": persona.persona_id,
                    "name": persona.name,
                    "personality_traits": persona.personality_traits,
                    "goals": persona.goals,
                    "challenges": persona.challenges,
                    "communication_style": persona.communication_style,
                    "persona_background": persona.persona_background,
                    "prompt_set": persona.prompt_set,
                }
                return PersonaAdapter.from_noo_core(data)
        except Exception:
            log.exception("persona_adapter.load_error", persona_id=persona_id)
            return None

    @staticmethod
    def apply_to_query(lens: PersonaLens, query: str) -> str:
        """Augment a query string with persona context for better retrieval."""
        parts = [query]

        if lens.constraints:
            context = "; ".join(lens.constraints[:3])
            parts.append(f"(context: {context})")

        if lens.system_prompt_context:
            # Take first 200 chars of prompt context
            snippet = lens.system_prompt_context[:200]
            parts.append(f"(persona: {snippet})")

        return " ".join(parts)

    @staticmethod
    def bias_entity_scores(
        lens: PersonaLens,
        entities: list[dict],
    ) -> list[dict]:
        """Rerank entities by domain_weights from the persona lens.

        Entities whose ``entity_type`` matches a high-weight domain get
        a score boost.  The list is re-sorted descending by adjusted score.
        """
        if not lens.domain_weights:
            return entities

        boosted: list[dict] = []
        for ent in entities:
            etype = ent.get("entity_type", "")
            weight = lens.domain_weights.get(etype, 0.0)
            # Boost: multiply original relevance by (1 + weight)
            original_score = ent.get("score", 1.0)
            adjusted = original_score * (1.0 + weight)
            boosted.append({**ent, "score": adjusted})

        boosted.sort(key=lambda e: e.get("score", 0), reverse=True)
        return boosted
