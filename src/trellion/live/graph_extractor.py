# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""LLM-based entity extraction from PageIndex pages.

Extracts named entities from page text using an Ollama/vLLM-compatible
LLM endpoint, then stores them in the PageIndex graph via
PageIndexStore.upsert_entity() and insert_mention().
"""

from __future__ import annotations

import asyncio
# The fast shim is the module-wide `json` (orjson-backed, no incremental
# decoder). Salvage needs `raw_decode`, which only the stdlib has, so it is
# imported alongside rather than in place of it.
import json as _stdjson
from .. import json_fast as json
import re
from typing import Any, Optional

import httpx
import structlog

from trellion.live._tls import ssl_verify, client_cert
from trellion.live.entity_quality import DropTally, check_entity
from trellion.live.gateway_auth import GatewayHmacAuth
from trellion.live.llm_throttle import LLMBreakerOpen, LLMUnavailable, get_llm_throttle
from trellion.live.pageindex_store import PageIndexStore

log = structlog.get_logger(__name__)

ENTITY_TYPES = [
    "component",
    "parameter",
    "material",
    "variable",
    "process",
    "concept",
    "person",
    "organization",
    "location",
]

EXTRACTION_PROMPT = """\
Extract named entities from the text below.
Return ONLY a JSON array. Each element MUST be a JSON object with exactly these keys:
  "name"    - canonical name of the entity (string, lowercase)
  "type"    - one of: {types}
  "aliases" - list of alternate names (may be empty list [])
  "quote"   - short verbatim quote from the text where the entity appears (string)

Rules:
- Be thorough; extract every meaningful entity (people, organizations, products, concepts, parameters, materials, processes, locations).
- Use lowercase for names.
- The "name" key is required for every entity.
- Do NOT extract mathematical variables, symbols or indices (a, x, d, theta, x_t, f(x)), single letters, bare numbers, quantities or amounts (0.003, 12, $1,200), or pronouns and function words. If a symbol is defined in the text, extract the thing it stands for (e.g. "learning rate"), not the symbol.
- Do NOT wrap the JSON in markdown fences. Do NOT add prose before or after the array.
- If you genuinely find no entities, return an empty array: []

Example output for the text "The Apollo 11 mission landed Neil Armstrong on the moon in 1969.":
[
  {{"name": "apollo 11", "type": "concept", "aliases": [], "quote": "The Apollo 11 mission"}},
  {{"name": "neil armstrong", "type": "person", "aliases": [], "quote": "landed Neil Armstrong on the moon"}},
  {{"name": "moon", "type": "location", "aliases": [], "quote": "landed Neil Armstrong on the moon"}}
]

Text:
{text}
"""


def _normalize_name(name: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    return re.sub(r"\s+", " ", name.strip().lower())


def _parse_json_array(raw: str) -> list[dict]:
    """Best-effort extraction of a JSON array from LLM output."""
    # Strip markdown code fences if the model wraps them anyway
    cleaned = re.sub(r"```(?:json)?\s*", "", raw)
    cleaned = cleaned.strip().rstrip("`")

    # Try parsing the whole thing first
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: find the first [ ... ] block
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Last resort: the array never closed. Salvage the elements that DID
    # complete.
    #
    # Both strategies above are all-or-nothing, and a truncated response has
    # no closing bracket — so a completion cut off at the token cap threw
    # away every entity in it, however many were whole. On the live
    # controller that was the normal case, not the edge: extraction ran
    # continuously, responses clustered just under the 2048-token ceiling
    # (6,720 characters at the top), and `parsed_elements=0` came back for
    # page after page whose preview plainly began
    # `[{"name": "jia deng", "type": "person", ...`.
    #
    # The graph showed it: 10,039 of 11,619 edges left as generic
    # RELATED_TO, classification stuck at 14%, entity count moving by
    # single digits over hours of LLM calls. Every one of those calls was
    # paid for and discarded.
    return _salvage_objects(cleaned)


def _salvage_objects(cleaned: str) -> list[dict]:
    """Every complete JSON object in a possibly-unterminated array."""
    start = cleaned.find("[")
    if start < 0:
        return []
    decoder = _stdjson.JSONDecoder()
    out: list[dict] = []
    i = start + 1
    n = len(cleaned)
    while i < n:
        while i < n and cleaned[i] in " \t\r\n,":
            i += 1
        if i >= n or cleaned[i] == "]":
            break
        try:
            obj, i = decoder.raw_decode(cleaned, i)
        except ValueError:
            # The tail is the half-written element the cap cut. Everything
            # before it is intact, which is the whole point.
            break
        out.append(obj)
    return out


def _closes_cleanly(raw: str) -> bool:
    """Did the model finish the array it started?"""
    tail = re.sub(r"```\s*$", "", (raw or "").strip()).rstrip("` \t\r\n")
    return tail.endswith("]")


class EntityExtractor:
    """Extracts entities from page text using an LLM and stores them."""

    def __init__(
        self,
        store: PageIndexStore,
        llm_url: str = "https://gateway:8080",
        model: str = "llama3",
        messaging: Optional[Any] = None,
    ) -> None:
        self.store = store
        self.llm_url = llm_url.rstrip("/")
        self.model = model
        self.messaging = messaging  # Optional ClusterMessaging
        self._running = False
        self._task: Optional[asyncio.Task] = None
        # Signed per request: llm_url is the gateway's /v1 surface (mesh-
        # routed), which authenticates every caller. A plain backend URL
        # ignores the header, so the old direct wiring still works.
        self._http = httpx.AsyncClient(
            timeout=300.0, verify=ssl_verify(), cert=client_cert(), auth=GatewayHmacAuth(),
        )
        # Connection-error log throttling. The graph extraction queue can
        # have hundreds of pages waiting; if vLLM is still warming up at
        # gateway startup, this flooded the logs at ERROR level. Now we
        # log first failure + every Nth subsequent + a "recovered" line.
        self._connect_failures = 0
        self._connect_failure_logged_at = 0  # count when last logged

    # ------------------------------------------------------------------
    # LLM interaction
    # ------------------------------------------------------------------

    async def _llm_generate(self, prompt: str) -> str:
        """Call LLM via OpenAI-compatible /v1/chat/completions (works with vLLM, Ollama, etc.)."""
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048,
            # Disable Qwen3 thinking mode for structured extraction
            "chat_template_kwargs": {"enable_thinking": False},
        }
        async def _post() -> Any:
            resp = await self._http.post(
                f"{self.llm_url}/v1/chat/completions",
                json=body,
            )
            resp.raise_for_status()
            return resp.json()

        # One throttle + per-model breaker per process (llm_throttle): backoff
        # on 429/502/503/504, no call at all while the model is known absent.
        data = await get_llm_throttle().call(self.model, _post, what="entity_extraction")
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            # Strip <think> blocks if the model still produced them
            return re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        return ""

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    async def extract_entities(self, page_id: str, text: str) -> Optional[list[dict]]:
        """Use LLM to extract entities from a page's text.

        Returns list of ``{name, type, aliases, quote}`` — an EMPTY list is a
        real answer (the page names nothing) and the caller records the page
        as extracted. Returns ``None`` when the model could not be asked
        (unreachable, HTTP error): that is not an answer, and the page must
        be retried. The two used to be the same ``[]``.
        """
        if not text or not text.strip():
            return []

        prompt = EXTRACTION_PROMPT.format(
            types=", ".join(ENTITY_TYPES),
            text=text[:8000],  # guard against very long pages
        )

        try:
            raw = await self._llm_generate(prompt)
            # Recovery path — if we were in a connection-error streak, note
            # that we're back so operators see the resolution.
            if self._connect_failures > 0:
                log.info(
                    "entity_extraction.llm_recovered",
                    after_failures=self._connect_failures,
                )
                self._connect_failures = 0
                self._connect_failure_logged_at = 0
        except LLMUnavailable as exc:
            # Expected under outage: the throttle logged the breaker opening
            # once. The page goes back to the queue; no per-page line, no
            # traceback.
            self._connect_failures += 1
            if not isinstance(exc, LLMBreakerOpen):
                log.debug(
                    "entity_extraction.llm_unavailable",
                    page_id=page_id,
                    model=exc.model,
                    reason=exc.reason,
                )
            return None
        except httpx.HTTPError as exc:
            # Non-connect HTTP errors (4xx/5xx, timeouts) — keep ERROR
            # since they're more likely to indicate a real problem.
            log.error(
                "entity_extraction.llm_error",
                page_id=page_id,
                error_type=type(exc).__name__,
                error=repr(exc),
            )
            return None

        entities = _parse_json_array(raw)
        # A response that parses only by salvage was cut off at the token
        # cap. Say so: the alternative is an operator seeing a healthy count
        # and never learning the tail of every page is being dropped.
        truncated = bool(entities) and not _closes_cleanly(raw)
        valid: list[dict] = []
        rejected_no_name = 0
        junk = DropTally()
        for ent in entities:
            name = ent.get("name") if isinstance(ent, dict) else None
            etype = (ent.get("type", "concept") if isinstance(ent, dict) else "concept")
            if not name:
                rejected_no_name += 1
                continue
            etype = etype.lower().strip()
            if etype not in ENTITY_TYPES:
                etype = "concept"
            # Validate after parsing, on the name as the model wrote it. The
            # prompt forbids symbols, but the prompt is a request; this is
            # the gate. A dropped entity gets no row and no mention, so no
            # edge can later attach to it.
            reason = check_entity(name, etype)
            if reason:
                junk.add(name, reason)
                continue
            valid.append(
                {
                    "name": _normalize_name(name),
                    "type": etype,
                    "aliases": [
                        _normalize_name(a)
                        for a in (ent.get("aliases") or [])
                        if isinstance(a, str) and a.strip()
                    ],
                    "quote": (ent.get("quote") or "")[:500],
                }
            )
        junk.log("entity_extraction.junk_dropped", page_id=page_id, kept=len(valid))
        # When count=0, surface what the model returned so we can tell
        # whether it's (a) genuinely an empty page, (b) the model returned
        # `[]` despite content, (c) prose preamble we couldn't parse, or
        # (d) wrong-shape elements (no "name" key). Truncated to keep logs
        # readable; only fires on the empty path so it doesn't flood.
        if not valid:
            log.warning(
                "entity_extraction.empty_result",
                page_id=page_id,
                text_len=len(text or ""),
                raw_len=len(raw or ""),
                raw_preview=(raw or "")[:300],
                parsed_elements=len(entities),
                rejected_no_name=rejected_no_name,
            )
        if truncated:
            log.warning(
                "entity_extraction.truncated",
                page_id=page_id,
                salvaged=len(valid),
                raw_len=len(raw or ""),
                detail=(
                    "the model hit its token cap mid-array; the completed "
                    "entities were salvaged and the tail of the page was lost"
                ),
            )
        log.info(
            "entity_extraction.done",
            page_id=page_id,
            count=len(valid),
        )
        return valid

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    async def process_page(self, page_id: str) -> list[str]:
        """Extract entities from a page and persist them.

        Returns list of entity_ids created/updated.
        """
        page = await self.store.get_page(page_id)
        if not page:
            log.warning("entity_extraction.page_not_found", page_id=page_id)
            return []

        # Model known unavailable (breaker open): leave the page unclaimed
        # and unextracted for a later pass instead of claim/fail/release.
        throttle = get_llm_throttle()
        if throttle.blocked_for(self.model) is not None:
            throttle.defer(self.model)
            return []

        # Claim before asking the model. Two ML containers run this loop
        # over the same table; without the claim each page went through
        # the chat model twice and every mention was written twice.
        if not await self.store.claim_page_for_extraction(page_id):
            log.debug("entity_extraction.page_taken", page_id=page_id)
            return []

        entities = await self.extract_entities(page_id, page["text"])
        if entities is None:
            # The model could not be asked. Not an answer: hand the page
            # back and let the next cycle try again.
            await self.store.release_extraction_claim(page_id)
            return []
        entity_ids: list[str] = []

        for ent in entities:
            entity_id = await self.store.upsert_entity(
                canonical_name=ent["name"],
                type=ent["type"],
                aliases=ent["aliases"] or None,
            )
            await self.store.insert_mention(
                entity_id=entity_id,
                page_id=page_id,
                quote=ent["quote"] or None,
            )
            entity_ids.append(entity_id)

            # Publish per-entity event if messaging available
            if self.messaging and self.messaging.is_connected:
                await self.messaging.publish_event(
                    "graph.entity.extracted",
                    {
                        "entity_id": entity_id,
                        "page_id": page_id,
                        "doc_id": page["doc_id"],
                    },
                )

        # Done — including when the model found nothing. Until this was
        # recorded, "found nothing" was indistinguishable from "never
        # asked", and the same empty pages were re-extracted every cycle.
        await self.store.mark_page_extracted(page_id)
        return entity_ids

    async def process_doc(self, doc_id: str) -> int:
        """Extract entities from every page of a document.

        Returns total number of entity mentions created.
        """
        pages = await self.store.get_doc_pages(doc_id)
        total = 0
        for page in pages:
            ids = await self.process_page(page["page_id"])
            total += len(ids)
        log.info("entity_extraction.doc_done", doc_id=doc_id, mentions=total)
        return total

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def start(self, interval: float = 10.0) -> None:
        """Start background loop that listens for ``ingest.doc.parsed`` events."""
        if self._running:
            return
        self._running = True

        # Subscribe to NATS events if messaging is available
        if self.messaging and self.messaging.is_connected:
            await self.messaging.subscribe_events(
                "ingest.doc.parsed",
                self._on_doc_parsed,
                durable_name="entity_extractor",
            )
            log.info("entity_extractor.subscribed", nats_subject="ingest.doc.parsed")

        self._task = asyncio.create_task(self._loop(interval))
        log.info("entity_extractor.started", interval=interval)

    async def stop(self) -> None:
        """Gracefully stop the background loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self._http.aclose()
        log.info("entity_extractor.stopped")

    async def _on_doc_parsed(self, event: dict) -> None:
        """Handle an ``ingest.doc.parsed`` NATS event."""
        doc_id = event.get("doc_id")
        if doc_id:
            log.info("entity_extractor.event_received", doc_id=doc_id)
            await self.process_doc(doc_id)

    async def _loop(self, interval: float) -> None:
        """Polling fallback: find pages that have not been extracted yet."""
        try:
            # Containers restarted together must not begin together.
            await get_llm_throttle().startup_delay()
        except asyncio.CancelledError:
            return
        while self._running:
            try:
                await asyncio.sleep(interval)
                throttle = get_llm_throttle()
                if throttle.blocked_for(self.model) is not None:
                    continue  # breaker open: pages stay queued, no DB or LLM work
                pages = await self._get_unextracted_pages()
                for i, page in enumerate(pages):
                    if not self._running:
                        break
                    if throttle.blocked_for(self.model) is not None:
                        throttle.defer(self.model, len(pages) - i)
                        break
                    await self.process_page(page["page_id"])
            except asyncio.CancelledError:
                break
            except Exception:
                log.exception("entity_extractor.loop_error")
                await asyncio.sleep(interval)

    async def _get_unextracted_pages(self, limit: int = 50) -> list[dict]:
        """Pages never extracted and not currently claimed by another
        extractor (a claim older than fifteen minutes belongs to a worker
        that is gone). Selecting on ``extracted_at`` is what makes the loop
        converge: a page that yielded nothing is still done."""
        from sqlalchemy import func, select, text as _text
        from ..db.models import Page
        _page = Page.__table__

        async with self.store._engine.connect() as conn:
            result = await conn.execute(
                select(_page.c.page_id, _page.c.text)
                .where(_page.c.extracted_at.is_(None))
                .where(
                    (_page.c.extraction_claimed_at.is_(None))
                    | (_page.c.extraction_claimed_at < func.now() - _text("interval '15 minutes'"))
                )
                .order_by(_page.c.page_id)
                .limit(limit)
            )
            rows = result.fetchall()
            return [{"page_id": str(r[0]), "text": r[1]} for r in rows]
