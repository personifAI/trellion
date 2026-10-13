# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026 PersonifAI LLC

"""Thin async LLM wrapper around the OpenAI client.

Works against:
  - local vLLM (base_url=http://localhost:8000/v1)
  - HuggingFace TGI behind OpenAI shim
  - real OpenAI
  - Anthropic via an OpenAI-compatible proxy

Concurrency is bounded by a semaphore so we don't melt the vLLM server.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from openai import AsyncOpenAI

from trellion.common.config import LLMConfig
from trellion.live.llm_throttle import LLMUnavailable, get_llm_throttle

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_s,
            # The shared throttle owns retry/backoff; the SDK's own silent
            # retries would multiply every attempt.
            max_retries=0,
        )
        self._sem = asyncio.Semaphore(cfg.max_concurrent)
        # Lightweight running counters so callers can snapshot usage at
        # stage boundaries without rewriting every call site.
        self.calls: int = 0
        self.tokens_in: int = 0
        self.tokens_out: int = 0

    def snapshot(self) -> dict[str, int]:
        """Return current cumulative usage counters."""
        return {
            "calls": self.calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
        }

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async def _create() -> Any:
            async with self._sem:
                return await self._client.chat.completions.create(
                    model=self.cfg.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature if temperature is not None else self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    response_format=response_format,  # type: ignore[arg-type]
                )

        resp = await self._call_throttled(_create)
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.tokens_in += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.tokens_out += int(getattr(usage, "completion_tokens", 0) or 0)
        self.calls += 1
        return content

    async def _call_throttled(self, fn: Any) -> Any:
        """Run one request through the shared throttle + per-model breaker.

        A build cannot skip a chunk and call the graph done, so while the
        model's breaker is open this WAITS for the next probe slot (up to
        ``TRELLION_LLM_BUILD_WAIT_S``, default 1800 s) and then raises
        :class:`LLMUnavailable`: the build fails loudly instead of writing
        an empty extraction. In-flight is bounded by ``cfg.max_concurrent``
        (the build's own knob), so the interactive-style bound is skipped.
        """
        throttle = get_llm_throttle()
        model = self.cfg.model
        try:
            budget = float(os.environ.get("TRELLION_LLM_BUILD_WAIT_S", 1800))
        except ValueError:
            budget = 1800.0
        waited = 0.0
        while True:
            try:
                return await throttle.call(model, fn, what="build_extraction", bounded=False)
            except LLMUnavailable:
                if not throttle.is_open(model):
                    raise  # retries exhausted, model not known absent: fail loudly
                wait = throttle.blocked_for(model) or 1.0
                if waited + wait > budget:
                    raise
                await asyncio.sleep(wait)
                waited += wait

    async def complete_batch(
        self, prompts: list[str], *, system: str | None = None
    ) -> list[str]:
        """Fan out concurrently; internal semaphore caps throughput."""
        return await asyncio.gather(*(self.complete(p, system=system) for p in prompts))
