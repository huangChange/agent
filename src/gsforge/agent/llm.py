"""Claude API access for the agent: one place for model choice, streaming, caching, fallbacks,
retries, concurrency limits and cost accounting.

* Streaming + get_final_message(): long tool inputs (generated code) never hit HTTP timeouts.
* eager_input_streaming on client tools: inputs are validated by our pydantic models before running.
* Top-level automatic prompt caching: tools + system prompt are byte-stable across turns and jobs.
* Server-side fallback ("default" routing) on the Claude API: a classifier decline is retried on
  the recommended fallback model inside the same call. Disable via llm.server_side_fallback on
  platforms that do not support it.
* History is append-only (we never edit earlier turns), as required for thinking-block replay.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import anthropic

from ..core.config import LLMConfig

log = logging.getLogger("gsforge.llm")
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    requests: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def add(self, usage: Any, model: str) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.requests += 1
        self.by_model[model] = self.by_model.get(model, 0) + 1

    def cost_usd(self, prices: dict[str, tuple[float, float, float, float]], model: str) -> float:
        p_in, p_out, p_read, p_write = prices.get(model, (0.0, 0.0, 0.0, 0.0))
        return (self.input_tokens * p_in + self.output_tokens * p_out + self.cache_read_tokens * p_read
                + self.cache_write_tokens * p_write) / 1e6


class LLM(Protocol):
    """What the agent loop needs. Tests substitute a scripted fake."""

    async def turn(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                   model: str | None = None, effort: str | None = None) -> Any: ...

    async def structured(self, *, system: str, content: list[dict[str, Any]], schema: dict[str, Any],
                         model: str | None = None, effort: str | None = None) -> tuple[str, Any]: ...


class ClaudeLLM:
    def __init__(self, cfg: LLMConfig, concurrency: int = 8, client: anthropic.AsyncAnthropic | None = None):
        self.cfg = cfg
        self.client = client or anthropic.AsyncAnthropic(max_retries=cfg.max_retries, timeout=cfg.request_timeout_s)
        self._sem = asyncio.Semaphore(concurrency)

    def _fallback_kwargs(self) -> dict[str, Any]:
        if not self.cfg.server_side_fallback:
            return {}
        return {"betas": [FALLBACK_BETA], "fallbacks": "default"}

    async def turn(self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
                   model: str | None = None, effort: str | None = None) -> Any:
        model = model or self.cfg.model
        tools = [{**t, "eager_input_streaming": True} for t in tools]
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=messages,
            tools=tools,
            thinking={"type": "adaptive"},
            output_config={"effort": effort or self.cfg.effort},
            cache_control={"type": "ephemeral"},
        )
        kwargs.update(self._fallback_kwargs())
        api = self.client.beta.messages if "betas" in kwargs else self.client.messages
        attempts = 0
        while True:
            attempts += 1
            try:
                async with self._sem, api.stream(**kwargs) as stream:
                    return await stream.get_final_message()
            except ValueError as e:
                # The streamed tool input was not parseable JSON; there is no tool_use id to answer,
                # so the only correct recovery is to re-issue the request.
                if attempts >= 3:
                    raise
                log.warning("unparseable streamed tool input, retrying request: %s", e)

    async def structured(self, *, system: str, content: list[dict[str, Any]], schema: dict[str, Any],
                         model: str | None = None, effort: str | None = None) -> tuple[str, Any]:
        """One-shot JSON-schema constrained call (critic / classifiers). Returns (JSON text, usage)."""
        model = model or self.cfg.critic_model
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": effort or self.cfg.critic_effort,
                           "format": {"type": "json_schema", "schema": schema}},
        )
        kwargs.update(self._fallback_kwargs())
        api = self.client.beta.messages if "betas" in kwargs else self.client.messages
        async with self._sem:
            resp = await api.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model declined the request: {getattr(resp, 'stop_details', None)}")
        return next(b.text for b in resp.content if b.type == "text"), resp.usage
