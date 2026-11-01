"""ToolRunner: validated, cached, timed tool invocation on top of a Backend.

Cache key = (tool name, tool version, params with artifacts reduced to their sha256). Because
artifacts are content-addressed, a re-run with identical inputs is a lookup, a change upstream
naturally invalidates everything downstream, and resumed jobs skip finished work for free.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import BaseModel

from ..core.config import Settings
from ..core.hashing import stable_key
from ..storage.artifacts import ArtifactStore
from ..tools.base import ToolResult
from ..tools.registry import get_tool
from .backends import Backend

log = logging.getLogger("gsforge.runner")


def _cache_view(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "sha256" in obj and "kind" in obj and "uri" in obj:
            return {"sha256": obj["sha256"]}
        return {k: _cache_view(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_cache_view(v) for v in obj]
    return obj


class ToolCallRecord(BaseModel):
    tool: str
    version: str
    params: dict[str, Any]
    result: ToolResult | None = None
    error: str | None = None
    cache_hit: bool = False
    seconds: float = 0.0


class ToolRunner:
    def __init__(self, store: ArtifactStore, settings: Settings, backend: Backend):
        self.store = store
        self.settings = settings
        self.backend = backend
        self._settings_json = settings.model_dump(mode="json")

    async def run(self, name: str, params: dict[str, Any] | BaseModel, *, job_id: str = "adhoc",
                  instance_id: str = "adhoc", use_cache: bool = True) -> ToolCallRecord:
        spec = get_tool(name)
        model = params if isinstance(params, BaseModel) else spec.params_model.model_validate(params)
        params_json = model.model_dump(mode="json")
        key = stable_key(spec.name, spec.version, _cache_view(params_json))
        cache_key = f"cache/{key[:2]}/{key}.json"
        t0 = time.perf_counter()
        if use_cache and spec.cacheable:
            hit = self.store.get_named(cache_key)
            if hit is not None:
                return ToolCallRecord(tool=name, version=spec.version, params=params_json,
                                      result=ToolResult.model_validate_json(hit), cache_hit=True)
        try:
            raw = await self.backend.submit(spec.resource, name, params_json, self._settings_json, job_id, instance_id)
        except Exception as e:  # recorded for the agent / checkpoint, then re-raised by callers that need it
            log.warning("tool %s failed: %s", name, e)
            return ToolCallRecord(tool=name, version=spec.version, params=params_json,
                                  error=f"{type(e).__name__}: {e}", seconds=time.perf_counter() - t0)
        result = ToolResult.model_validate(raw)
        if spec.cacheable:
            self.store.put_named(cache_key, json.dumps(raw).encode("utf-8"))
        return ToolCallRecord(tool=name, version=spec.version, params=params_json, result=result,
                              seconds=time.perf_counter() - t0)
