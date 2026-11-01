"""Long-lived services a pipeline run needs, bundled so stages stay plain functions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..agent.critic import Critic
from ..agent.loop import RepairAgent
from ..core.config import Settings
from ..core.errors import ToolError
from ..runtime.runner import ToolRunner
from ..storage.artifacts import ArtifactStore
from ..tools.base import ToolContext, ToolResult
from ..tools.registry import get_tool


@dataclass
class Services:
    settings: Settings
    store: ArtifactStore
    runner: ToolRunner
    agent: RepairAgent | None = None
    critic: Critic | None = None
    gpu_available: bool = False

    async def call(self, tool: str, job_id: str, instance_id: str, **params: Any) -> ToolResult:
        """Deterministic stage call: a tool error is a stage failure (raised), not something to inspect."""
        rec = await self.runner.run(tool, params, job_id=job_id, instance_id=instance_id)
        if rec.error:
            raise ToolError(f"{tool}: {rec.error}")
        return rec.result

    def local_ctx(self, job_id: str, instance_id: str) -> ToolContext:
        """In-process context for evaluators (they only read artifacts and compute metrics)."""
        workdir = Path(self.settings.runtime.workdir) / job_id / instance_id / f"eval-{uuid.uuid4().hex[:8]}"
        return ToolContext(store=self.store, settings=self.settings, workdir=workdir, job_id=job_id,
                           instance_id=instance_id)

    def tool_versions(self) -> dict[str, str]:
        from ..tools.registry import load_builtin_tools

        return {name: spec.version for name, spec in sorted(load_builtin_tools().items())}

    @staticmethod
    def has_tool(name: str) -> bool:
        try:
            get_tool(name)
            return True
        except KeyError:
            return False
