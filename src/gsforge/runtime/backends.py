"""Execution backends: where a tool call physically runs.

LocalBackend  one machine: process pool for CPU tools, one single-slot process pool per GPU
              (pins CUDA_VISIBLE_DEVICES), threads for IO-bound tools. `inline=True` runs everything
              in threads of the current process (tests, debugging).
RayBackend    cluster: each call becomes a Ray task with num_cpus / num_gpus resources, so the Ray
              scheduler packs CPU and GPU work across nodes and autoscaling can follow queue depth.

Both execute the same picklable entry point, `execute_tool`, which rebuilds its context from
serialized settings: no live objects cross the process boundary.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..core.config import Settings
from ..core.errors import DependencyMissingError


def execute_tool(tool_name: str, params_json: dict[str, Any], settings_json: dict[str, Any],
                 job_id: str, instance_id: str) -> dict[str, Any]:
    """Worker-side entry point. Must stay importable and free of closures (pickled by name)."""
    from ..storage.artifacts import open_store
    from ..tools.base import ToolContext
    from ..tools.registry import get_tool

    settings = Settings.model_validate(settings_json)
    spec = get_tool(tool_name)
    params = spec.params_model.model_validate(params_json)
    workdir = Path(settings.runtime.workdir) / job_id / instance_id / f"{tool_name}-{uuid.uuid4().hex[:8]}"
    ctx = ToolContext(store=open_store(settings.storage), settings=settings, workdir=workdir,
                      job_id=job_id, instance_id=instance_id)
    try:
        return spec.fn(ctx, params).model_dump(mode="json")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _pin_gpu(device: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)


class Backend(ABC):
    @abstractmethod
    async def submit(self, resource: str, *args: Any) -> dict[str, Any]: ...

    def shutdown(self) -> None:  # noqa: B027 - optional hook
        pass


class LocalBackend(Backend):
    def __init__(self, settings: Settings, inline: bool = False):
        self.inline = inline
        self.io = ThreadPoolExecutor(max_workers=16, thread_name_prefix="gsforge-io")
        if inline:
            self.cpu: Executor = ThreadPoolExecutor(max_workers=settings.runtime.cpu_workers)
            self.gpus: list[Executor] = []
        else:
            self.cpu = ProcessPoolExecutor(max_workers=settings.runtime.cpu_workers)
            self.gpus = [ProcessPoolExecutor(max_workers=1, initializer=_pin_gpu, initargs=(d,))
                         for d in settings.runtime.gpu_devices]
        self._gpu_rr = itertools.cycle(range(len(self.gpus))) if self.gpus else None

    def _executor(self, resource: str) -> Executor:
        if resource == "gpu":
            if self.inline:
                return self.cpu
            if not self.gpus:
                raise DependencyMissingError("GPU worker (runtime.gpu_devices is empty)", "gpu")
            return self.gpus[next(self._gpu_rr)]
        return self.io if resource == "io" else self.cpu

    async def submit(self, resource: str, *args: Any) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor(resource), execute_tool, *args)

    def shutdown(self) -> None:
        for ex in [self.io, self.cpu, *self.gpus]:
            ex.shutdown(wait=False, cancel_futures=True)


class RayBackend(Backend):  # pragma: no cover - requires a Ray cluster
    def __init__(self, settings: Settings, address: str | None = "auto"):
        try:
            import ray
        except ImportError as e:
            raise DependencyMissingError("ray", "cloud") from e
        if not ray.is_initialized():
            ray.init(address=address)
        self._ray = ray
        self._cpu = ray.remote(num_cpus=1)(execute_tool)
        self._gpu = ray.remote(num_cpus=1, num_gpus=1)(execute_tool)

    async def submit(self, resource: str, *args: Any) -> dict[str, Any]:
        task = self._gpu if resource == "gpu" else self._cpu
        return await task.remote(*args)


def make_backend(settings: Settings, inline: bool = False) -> Backend:
    if settings.runtime.backend == "ray":
        return RayBackend(settings)
    return LocalBackend(settings, inline=inline)
