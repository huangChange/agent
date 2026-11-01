"""Wiring: build all services for a worker process from Settings."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from ..agent.critic import Critic
from ..agent.llm import LLM, ClaudeLLM
from ..agent.loop import RepairAgent
from ..agent.sandbox import Sandbox
from ..agent.skills import HarvestLibrary, load_skills
from ..core.config import Settings
from ..runtime.backends import make_backend
from ..runtime.runner import ToolRunner
from ..storage.artifacts import open_store
from ..tools.registry import load_builtin_tools
from .services import Services


def _gpu_available(settings: Settings) -> bool:
    if not settings.runtime.gpu_devices or importlib.util.find_spec("torch") is None:
        return False
    import torch

    return bool(torch.cuda.is_available())


def build_services(settings: Settings, *, inline: bool = False, llm: LLM | None = None,
                   use_llm: bool = True) -> Services:
    load_builtin_tools()
    store = open_store(settings.storage)
    runner = ToolRunner(store, settings, make_backend(settings, inline=inline))
    svc = Services(settings=settings, store=store, runner=runner, gpu_available=_gpu_available(settings))
    if use_llm:
        llm = llm or ClaudeLLM(settings.llm, concurrency=settings.runtime.llm_concurrency)
        sandbox = Sandbox(settings.agent.sandbox, store, Path(settings.runtime.workdir) / "sandbox")
        svc.agent = RepairAgent(llm=llm, runner=runner, store=store, settings=settings, sandbox=sandbox,
                                harvest=HarvestLibrary(store), skills=load_skills(settings.agent.skill_dirs))
        svc.critic = Critic(llm, store)
    return svc
