"""Tool discovery. Importing a tool module registers its tools in REGISTRY."""

from __future__ import annotations

import importlib

from .base import REGISTRY, ToolSpec

BUILTIN_MODULES = (
    "gsforge.tools.segmentation.cleanup",
    "gsforge.tools.segmentation.instance_split",
    "gsforge.tools.geometry.extract",
    "gsforge.tools.geometry.repair",
    "gsforge.tools.geometry.align",
    "gsforge.tools.texture.color",
    "gsforge.tools.texture.bake",
    "gsforge.tools.material.pbr",
    "gsforge.tools.physics.collision",
    "gsforge.tools.physics.simcheck",
    "gsforge.tools.inspect",
)

_loaded = False


def load_builtin_tools() -> dict[str, ToolSpec]:
    global _loaded
    if not _loaded:
        for mod in BUILTIN_MODULES:
            importlib.import_module(mod)
        _loaded = True
    return REGISTRY


def get_tool(name: str) -> ToolSpec:
    load_builtin_tools()
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown tool '{name}'. known: {sorted(REGISTRY)}") from None


def tools_for_stage(stage: str, agent_only: bool = True) -> list[ToolSpec]:
    load_builtin_tools()
    specs = [t for t in REGISTRY.values() if (stage in t.stages or "*" in t.stages)
             and (t.agent_visible or not agent_only)]
    return sorted(specs, key=lambda t: t.name)  # deterministic order keeps the prompt cache warm
