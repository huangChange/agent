"""Stage evaluators: compute the metrics a stage's gates are defined over.

They are run by the pipeline, never by the agent itself; the agent can only *request* an
evaluation. That separation keeps the actor from grading its own work.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.models import InstanceState, QAReport
from ..tools.base import ToolContext
from .gates import evaluate_gates
from .metrics import fidelity_metrics, scale_plausibility, topology_metrics

Evaluator = Callable[[ToolContext, InstanceState], dict[str, float]]


def _geometry(ctx: ToolContext, st: InstanceState) -> dict[str, float]:
    mesh = ctx.load_mesh(st.current["mesh"])
    g = ctx.load_gaussians(st.current["gaussians"])
    return {**topology_metrics(mesh), **fidelity_metrics(mesh, g)}


def _align(ctx: ToolContext, st: InstanceState) -> dict[str, float]:
    mesh = ctx.load_mesh(st.current["mesh"])
    size = mesh.bounds[1] - mesh.bounds[0]
    out = {"min_z": float(mesh.bounds[0][2]), "longest_dim_m": float(np.max(size))}
    out.update(scale_plausibility(st.instance.label, size))
    return out


def _texture(ctx: ToolContext, st: InstanceState) -> dict[str, float]:
    return {"generated_ratio": float(st.generated_ratio),
            "texture_present": float("texture" in st.current or st.facts.get("vertex_colors", False))}


def _physics(ctx: ToolContext, st: InstanceState) -> dict[str, float]:
    return {"mass_kg": float(st.facts.get("mass", 0.0)),
            "collision_parts": float(len([k for k in st.current if k.startswith("collision_")])),
            **st.facts.get("sim", {})}  # drop-test metrics, present when mujoco is installed


EVALUATORS: dict[str, Evaluator] = {"geometry": _geometry, "align": _align, "texture": _texture,
                                    "physics": _physics}


def evaluate_stage(ctx: ToolContext, stage: str, st: InstanceState) -> QAReport | None:
    fn = EVALUATORS.get(stage)
    if fn is None:
        return None
    metrics = fn(ctx, st)
    rules = ctx.settings.qa.get(stage, {})
    return evaluate_gates(stage, metrics, rules)
