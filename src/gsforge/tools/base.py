"""Tool contract.

A tool is a pure function `(ctx, params) -> ToolResult`:
  * params is a pydantic model -> validated input and an auto-generated JSON schema for the agent,
  * inputs/outputs are ArtifactRefs -> no large arrays cross process boundaries,
  * a tool never mutates its inputs -> any call can be cached, retried or run on another worker,
  * `version` is bumped on behavior changes -> invalidates the step cache and is recorded in lineage.

The same registry serves the deterministic pipeline and the repair agent; that is how ad-hoc agent
code gets "promoted": once reviewed, it is rewritten as a tool here and becomes available to both.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, Field

from ..core.config import Settings
from ..core.models import ArtifactKind, ArtifactRef
from ..io.gaussians import Gaussians, read_gaussians, write_gaussians
from ..io.mesh_io import load_mesh, save_mesh
from ..storage.artifacts import ArtifactStore

Resource = Literal["cpu", "gpu", "io"]
ARTIFACT_ID_RE = re.compile(r"^art_[0-9a-f]{12}$")


class ToolResult(BaseModel):
    outputs: dict[str, ArtifactRef] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    facts: dict[str, Any] = Field(default_factory=dict)  # small JSON values (transforms, labels, ...)
    previews: list[ArtifactRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass
class ToolContext:
    store: ArtifactStore
    settings: Settings
    workdir: Path
    job_id: str = "adhoc"
    instance_id: str = "adhoc"
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("gsforge.tool"))

    def scratch(self, name: str) -> Path:
        self.workdir.mkdir(parents=True, exist_ok=True)
        return self.workdir / name

    # ---- typed load/store helpers so tools stay short
    def load_gaussians(self, ref: ArtifactRef) -> Gaussians:
        return read_gaussians(self.store.local_path(ref))

    def put_gaussians(self, g: Gaussians, name: str = "gaussians.ply", **meta: Any) -> ArtifactRef:
        p = self.scratch(name)
        write_gaussians(g, p)
        return self.store.put_file(p, ArtifactKind.GAUSSIANS, {"count": len(g), **meta})

    def load_mesh(self, ref: ArtifactRef) -> trimesh.Trimesh:
        return load_mesh(self.store.local_path(ref))

    def put_mesh(self, mesh: trimesh.Trimesh, name: str = "mesh.ply", **meta: Any) -> ArtifactRef:
        p = save_mesh(mesh, self.scratch(name))
        return self.store.put_file(p, ArtifactKind.MESH,
                                   {"vertices": len(mesh.vertices), "faces": len(mesh.faces), **meta})

    def put_image(self, rgb: np.ndarray, name: str, **meta: Any) -> ArtifactRef:
        from PIL import Image

        p = self.scratch(name)
        Image.fromarray(rgb).save(p)
        return self.store.put_file(p, ArtifactKind.IMAGE, meta)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    version: str
    params_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], ToolResult]
    stages: tuple[str, ...]
    resource: Resource = "cpu"
    agent_visible: bool = True
    cacheable: bool = True

    def agent_schema(self) -> dict[str, Any]:
        """Claude tool definition. ArtifactRef fields are exposed as artifact id strings."""
        schema = _artifact_refs_to_ids(self.params_model.model_json_schema())
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, *, stages: tuple[str, ...], version: str, resource: Resource = "cpu",
         agent_visible: bool = True, cacheable: bool = True):
    """Register `fn(ctx, params: SomeParams) -> ToolResult`. The docstring becomes the agent description."""

    def deco(fn: Callable[[ToolContext, Any], ToolResult]):
        params_model = fn.__annotations__.get("p")
        if isinstance(params_model, str):  # from __future__ annotations
            params_model = fn.__globals__[params_model]
        if not (isinstance(params_model, type) and issubclass(params_model, BaseModel)):
            raise TypeError(f"tool {name}: second argument must be annotated `p: <pydantic model>`")
        if name in REGISTRY:
            raise ValueError(f"duplicate tool name {name}")
        REGISTRY[name] = ToolSpec(name=name, description=inspect.cleandoc(fn.__doc__ or ""), version=version,
                                  params_model=params_model, fn=fn, stages=stages, resource=resource,
                                  agent_visible=agent_visible, cacheable=cacheable)
        return fn

    return deco


def _artifact_refs_to_ids(schema: dict[str, Any]) -> dict[str, Any]:
    defs = schema.pop("$defs", {})
    defs.pop("ArtifactRef", None)
    defs.pop("ArtifactKind", None)
    id_schema = {"type": "string", "description": "artifact id, e.g. art_0123456789ab"}

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("$ref") == "#/$defs/ArtifactRef":
                return dict(id_schema)
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    out = walk(schema)
    if defs:
        out["$defs"] = walk(defs)
    return out


def resolve_artifact_ids(data: Any, lookup: Callable[[str], ArtifactRef]) -> Any:
    """Replace artifact id strings coming from the agent with full ArtifactRef dicts."""
    if isinstance(data, str) and ARTIFACT_ID_RE.match(data):
        return lookup(data).model_dump(mode="json")
    if isinstance(data, dict):
        return {k: resolve_artifact_ids(v, lookup) for k, v in data.items()}
    if isinstance(data, list):
        return [resolve_artifact_ids(v, lookup) for v in data]
    return data
