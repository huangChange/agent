"""VLM critic: an independent visual review of an asset against its capture.

Runs on a separate request with its own prompt and never sees the repair agent's reasoning, so it
is not anchored on the actor's conclusions. Its output feeds soft QA checks and the label /
material facts used by the material and physics stages.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from pydantic import BaseModel

from ..core.models import ArtifactRef
from ..storage.artifacts import ArtifactStore
from .llm import LLM
from .roles import critic_prompt

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "description": "object category, lowercase, e.g. 'tv remote'"},
        "material_class": {"type": "string",
                           "enum": ["wood", "plastic", "metal", "fabric", "glass", "ceramic", "rubber", "paper",
                                    "stone", "leather", "mixed"]},
        "faithfulness": {"type": "number", "description": "0-1, geometry and appearance vs reference"},
        "completeness": {"type": "number", "description": "0-1, 1 = no visibly missing or half-finished areas"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["missing_part", "extra_part", "hole", "proportion",
                                                        "texture_smear", "illegible_text", "baked_lighting",
                                                        "color_shift", "physics_risk", "other"]},
                    "severity": {"type": "string", "enum": ["minor", "major", "blocking"]},
                    "where": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["kind", "severity", "where", "description"],
                "additionalProperties": False,
            },
        },
        "acceptable": {"type": "boolean"},
    },
    "required": ["category", "material_class", "faithfulness", "completeness", "issues", "acceptable"],
    "additionalProperties": False,
}


class Verdict(BaseModel):
    category: str
    material_class: str
    faithfulness: float
    completeness: float
    issues: list[dict[str, str]]
    acceptable: bool

    def metrics(self) -> dict[str, float]:
        blocking = sum(1 for i in self.issues if i.get("severity") == "blocking")
        return {"critic_faithfulness": self.faithfulness, "critic_completeness": self.completeness,
                "critic_blocking_issues": float(blocking), "critic_acceptable": float(self.acceptable)}


class Critic:
    def __init__(self, llm: LLM, store: ArtifactStore):
        self.llm = llm
        self.store = store

    def _image(self, ref: ArtifactRef) -> dict[str, Any]:
        data = base64.standard_b64encode(self.store.local_path(ref).read_bytes()).decode("ascii")
        return {"type": "image", "source": {"type": "base64", "media_type": ref.media_type, "data": data}}

    async def review(self, mesh_views: ArtifactRef, reference_views: ArtifactRef,
                     photos: list[ArtifactRef] | None = None, label_hint: str | None = None) -> tuple[Verdict, Any]:
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Delivered mesh, rendered from several cameras:"}, self._image(mesh_views),
            {"type": "text", "text": "Reference: the captured gaussians from the same cameras:"},
            self._image(reference_views),
        ]
        for p in (photos or [])[:3]:
            content += [{"type": "text", "text": "Original capture photo:"}, self._image(p)]
        if label_hint:
            content.append({"type": "text", "text": f"Segmentation label hint: {label_hint}"})
        text, usage = await self.llm.structured(system=critic_prompt(), content=content, schema=VERDICT_SCHEMA)
        return Verdict.model_validate(json.loads(text)), usage
