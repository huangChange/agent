"""Material assignment: PBR factors + physical properties from a material class.

The class comes from the label prior table or from the VLM material classifier (agent role).
Values live in configs/priors/materials.yaml so they can be tuned without code changes.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ...core.errors import ToolError
from ...qa.metrics import load_priors
from ..base import ToolContext, ToolResult, tool


class MaterialParams(BaseModel):
    material_class: str | None = Field(None, description="e.g. wood, plastic, metal, fabric, glass, ceramic")
    label: str | None = Field(None, description="object category; used when material_class is unknown")


@tool("assign_material", stages=("material",), version="1.0.0", cacheable=False)
def assign_material(ctx: ToolContext, p: MaterialParams) -> ToolResult:
    """Look up PBR factors (roughness, metallic) and physics (density kg/m^3, friction) for a
    material class, falling back to the category's default material, then to 'plastic'."""
    mats = load_priors("materials")
    if not mats:
        raise ToolError("configs/priors/materials.yaml is missing")
    cls = p.material_class
    if not cls and p.label:
        cls = load_priors("sizes").get(p.label.lower(), {}).get("material")
    cls = (cls or "plastic").lower()
    if cls not in mats:
        return ToolResult(facts={"material_class": "plastic", **mats["plastic"]},
                          warnings=[f"unknown material '{cls}', used plastic"])
    return ToolResult(facts={"material_class": cls, **mats[cls]})
