"""Simulator acceptance test: does the asset behave in a physics engine?

Drops the collision geometry (with the computed mass and inertia) onto a ground plane in MuJoCo and
checks that it settles: no NaNs or explosions, comes to rest, stays upright, doesn't sink. This is
the check robotics customers care about most, and it catches problems no geometric metric does
(inverted normals in collision hulls, absurd inertia, a centre of mass outside the support).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from pydantic import BaseModel, Field

from ...core.errors import DependencyMissingError
from ...core.models import ArtifactRef
from ..base import ToolContext, ToolResult, tool


class DropParams(BaseModel):
    collision: list[ArtifactRef] = Field(min_length=1)
    mass: float = Field(gt=0)
    center_of_mass: tuple[float, float, float]
    inertia: tuple[float, float, float, float, float, float]  # ixx ixy ixz iyy iyz izz
    friction: float = 0.6
    drop_height: float = Field(0.005, description="gap between the lowest point and the ground at start")
    seconds: float = 3.0


@tool("sim_drop_test", stages=("physics",), version="1.0.0")
def sim_drop_test(ctx: ToolContext, p: DropParams) -> ToolResult:
    """Drop the asset onto a ground plane in MuJoCo and report whether it settles: sim_settled (1/0),
    sim_tilt_deg (rotation at rest), sim_sink_m (penetration below the ground), sim_max_speed."""
    try:
        import mujoco
    except ImportError as e:
        raise DependencyMissingError("mujoco", "sim") from e
    parts = [ctx.load_mesh(r) for r in p.collision]
    lowest = min(float(m.vertices[:, 2].min()) for m in parts)
    root = ET.Element("mujoco", model="drop_test")
    ET.SubElement(root, "option", timestep="0.002")
    assets = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="plane", size="5 5 0.1", friction=f"{p.friction} 0.005 0.0001")
    body = ET.SubElement(world, "body", name="asset", pos=f"0 0 {p.drop_height - lowest:.6f}")
    ET.SubElement(body, "freejoint")
    ixx, ixy, ixz, iyy, iyz, izz = p.inertia
    ET.SubElement(body, "inertial", pos=" ".join(f"{v:.6g}" for v in p.center_of_mass), mass=f"{p.mass:.6g}",
                  fullinertia=" ".join(f"{v:.6g}" for v in (ixx, iyy, izz, ixy, ixz, iyz)))
    for i, m in enumerate(parts):
        f = ctx.scratch(f"col_{i}.obj")
        f.write_text(m.export(file_type="obj"), encoding="utf-8")
        ET.SubElement(assets, "mesh", name=f"c{i}", file=str(f.resolve()))
        ET.SubElement(body, "geom", type="mesh", mesh=f"c{i}", friction=f"{p.friction} 0.005 0.0001")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    q0 = data.qpos[3:7].copy()
    max_speed = 0.0
    for _ in range(int(p.seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        max_speed = max(max_speed, float(np.linalg.norm(data.qvel[:3])))
        if not np.isfinite(data.qpos).all():
            return ToolResult(metrics={"sim_settled": 0.0, "sim_max_speed": float("inf")},
                              warnings=["simulation diverged (NaN)"])
    q = data.qpos[3:7]
    tilt = float(np.degrees(2 * np.arccos(np.clip(abs(float(np.dot(q, q0))), -1.0, 1.0))))
    # lowest collision vertex in the final pose
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, q)
    R, t = rot.reshape(3, 3), data.qpos[:3]
    z_min = min(float(((R @ m.vertices.T).T + t)[:, 2].min()) for m in parts)
    speed = float(np.linalg.norm(data.qvel))
    settled = speed < 0.02 and max_speed < 5.0
    return ToolResult(metrics={"sim_settled": float(settled), "sim_tilt_deg": tilt, "sim_sink_m": max(0.0, -z_min),
                               "sim_final_speed": speed, "sim_max_speed": max_speed})
