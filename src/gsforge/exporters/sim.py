"""Simulator descriptions (URDF for Isaac / PyBullet / ROS tooling, MJCF for MuJoCo) from AssetModel.

Mesh paths are relative to the description file. Rigid assets become one link (URDF) or one free
body (MJCF); articulated assets become a link tree with revolute / prismatic joints.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..core.models import AssetModel, JointType


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def write_urdf(asset: AssetModel, visual_files: dict[str, str], collision_files: dict[str, list[str]],
               path: str | Path) -> Path:
    robot = ET.Element("robot", name=asset.name)
    for link in asset.links:
        el = ET.SubElement(robot, "link", name=link.name)
        inertial = ET.SubElement(el, "inertial")
        ET.SubElement(inertial, "origin", xyz=_fmt(link.center_of_mass), rpy="0 0 0")
        ET.SubElement(inertial, "mass", value=f"{link.mass:.6g}")
        ixx, ixy, ixz, iyy, iyz, izz = link.inertia
        ET.SubElement(inertial, "inertia", ixx=f"{ixx:.6g}", ixy=f"{ixy:.6g}", ixz=f"{ixz:.6g}",
                      iyy=f"{iyy:.6g}", iyz=f"{iyz:.6g}", izz=f"{izz:.6g}")
        vis = ET.SubElement(el, "visual")
        ET.SubElement(ET.SubElement(vis, "geometry"), "mesh", filename=visual_files[link.name])
        for cf in collision_files.get(link.name, []):
            col = ET.SubElement(el, "collision")
            ET.SubElement(ET.SubElement(col, "geometry"), "mesh", filename=cf)
    for j in asset.joints:
        el = ET.SubElement(robot, "joint", name=j.name, type=j.type.value)
        ET.SubElement(el, "parent", link=j.parent)
        ET.SubElement(el, "child", link=j.child)
        ET.SubElement(el, "origin", xyz=_fmt(j.origin_xyz), rpy=_fmt(j.origin_rpy))
        if j.type != JointType.FIXED:
            ET.SubElement(el, "axis", xyz=_fmt(j.axis))
        if j.type in (JointType.REVOLUTE, JointType.PRISMATIC):
            ET.SubElement(el, "limit", lower=f"{j.lower or 0.0:.6g}", upper=f"{j.upper or 0.0:.6g}",
                          effort=f"{j.effort:.6g}", velocity=f"{j.velocity:.6g}")
    ET.indent(robot)
    path = Path(path)
    path.write_text(ET.tostring(robot, encoding="unicode", xml_declaration=True), encoding="utf-8")
    return path


def write_mjcf(asset: AssetModel, visual_files: dict[str, str], collision_files: dict[str, list[str]],
               path: str | Path, texture_files: dict[str, str] | None = None) -> Path:
    root = ET.Element("mujoco", model=asset.name)
    ET.SubElement(root, "compiler", angle="radian", meshdir=".", texturedir=".")
    assets = ET.SubElement(root, "asset")
    for link in asset.links:
        ET.SubElement(assets, "mesh", name=f"{link.name}_visual", file=visual_files[link.name])
        for i, cf in enumerate(collision_files.get(link.name, [])):
            ET.SubElement(assets, "mesh", name=f"{link.name}_col{i}", file=cf)
        if texture_files and link.name in texture_files:
            ET.SubElement(assets, "texture", name=f"{link.name}_tex", type="2d", file=texture_files[link.name])
            ET.SubElement(assets, "material", name=f"{link.name}_mat", texture=f"{link.name}_tex")

    children: dict[str, list] = {}
    for j in asset.joints:
        children.setdefault(j.parent, []).append(j)
    child_names = {j.child for j in asset.joints}
    links = {link.name: link for link in asset.links}
    roots = [name for name in links if name not in child_names]

    def add_body(parent_el, link_name: str, joint=None):
        link = links[link_name]
        attrs = {"name": link_name}
        if joint is not None:
            attrs["pos"] = _fmt(joint.origin_xyz)
        body = ET.SubElement(parent_el, "body", **attrs)
        if joint is None:
            ET.SubElement(body, "freejoint", name=f"{link_name}_free")
        elif joint.type != JointType.FIXED:
            jtype = "slide" if joint.type == JointType.PRISMATIC else "hinge"
            jattrs = {"name": joint.name, "type": jtype, "axis": _fmt(joint.axis)}
            if joint.lower is not None and joint.upper is not None:
                jattrs.update(range=f"{joint.lower:.6g} {joint.upper:.6g}", limited="true")
            ET.SubElement(body, "joint", **jattrs)
        ixx, ixy, ixz, iyy, iyz, izz = link.inertia
        ET.SubElement(body, "inertial", pos=_fmt(link.center_of_mass), mass=f"{link.mass:.6g}",
                      fullinertia=_fmt((ixx, iyy, izz, ixy, ixz, iyz)))  # MuJoCo order
        vis = {"type": "mesh", "mesh": f"{link_name}_visual", "contype": "0", "conaffinity": "0", "group": "2"}
        if texture_files and link_name in texture_files:
            vis["material"] = f"{link_name}_mat"
        ET.SubElement(body, "geom", **vis)
        for i, _ in enumerate(collision_files.get(link_name, [])):
            ET.SubElement(body, "geom", type="mesh", mesh=f"{link_name}_col{i}", group="3",
                          friction=f"{link.friction:.6g} 0.005 0.0001")
        for j in children.get(link_name, []):
            add_body(body, j.child, j)

    world = ET.SubElement(root, "worldbody")
    for r in roots:
        add_body(world, r)
    ET.indent(root)
    path = Path(path)
    path.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return path
