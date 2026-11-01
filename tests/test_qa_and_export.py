import xml.etree.ElementTree as ET

import trimesh

from gsforge.core.models import AssetModel, Grade, Joint, JointType, Link
from gsforge.exporters.files import export_glb, export_ply
from gsforge.exporters.sim import write_mjcf, write_urdf
from gsforge.qa.gates import evaluate_gates, grade

RULES = {"recall": {"threshold": 0.9, "comparator": ">="},
         "components": {"threshold": 1, "comparator": "<=", "severity": "soft"},
         "scale_plausibility": {"threshold": 0.5, "comparator": ">=", "severity": "soft", "required": False}}


def test_gates_hard_soft_and_optional():
    ok = evaluate_gates("geometry", {"recall": 0.95, "components": 1}, RULES)
    assert ok.passed and not ok.soft_failures and ok.score == 1.0
    soft = evaluate_gates("geometry", {"recall": 0.95, "components": 3}, RULES)
    assert soft.passed and [c.name for c in soft.soft_failures] == ["components"]
    hard = evaluate_gates("geometry", {"components": 1}, RULES)  # required metric missing -> hard failure
    assert not hard.passed and hard.hard_failures[0].detail == "metric was not computed"
    assert grade([ok], generated_ratio=0.05) == Grade.A
    assert grade([soft], generated_ratio=0.05) == Grade.B
    assert grade([ok], generated_ratio=0.4) == Grade.B
    assert grade([hard], generated_ratio=0.0) == Grade.REJECT


def test_glb_is_y_up_and_ply_has_colors(tmp_path):
    box = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    box.visual.vertex_colors = [200, 10, 10, 255]
    export_glb({"obj": box}, tmp_path / "a.glb")
    loaded = trimesh.load(tmp_path / "a.glb", force="mesh")
    ext = loaded.bounds[1] - loaded.bounds[0]
    assert abs(ext[1] - 0.3) < 1e-6  # canonical Z (0.3) became glTF Y
    export_ply(box, tmp_path / "a.ply")
    ply = trimesh.load(tmp_path / "a.ply", force="mesh")
    assert ply.visual.vertex_colors[0][0] == 200


def test_urdf_and_mjcf_articulated(tmp_path, store):
    ref = store.put_bytes(b"x", kind="mesh", suffix=".ply")
    asset = AssetModel(name="cabinet", links=[Link(name="body", visual_mesh=ref, mass=20.0),
                                              Link(name="drawer", visual_mesh=ref, mass=2.0)],
                       joints=[Joint(name="slide", type=JointType.PRISMATIC, parent="body", child="drawer",
                                     origin_xyz=(0.3, 0, 0.5), axis=(1, 0, 0), lower=0.0, upper=0.35)])
    vis = {"body": "body.obj", "drawer": "drawer.obj"}
    col = {"body": ["body_c0.obj"], "drawer": ["drawer_c0.obj"]}
    urdf = ET.parse(write_urdf(asset, vis, col, tmp_path / "a.urdf")).getroot()
    assert urdf.find("joint").get("type") == "prismatic"
    assert urdf.find("joint/limit").get("upper") == "0.35"
    mj = ET.parse(write_mjcf(asset, vis, col, tmp_path / "a.xml")).getroot()
    body = mj.find("worldbody/body")
    assert body.find("freejoint") is not None
    drawer = body.find("body")
    assert drawer.get("name") == "drawer" and drawer.find("joint").get("type") == "slide"
