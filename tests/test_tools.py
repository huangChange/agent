import numpy as np
import trimesh

from gsforge.core.models import ArtifactKind
from gsforge.io.cameras import orbit_cameras
from gsforge.io.gaussians import gaussians_from_points
from gsforge.qa.metrics import fidelity_metrics, topology_metrics
from gsforge.tools.geometry.extract import tsdf_fuse
from gsforge.tools.registry import get_tool


def call(ctx, name, **params):
    spec = get_tool(name)
    return spec.fn(ctx, spec.params_model.model_validate(params))


def test_split_table_remote_mug(ctx, table_scene):
    seg = ctx.store.put_file(table_scene.segment_gaussians, ArtifactKind.GAUSSIANS)
    r = call(ctx, "split_instances", gaussians=seg, up=(0, 0, 1))
    roles = sorted(i["role"] for i in r.facts["instances"])
    assert roles == ["support", "supported", "supported"]
    assert abs(r.facts["support_plane_height"] - 0.74) < 0.02
    sizes = {i["role"]: i for i in r.facts["instances"]}
    support = sizes["support"]
    assert support["bbox_max"][0] - support["bbox_min"][0] > 1.1  # the table, not a small object

    sup_ref = next(r.outputs[i["key"]] for i in r.facts["instances"] if i["role"] == "support")
    patch = call(ctx, "patch_support_surface", gaussians=sup_ref, plane_height=r.facts["support_plane_height"],
                 spacing=r.facts["link_distance"] / 3)
    assert patch.metrics["generated_count"] > 0  # holes under remote and mug were filled
    assert patch.metrics["generated_ratio"] < 0.1


def test_estimate_up_is_level_and_points_away_from_floor(ctx, table_scene):
    # regression: a scene-relative RANSAC band let a plane wedged inside the table top win (2.6 deg tilt),
    # and "most content on one side" then flipped up to -Z
    scene = ctx.store.put_file(table_scene.scene_gaussians, ArtifactKind.GAUSSIANS)
    up = np.asarray(call(ctx, "estimate_up", scene_gaussians=scene).facts["up"])
    assert up @ np.array([0, 0, 1.0]) > np.cos(np.radians(0.5))


def test_open_bottom_object_becomes_closed_solid(ctx):
    # a mug standing on a table: walls and top observed, bottom never
    mug = trimesh.creation.cylinder(radius=0.042, height=0.1, sections=64)
    mug.apply_translation((0, 0, 0.05))
    pts, fid = trimesh.sample.sample_surface(mug, 18000, seed=0)
    keep = mug.face_normals[fid][:, 2] > -0.9
    g = gaussians_from_points(pts[keep], np.full((keep.sum(), 3), 0.8), mug.face_normals[fid][keep], scale=0.001)
    closed = call(ctx, "close_contact_base", gaussians=ctx.put_gaussians(g), base_height=0.0)
    assert 0.08 < closed.metrics["generated_ratio"] < 0.25  # ~ base area / total area, reported honestly
    mesh = ctx.load_mesh(call(ctx, "extract_density_mc", gaussians=closed.outputs["gaussians"]).outputs["mesh"])
    topo = topology_metrics(mesh)
    assert topo["watertight"] == 1.0 and topo["components"] == 1.0
    assert abs(mesh.volume - mug.volume) / mug.volume < 0.05


def test_density_mc_sphere_is_watertight_and_faithful(ctx):
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.1)
    pts, fid = trimesh.sample.sample_surface(sphere, 20000, seed=0)
    g = gaussians_from_points(pts, np.full((len(pts), 3), 0.6), sphere.face_normals[fid], scale=0.002)
    ref = ctx.put_gaussians(g)
    mesh_ref = call(ctx, "extract_density_mc", gaussians=ref, resolution=96).outputs["mesh"]
    mesh = ctx.load_mesh(mesh_ref)
    topo = topology_metrics(mesh)
    assert topo["watertight"] == 1.0 and topo["components"] == 1.0
    fid_m = fidelity_metrics(mesh, g)
    assert fid_m["fscore"] > 0.9, fid_m
    assert abs(mesh.volume - sphere.volume) / sphere.volume < 0.15


def test_tsdf_fuse_recovers_sphere():
    r, c = 0.3, np.zeros(3)
    cams = orbit_cameras(c, 1.5, 12, elevation_deg=20, size=96) + orbit_cameras(c, 1.5, 6, elevation_deg=-50, size=96)
    depths = []
    for cam in cams:  # analytic ray/sphere depth
        ys, xs = np.mgrid[0:cam.height, 0:cam.width] + 0.5
        d_cam = np.stack([(xs - cam.K[0, 2]) / cam.K[0, 0], (ys - cam.K[1, 2]) / cam.K[1, 1], np.ones_like(xs)], -1)
        d_world = d_cam @ cam.c2w[:3, :3].T
        dn = d_world / np.linalg.norm(d_world, axis=-1, keepdims=True)
        o = cam.center - c
        b = dn @ o
        disc = b ** 2 - (o @ o - r ** 2)
        t = -b - np.sqrt(np.clip(disc, 0, None))
        z = t * (dn @ cam.c2w[:3, 2])  # ray length -> z-depth
        depths.append(np.where(disc > 0, z, 0.0))
    mesh = tsdf_fuse(depths, cams, c - r, c + r, resolution=48)
    radii = np.linalg.norm(mesh.vertices - c, axis=1)
    assert abs(np.median(radii) - r) < 0.02


def test_repair_closes_open_crop(ctx):
    box = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    keep = np.flatnonzero(box.face_normals[:, 2] > -0.5)  # remove the bottom: an open contact region
    open_box = box.submesh([keep], append=True)
    ref = ctx.put_mesh(open_box)
    r = call(ctx, "repair_mesh", mesh=ref)
    assert r.metrics["before_watertight"] == 0.0
    assert r.metrics["after_watertight"] == 1.0


def test_canonicalize_and_collision(ctx):
    box = trimesh.creation.box(extents=(0.4, 0.2, 0.1))
    box.apply_translation((5, 5, 5))
    pts, fid = trimesh.sample.sample_surface(box, 2000, seed=0)
    g = ctx.put_gaussians(gaussians_from_points(pts, np.full((len(pts), 3), 0.5), box.face_normals[fid]))
    r = call(ctx, "canonicalize_pose", mesh=ctx.put_mesh(box), gaussians=g, up=(0, 0, 1))
    mesh = ctx.load_mesh(r.outputs["mesh"])
    assert abs(mesh.bounds[0][2]) < 1e-9 and np.allclose(mesh.bounds.mean(axis=0)[:2], 0, atol=1e-9)
    c = call(ctx, "build_collision", mesh=r.outputs["mesh"], density=1000.0)
    assert abs(c.facts["mass"] - 0.4 * 0.2 * 0.1 * 1000) < 1e-4  # PLY stores float32 vertices


def test_vertex_colors_and_bake(ctx):
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    pts, fid = trimesh.sample.sample_surface(sphere, 8000, seed=0)
    pts = pts[pts[:, 2] > -0.05]  # bottom cap unobserved
    colors = np.where(pts[:, :1] > 0, [[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]])
    g = ctx.put_gaussians(gaussians_from_points(pts, colors))
    r = call(ctx, "transfer_vertex_colors", mesh=ctx.put_mesh(sphere), gaussians=g)
    assert 0.02 < r.metrics["generated_ratio"] < 0.3  # the unobserved cap is reported, not hidden
    mesh = ctx.load_mesh(r.outputs["mesh"])
    vc = np.asarray(mesh.visual.vertex_colors)
    right = mesh.vertices[:, 0] > 0.03
    assert vc[right, 0].mean() > 200 and vc[right, 2].mean() < 60
    b = call(ctx, "bake_vertex_colors", mesh=r.outputs["mesh"], texture_size=256)
    assert b.metrics["uv_coverage"] > 0.3
    textured = ctx.load_mesh(b.outputs["mesh"])
    assert textured.visual.kind == "texture"
