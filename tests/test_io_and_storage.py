import numpy as np

from gsforge.core.models import ArtifactKind
from gsforge.io.gaussians import gaussians_from_points, read_gaussians, write_gaussians
from gsforge.tools.geometry.align import rotation_between


def test_gaussian_ply_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(500, 3))
    nrm = rng.normal(size=(500, 3))
    g = gaussians_from_points(pts, rng.uniform(size=(500, 3)), nrm)
    write_gaussians(g, tmp_path / "g.ply")
    h = read_gaussians(tmp_path / "g.ply")
    assert len(h) == 500
    np.testing.assert_allclose(h.means, g.means, atol=1e-6)
    np.testing.assert_allclose(h.rgb, g.rgb, atol=1e-5)
    # flattened gaussians: the smallest axis is the surface normal
    n = nrm / np.linalg.norm(nrm, axis=1, keepdims=True)
    assert np.abs(np.sum(h.normals() * n, axis=1)).min() > 0.99


def test_transform_moves_means_and_scales(tmp_path):
    g = gaussians_from_points(np.eye(3), np.full((3, 3), 0.5))
    R = rotation_between(np.array([0, 1.0, 0]), np.array([0, 0, 1.0]))
    t = np.array([1.0, 2.0, 3.0])
    h = g.transformed(R, t, scale=2.0)
    np.testing.assert_allclose(h.means[1], [1, 2, 5], atol=1e-6)  # (0,1,0) -> (0,0,1)*2 + t
    np.testing.assert_allclose(h.scales, g.scales * 2.0, rtol=1e-5)


def test_store_is_content_addressed(store, tmp_path):
    a = tmp_path / "a.json"
    a.write_text('{"x": 1}')
    r1 = store.put_file(a, ArtifactKind.JSON)
    r2 = store.put_file(a, ArtifactKind.JSON)
    assert r1.sha256 == r2.sha256 and r1.short_id.startswith("art_")
    assert store.local_path(r1).read_text() == '{"x": 1}'
    store.put_named("k/v.json", b"1")
    assert store.get_named("k/v.json") == b"1" and store.get_named("missing") is None
