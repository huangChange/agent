from __future__ import annotations

from pathlib import Path

import pytest

from gsforge.core.config import load_settings
from gsforge.evals.synthetic import make_table_scene
from gsforge.storage.artifacts import LocalArtifactStore
from gsforge.tools.base import ToolContext
from gsforge.tools.registry import load_builtin_tools


@pytest.fixture()
def settings(tmp_path: Path):
    return load_settings(extra={
        "storage": {"root": str(tmp_path / "store")},
        "runtime": {"workdir": str(tmp_path / "work"), "cpu_workers": 2},
        "export": {"out_dir": str(tmp_path / "deliveries")},
        "geometry": {"voxel_resolution": 96, "target_faces": 8000},
        "texture": {"size": 256},
    })


@pytest.fixture()
def store(settings):
    return LocalArtifactStore(settings.storage.root)


@pytest.fixture()
def ctx(store, settings, tmp_path):
    load_builtin_tools()
    return ToolContext(store=store, settings=settings, workdir=tmp_path / "ctx")


@pytest.fixture(scope="session")
def table_scene(tmp_path_factory):
    return make_table_scene(tmp_path_factory.mktemp("synthetic"), density=25000.0)
