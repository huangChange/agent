"""Typed configuration: configs/default.yaml, deep-merged with a delivery profile and env overrides.

Env overrides use GSFORGE__SECTION__KEY=value (double underscore as separator), e.g.
GSFORGE__AGENT__MAX_COST_USD=2.5
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from .hashing import stable_key

# repo checkout layout by default; containers mount configs elsewhere and set GSFORGE_CONFIG_DIR
CONFIG_DIR = Path(os.environ.get("GSFORGE_CONFIG_DIR", Path(__file__).resolve().parents[3] / "configs"))


class StorageConfig(BaseModel):
    backend: Literal["local", "s3"] = "local"
    root: str = "./.gsforge/store"  # local backend root, or local cache dir for s3
    bucket: str | None = None
    prefix: str = "gsforge"
    endpoint_url: str | None = None  # S3-compatible stores (MinIO, OSS, COS)


class RuntimeConfig(BaseModel):
    backend: Literal["local", "ray"] = "local"
    cpu_workers: int = 4
    gpu_devices: list[int] = Field(default_factory=list)
    max_concurrent_instances: int = 8
    llm_concurrency: int = 8  # global cap on in-flight LLM requests per worker process
    workdir: str = "./.gsforge/work"


class LLMConfig(BaseModel):
    model: str = "claude-opus-5-5"
    critic_model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    critic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = 64000
    server_side_fallback: bool = True  # Claude API only; disable on Bedrock / Vertex / Foundry
    request_timeout_s: float = 600.0
    max_retries: int = 4
    # USD per million tokens: input, output, cache_read, cache_write
    prices: dict[str, tuple[float, float, float, float]] = Field(
        default_factory=lambda: {
            "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
            "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
            "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
        }
    )


class SandboxConfig(BaseModel):
    backend: Literal["subprocess", "docker"] = "subprocess"
    docker_image: str = "gsforge-sandbox:latest"
    timeout_s: float = 300.0
    memory_mb: int = 8192
    cpus: float = 4.0
    max_output_chars: int = 8000


class AgentConfig(BaseModel):
    enabled: bool = True
    max_turns: int = 24
    max_tool_calls: int = 60
    max_cost_usd: float = 3.0  # per instance per stage
    max_wall_s: float = 1800.0
    preview_views: int = 6
    preview_size: int = 512
    allow_codegen: bool = True
    # extra skill directories (customer- or deployment-specific); built-in skills are always loaded
    skill_dirs: list[str] = Field(default_factory=list)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)


class StagePolicy(BaseModel):
    enabled: bool = True
    on_fail: Literal["agent", "review", "fail"] = "agent"


class GeometryConfig(BaseModel):
    voxel_resolution: int = 192  # cells along the longest bbox axis for density-field extraction
    min_opacity: float = 0.05
    target_faces: int = 60000
    hole_fill_max_edges: int = 200
    candidates: list[str] = Field(default_factory=lambda: ["density_mc", "scene_mesh_crop"])


class TextureConfig(BaseModel):
    size: int = 2048
    prefer_capture_images: bool = True


class ExportConfig(BaseModel):
    formats: list[Literal["glb", "ply", "gaussians_ply", "urdf", "mjcf"]] = Field(
        default_factory=lambda: ["glb", "ply", "gaussians_ply"]
    )
    collision: Literal["none", "convex_hull", "coacd"] = "convex_hull"
    out_dir: str = "./.gsforge/deliveries"


class Settings(BaseModel):
    storage: StorageConfig = Field(default_factory=StorageConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    texture: TextureConfig = Field(default_factory=TextureConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    stages: dict[str, StagePolicy] = Field(default_factory=dict)
    # stage -> metric -> {threshold, comparator, severity, required}
    qa: dict[str, dict[str, dict[str, Any]]] = Field(default_factory=dict)

    def stage_policy(self, stage: str) -> StagePolicy:
        return self.stages.get(stage, StagePolicy())

    @property
    def fingerprint(self) -> str:
        return stable_key(self.model_dump(mode="json"))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _env_overrides(prefix: str = "GSFORGE__") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, raw in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        node = out
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = yaml.safe_load(raw)  # "3" -> 3, "true" -> True, "[0,1]" -> [0, 1]
    return out


def load_settings(profile: str = "default", config_dir: Path | None = None,
                  extra: dict[str, Any] | None = None) -> Settings:
    config_dir = config_dir or CONFIG_DIR
    data: dict[str, Any] = {}
    default_file = config_dir / "default.yaml"
    if default_file.exists():
        data = yaml.safe_load(default_file.read_text(encoding="utf-8")) or {}
    if profile != "default":
        profile_file = config_dir / "profiles" / f"{profile}.yaml"
        if not profile_file.exists():
            raise FileNotFoundError(f"unknown delivery profile '{profile}' ({profile_file})")
        data = _deep_merge(data, yaml.safe_load(profile_file.read_text(encoding="utf-8")) or {})
    data = _deep_merge(data, _env_overrides())
    if extra:
        data = _deep_merge(data, extra)
    return Settings.model_validate(data)
