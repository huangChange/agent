"""Domain models shared by pipeline, tools, agent and exporters.

Everything that crosses a process or service boundary is a pydantic model so it can be
serialized into checkpoints, manifests and agent tool schemas without custom code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactKind(str, Enum):
    GAUSSIANS = "gaussians"  # 3DGS ply
    MESH = "mesh"  # internal working mesh
    TEXTURE = "texture"
    IMAGE = "image"  # preview / render
    CAMERAS = "cameras"
    JSON = "json"
    CODE = "code"
    EXPORT = "export"  # deliverable file (glb / ply / urdf / mjcf)
    LOG = "log"


class ArtifactRef(BaseModel):
    """Pointer to an immutable, content-addressed blob in the artifact store."""

    uri: str
    sha256: str
    kind: ArtifactKind
    media_type: str = "application/octet-stream"
    size: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def short_id(self) -> str:
        return f"art_{self.sha256[:12]}"


# ---------------------------------------------------------------- inputs


class SegmentInput(BaseModel):
    """One segmented gaussian group. May contain several physical objects (table + remote)."""

    segment_id: str
    gaussians: ArtifactRef
    label_hint: str | None = None


class CaptureInputs(BaseModel):
    """Original capture. Preferred over gaussian re-renders for texture baking (much sharper)."""

    images_dir: str
    cameras: ArtifactRef  # transforms.json (nerfstudio / instant-ngp convention)
    metric_scale: bool = True  # poses already in meters (SLAM / ARKit); otherwise scale must be estimated


class SceneInputs(BaseModel):
    scene_id: str
    scene_gaussians: ArtifactRef
    scene_mesh: ArtifactRef | None = None  # algorithmic, quality varies -> a candidate, never ground truth
    segments: list[SegmentInput]
    capture: CaptureInputs | None = None
    up_axis: Literal["+x", "-x", "+y", "-y", "+z", "-z", "auto"] = "auto"


class JobSpec(BaseModel):
    job_id: str
    inputs: SceneInputs
    profile: str = "default"  # delivery profile under configs/profiles
    priority: int = 0
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------- QA


class Severity(str, Enum):
    HARD = "hard"  # failing blocks delivery
    SOFT = "soft"  # failing lowers the grade
    INFO = "info"  # recorded only


class QACheck(BaseModel):
    name: str
    value: float | None
    threshold: float | None = None
    comparator: Literal["<=", ">=", "=="] = "<="
    passed: bool
    severity: Severity = Severity.HARD
    detail: str = ""


class Grade(str, Enum):
    A = "A"  # all hard + all soft pass
    B = "B"  # all hard pass, some soft fail
    C = "C"  # deliverable only after human review
    REJECT = "reject"


class QAReport(BaseModel):
    stage: str
    checks: list[QACheck] = Field(default_factory=list)
    previews: list[ArtifactRef] = Field(default_factory=list)

    @property
    def hard_failures(self) -> list[QACheck]:
        return [c for c in self.checks if c.severity == Severity.HARD and not c.passed]

    @property
    def soft_failures(self) -> list[QACheck]:
        return [c for c in self.checks if c.severity == Severity.SOFT and not c.passed]

    @property
    def passed(self) -> bool:
        return not self.hard_failures

    @property
    def score(self) -> float:
        """Weighted pass fraction; used to compare repair candidates against each other."""
        weights = {Severity.HARD: 3.0, Severity.SOFT: 1.0, Severity.INFO: 0.0}
        total = sum(weights[c.severity] for c in self.checks)
        if total == 0:
            return 1.0
        return sum(weights[c.severity] for c in self.checks if c.passed) / total

    def summary(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "score": round(self.score, 4),
            "failures": [
                {"name": c.name, "value": c.value, "threshold": c.threshold, "comparator": c.comparator,
                 "severity": c.severity.value, "detail": c.detail}
                for c in self.checks if not c.passed
            ],
        }


# ---------------------------------------------------------------- instances & assets


class ObjectInstance(BaseModel):
    instance_id: str
    segment_id: str
    label: str | None = None
    gaussians: ArtifactRef
    supported_by: str | None = None
    bbox_min: tuple[float, float, float] | None = None
    bbox_max: tuple[float, float, float] | None = None


class JointType(str, Enum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"
    CONTINUOUS = "continuous"


class Link(BaseModel):
    name: str
    visual_mesh: ArtifactRef
    collision_meshes: list[ArtifactRef] = Field(default_factory=list)
    mass: float = 1.0
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # ixx ixy ixz iyy iyz izz, about the center of mass, in the link frame
    inertia: tuple[float, float, float, float, float, float] = (1e-3, 0.0, 0.0, 1e-3, 0.0, 1e-3)
    friction: float = 0.6
    material_class: str | None = None


class Joint(BaseModel):
    name: str
    type: JointType
    parent: str
    child: str
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower: float | None = None
    upper: float | None = None
    effort: float = 10.0
    velocity: float = 1.0


class AssetModel(BaseModel):
    """Simulation-facing description of one delivered object (rigid = one link, no joints)."""

    name: str
    links: list[Link]
    joints: list[Joint] = Field(default_factory=list)

    @property
    def is_articulated(self) -> bool:
        return bool(self.joints)


class StageStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    REPAIRED = "repaired"  # failed deterministically, fixed by the agent
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    SKIPPED = "skipped"


class StageRecord(BaseModel):
    stage: str
    status: StageStatus = StageStatus.PENDING
    started_at: datetime | None = None
    finished_at: datetime | None = None
    outputs: dict[str, ArtifactRef] = Field(default_factory=dict)
    qa: QAReport | None = None
    agent_trace: ArtifactRef | None = None
    skills_used: list[str] = Field(default_factory=list)  # name@version loaded by the agent
    error: str | None = None
    cost_usd: float = 0.0


class InstanceState(BaseModel):
    """Checkpointed per-instance state. The pipeline resumes after the last finished stage."""

    instance: ObjectInstance
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    current: dict[str, ArtifactRef] = Field(default_factory=dict)  # latest artifact per role: mesh, texture...
    facts: dict[str, Any] = Field(default_factory=dict)  # small derived values: transform, material, mass...
    generated_ratio: float = 0.0  # fraction of surface whose geometry/appearance was synthesized
    grade: Grade | None = None


class Lineage(BaseModel):
    pipeline_version: str
    tool_versions: dict[str, str]
    model_versions: dict[str, str] = Field(default_factory=dict)
    skills_used: list[str] = Field(default_factory=list)  # name@version, across all repaired stages
    input_hashes: dict[str, str]
    config_hash: str


class AssetManifest(BaseModel):
    """Shipped next to every delivered asset. Buyers rely on it for trust and traceability."""

    job_id: str
    scene_id: str
    instance_id: str
    label: str | None
    grade: Grade
    files: dict[str, str]  # role -> relative path inside the delivery folder
    units: Literal["meters"] = "meters"
    glb_up_axis: Literal["+y"] = "+y"  # glTF convention; URDF / MJCF use +z
    bbox_size_m: tuple[float, float, float]
    generated_ratio: float
    articulated: bool = False
    qa: list[dict[str, Any]]
    lineage: Lineage
    created_at: datetime = Field(default_factory=utcnow)
