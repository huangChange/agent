"""Content-addressed artifact storage.

Artifacts are immutable blobs keyed by sha256. Tools never pass large arrays between processes;
they pass ArtifactRefs and materialize files locally on demand. This is what makes stages
cacheable, resumable and safe to fan out over many workers.
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..core.config import StorageConfig
from ..core.errors import DependencyMissingError
from ..core.hashing import sha256_file
from ..core.models import ArtifactKind, ArtifactRef

_EXTRA_TYPES = {".ply": "application/ply", ".glb": "model/gltf-binary", ".urdf": "application/xml",
                ".xml": "application/xml", ".jsonl": "application/x-ndjson", ".npz": "application/octet-stream"}


def _media_type(path: Path) -> str:
    return _EXTRA_TYPES.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


class ArtifactStore(ABC):
    """Interface. Implementations must be safe for concurrent writers (writes are idempotent by hash)."""

    @abstractmethod
    def put_file(self, path: str | Path, kind: ArtifactKind, meta: dict[str, Any] | None = None) -> ArtifactRef: ...

    @abstractmethod
    def local_path(self, ref: ArtifactRef) -> Path:
        """Return a local, read-only path for the blob (downloading into the cache if needed)."""

    @abstractmethod
    def put_named(self, key: str, data: bytes) -> None:
        """Mutable small objects addressed by key (checkpoints, cache index). Last writer wins."""

    @abstractmethod
    def get_named(self, key: str) -> bytes | None: ...

    def _staging_dir(self) -> str | None:
        """Where put_bytes stages blobs; None = system temp dir."""
        return None

    def put_bytes(self, data: bytes, kind: ArtifactKind, suffix: str,
                  meta: dict[str, Any] | None = None) -> ArtifactRef:
        fd, tmp = tempfile.mkstemp(suffix=suffix, dir=self._staging_dir())
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return self.put_file(tmp, kind, meta)
        finally:
            os.unlink(tmp)

    def put_json(self, obj: Any, kind: ArtifactKind = ArtifactKind.JSON,
                 meta: dict[str, Any] | None = None) -> ArtifactRef:
        return self.put_bytes(json.dumps(obj, indent=2, default=str).encode("utf-8"), kind, ".json", meta)

    def read_json(self, ref: ArtifactRef) -> Any:
        return json.loads(self.local_path(ref).read_text(encoding="utf-8"))


class LocalArtifactStore(ArtifactStore):
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        (self.root / "objects").mkdir(parents=True, exist_ok=True)
        (self.root / "named").mkdir(parents=True, exist_ok=True)
        (self.root / "staging").mkdir(parents=True, exist_ok=True)

    def _staging_dir(self) -> str:
        return str(self.root / "staging")  # same volume as objects/, so the final rename is atomic

    def _object_path(self, digest: str, suffix: str) -> Path:
        return self.root / "objects" / digest[:2] / f"{digest}{suffix}"

    def put_file(self, path: str | Path, kind: ArtifactKind, meta: dict[str, Any] | None = None) -> ArtifactRef:
        path = Path(path)
        digest = sha256_file(path)
        dest = self._object_path(digest, path.suffix.lower())
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
            shutil.copyfile(path, tmp)
            os.replace(tmp, dest)  # atomic; concurrent writers of the same hash converge
        return ArtifactRef(uri=dest.as_uri(), sha256=digest, kind=kind, media_type=_media_type(path),
                           size=dest.stat().st_size, meta=meta or {})

    def local_path(self, ref: ArtifactRef) -> Path:
        suffix = Path(ref.uri).suffix
        p = self._object_path(ref.sha256, suffix)
        if not p.exists():
            raise FileNotFoundError(f"artifact {ref.short_id} not found in {self.root}")
        return p

    def put_named(self, key: str, data: bytes) -> None:
        dest = self.root / "named" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, dest)

    def get_named(self, key: str) -> bytes | None:
        p = self.root / "named" / key
        return p.read_bytes() if p.exists() else None


class S3ArtifactStore(ArtifactStore):
    """S3 / S3-compatible object storage with a local read-through cache."""

    def __init__(self, bucket: str, prefix: str, cache_root: str | Path, endpoint_url: str | None = None):
        try:
            import boto3
        except ImportError as e:  # pragma: no cover - exercised only on cloud workers
            raise DependencyMissingError("boto3", "cloud") from e
        self.s3 = boto3.client("s3", endpoint_url=endpoint_url)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache = LocalArtifactStore(cache_root)

    def _staging_dir(self) -> str:
        return self.cache._staging_dir()

    def _key(self, digest: str, suffix: str) -> str:
        return f"{self.prefix}/objects/{digest[:2]}/{digest}{suffix}"

    def put_file(self, path: str | Path, kind: ArtifactKind, meta: dict[str, Any] | None = None) -> ArtifactRef:
        local = self.cache.put_file(path, kind, meta)
        key = self._key(local.sha256, Path(path).suffix.lower())
        self.s3.upload_file(str(self.cache.local_path(local)), self.bucket, key)
        return local.model_copy(update={"uri": f"s3://{self.bucket}/{key}"})

    def local_path(self, ref: ArtifactRef) -> Path:
        suffix = Path(ref.uri).suffix
        try:
            return self.cache.local_path(ref)
        except FileNotFoundError:
            dest = self.cache._object_path(ref.sha256, suffix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + f".{os.getpid()}.tmp")
            self.s3.download_file(self.bucket, self._key(ref.sha256, suffix), str(tmp))
            os.replace(tmp, dest)
            return dest

    def put_named(self, key: str, data: bytes) -> None:
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.prefix}/named/{key}", Body=data)

    def get_named(self, key: str) -> bytes | None:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=f"{self.prefix}/named/{key}")
        except self.s3.exceptions.NoSuchKey:
            return None
        return obj["Body"].read()


def open_store(cfg: StorageConfig) -> ArtifactStore:
    if cfg.backend == "local":
        return LocalArtifactStore(cfg.root)
    if not cfg.bucket:
        raise ValueError("storage.bucket is required for the s3 backend")
    return S3ArtifactStore(cfg.bucket, cfg.prefix, cfg.root, cfg.endpoint_url)
