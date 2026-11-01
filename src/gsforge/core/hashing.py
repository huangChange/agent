"""Deterministic hashing used for content addressing and step-cache keys."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> str:
    """Stable JSON: sorted keys, fixed separators. Required for cache keys and prompt-cache stability."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def stable_key(*parts: Any) -> str:
    return sha256_bytes(canonical_json(list(parts)).encode("utf-8"))
