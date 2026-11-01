"""Execution of agent-written Python.

Contract for the script (also stated in the tool description the model sees):
  * input artifacts are files under ./inputs/<name>.<ext>, listed in ./inputs/manifest.json
  * write results to ./outputs/; every file there becomes a new artifact
  * optionally write ./outputs/metrics.json ({name: number}) to report measurements
  * numpy, scipy, trimesh, plyfile, PIL, skimage and gsforge itself are importable

Isolation:
  docker      production. --network none, read-only root fs, memory/cpu/pids limits, non-root user,
              only the sandbox dir mounted. This is the security boundary.
  subprocess  development only. Same file contract, isolated interpreter (-I), scrubbed env, timeout.
              NOT a security boundary; the static tripwire below only catches obvious mistakes.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from ..core.config import SandboxConfig
from ..core.errors import SandboxError
from ..core.models import ArtifactKind, ArtifactRef
from ..storage.artifacts import ArtifactStore

_TRIPWIRE_MODULES = {"socket", "subprocess", "requests", "urllib", "http", "ftplib", "smtplib", "ctypes",
                     "multiprocessing", "anthropic"}
_TRIPWIRE_CALLS = {"system", "popen", "execv", "execve", "spawnv", "fork", "rmtree", "eval", "exec"}

_KIND_BY_SUFFIX = {".ply": ArtifactKind.MESH, ".obj": ArtifactKind.MESH, ".glb": ArtifactKind.MESH,
                   ".png": ArtifactKind.IMAGE, ".jpg": ArtifactKind.IMAGE, ".json": ArtifactKind.JSON}


def static_tripwire(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for n in names:
                if n.split(".")[0] in _TRIPWIRE_MODULES:
                    problems.append(f"import of '{n}' is not allowed in sandboxed code")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else fn.id if isinstance(fn, ast.Name) else ""
            if name in _TRIPWIRE_CALLS:
                problems.append(f"call to '{name}' is not allowed in sandboxed code")
    return problems


@dataclass
class SandboxResult:
    exit_code: int
    stdout: str
    stderr: str
    outputs: dict[str, ArtifactRef] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    timed_out: bool = False
    code_ref: ArtifactRef | None = None


class Sandbox:
    def __init__(self, cfg: SandboxConfig, store: ArtifactStore, root: Path):
        self.cfg = cfg
        self.store = store
        self.root = root

    def _command(self, box: Path) -> list[str]:
        if self.cfg.backend == "docker":
            return ["docker", "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp",
                    "--memory", f"{self.cfg.memory_mb}m", "--cpus", str(self.cfg.cpus), "--pids-limit", "256",
                    "--user", "1000:1000", "--security-opt", "no-new-privileges",
                    "-v", f"{box.resolve()}:/work", "-w", "/work", self.cfg.docker_image,
                    "python", "-I", "script.py"]
        return [sys.executable, "-I", "script.py"]

    def _env(self) -> dict[str, str]:
        keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE")  # never forward API keys
        env = {k: os.environ[k] for k in keep if k in os.environ}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # -I ignores PYTHONPATH, so make the installed gsforge visible via site-packages only
        return env

    async def run(self, code: str, inputs: dict[str, ArtifactRef]) -> SandboxResult:
        problems = static_tripwire(code)
        if problems:
            raise SandboxError("; ".join(problems))
        box = self.root / f"sbx-{uuid.uuid4().hex[:10]}"
        (box / "inputs").mkdir(parents=True)
        (box / "outputs").mkdir()
        try:
            manifest = {}
            for name, ref in inputs.items():
                src = self.store.local_path(ref)
                dst = box / "inputs" / f"{name}{src.suffix}"
                shutil.copyfile(src, dst)
                manifest[name] = {"path": f"inputs/{dst.name}", "kind": ref.kind.value}
            (box / "inputs" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (box / "script.py").write_text(code, encoding="utf-8")
            code_ref = self.store.put_file(box / "script.py", ArtifactKind.CODE)

            proc = await asyncio.create_subprocess_exec(
                *self._command(box), cwd=str(box), env=self._env(),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            timed_out = False
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=self.cfg.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                out, err = await proc.communicate()
                timed_out = True
            limit = self.cfg.max_output_chars
            result = SandboxResult(exit_code=proc.returncode if not timed_out else -9,
                                   stdout=out.decode("utf-8", "replace")[-limit:],
                                   stderr=err.decode("utf-8", "replace")[-limit:], timed_out=timed_out,
                                   code_ref=code_ref)
            metrics_file = box / "outputs" / "metrics.json"
            if metrics_file.exists():
                try:
                    raw = json.loads(metrics_file.read_text(encoding="utf-8"))
                    result.metrics = {k: float(v) for k, v in raw.items() if isinstance(v, (int, float))}
                except (ValueError, AttributeError):
                    result.stderr += "\n[sandbox] outputs/metrics.json is not a {name: number} object"
                metrics_file.unlink()
            for f in sorted((box / "outputs").iterdir()):
                if f.is_file():
                    kind = _KIND_BY_SUFFIX.get(f.suffix.lower(), ArtifactKind.JSON)
                    result.outputs[f.stem] = self.store.put_file(f, kind, {"source": "sandbox"})
            return result
        finally:
            shutil.rmtree(box, ignore_errors=True)
