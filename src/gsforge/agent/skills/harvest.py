"""Harvested candidate scripts: runtime data, not source code.

When a repair that used generated code is accepted by the independent QA, the script is recorded in
the artifact store under a *problem signature* (stage + failing checks). Two things follow:
  1. The next time the same signature fails, earlier successful scripts are offered to the agent as
     references, so repeated problems get cheaper and more consistent.
  2. Reviewers periodically go through the most frequent signatures and promote the winners:
     deterministic fixes become tools in gsforge/tools, judgment-heavy procedures become skills in
     this package (see README.md here).
Nothing is promoted automatically: generated code never enters the pipeline or the skill set unreviewed.
These records live in the artifact store (shared by all workers), never in the source tree.
"""

from __future__ import annotations

import json
from typing import Any

from ...core.hashing import stable_key
from ...core.models import ArtifactRef, QAReport, utcnow
from ...storage.artifacts import ArtifactStore


def problem_signature(report: QAReport) -> str:
    names = sorted(c.name for c in report.checks if not c.passed)
    return f"{report.stage}:{'+'.join(names) or 'none'}"


class HarvestLibrary:
    def __init__(self, store: ArtifactStore):
        self.store = store

    def _key(self, signature: str) -> str:
        return f"harvest/{stable_key(signature)[:16]}.json"

    def _load(self, signature: str) -> dict[str, Any]:
        raw = self.store.get_named(self._key(signature))
        return json.loads(raw) if raw else {"signature": signature, "entries": []}

    def record(self, report: QAReport, code_refs: list[ArtifactRef], score_before: float,
               score_after: float, summary: str, job_id: str, instance_id: str) -> None:
        if not code_refs:
            return
        sig = problem_signature(report)
        doc = self._load(sig)
        doc["entries"].append({
            "code": [r.model_dump(mode="json") for r in code_refs],
            "score_before": score_before, "score_after": score_after, "summary": summary,
            "job_id": job_id, "instance_id": instance_id, "at": utcnow().isoformat(),
        })
        doc["entries"] = doc["entries"][-50:]
        self.store.put_named(self._key(sig), json.dumps(doc, indent=2).encode("utf-8"))

    def examples(self, report: QAReport, limit: int = 2, max_chars: int = 6000) -> list[dict[str, str]]:
        """Best earlier scripts for this signature: largest score gain first, newest breaks ties."""
        doc = self._load(problem_signature(report))
        ranked = sorted(doc["entries"], key=lambda e: (e["score_after"] - e["score_before"], e["at"]), reverse=True)
        out = []
        for e in ranked[:limit]:
            ref = ArtifactRef.model_validate(e["code"][-1])
            try:
                code = self.store.local_path(ref).read_text(encoding="utf-8")[:max_chars]
            except FileNotFoundError:
                continue
            out.append({"summary": e["summary"], "code": code})
        return out
