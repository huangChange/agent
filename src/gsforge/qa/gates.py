"""Config-driven quality gates: metrics -> QAReport -> grade.

Thresholds live in configs (qa.<stage>.<metric>) so that each delivery profile / customer can
tighten or relax them without code changes, and so every report records exactly what it was held to.
"""

from __future__ import annotations

from typing import Any

from ..core.models import Grade, QACheck, QAReport, Severity


def _compare(value: float, threshold: float, comparator: str) -> bool:
    if comparator == "<=":
        return value <= threshold
    if comparator == ">=":
        return value >= threshold
    return abs(value - threshold) < 1e-9


def evaluate_gates(stage: str, metrics: dict[str, float], rules: dict[str, dict[str, Any]],
                   extra_checks: list[QACheck] | None = None) -> QAReport:
    checks: list[QACheck] = []
    for name, rule in sorted(rules.items()):
        severity = Severity(rule.get("severity", "hard"))
        comparator = rule.get("comparator", "<=")
        threshold = float(rule["threshold"])
        value = metrics.get(name)
        if value is None:
            if rule.get("required", True):
                checks.append(QACheck(name=name, value=None, threshold=threshold, comparator=comparator,
                                      passed=False, severity=severity, detail="metric was not computed"))
            continue
        checks.append(QACheck(name=name, value=float(value), threshold=threshold, comparator=comparator,
                              passed=_compare(float(value), threshold, comparator), severity=severity,
                              detail=rule.get("detail", "")))
    checks += extra_checks or []
    return QAReport(stage=stage, checks=checks)


def grade(reports: list[QAReport], generated_ratio: float, max_generated_for_a: float = 0.15) -> Grade:
    if any(not r.passed for r in reports):
        return Grade.REJECT
    soft_fail = any(r.soft_failures for r in reports)
    if not soft_fail and generated_ratio <= max_generated_for_a:
        return Grade.A
    return Grade.B
