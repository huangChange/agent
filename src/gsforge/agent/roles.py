"""Agent roles: one specialised system prompt + tool scope per pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"


@cache
def _read(name: str) -> str:
    return (PROMPTS / f"{name}.md").read_text(encoding="utf-8")


@dataclass(frozen=True)
class Role:
    name: str
    stages: tuple[str, ...]  # registry stages whose tools this role may call
    prompt: str
    effort: str = "high"

    def system_prompt(self) -> str:
        return _read("base") + "\n\n" + _read(self.prompt)


ROLES: dict[str, Role] = {
    "split": Role("segmentation", ("split",), "split"),
    "geometry": Role("geometry", ("geometry",), "geometry", effort="high"),
    "align": Role("geometry", ("geometry", "align"), "geometry"),
    "texture": Role("texture", ("texture",), "texture", effort="high"),
}


def role_for_stage(stage: str) -> Role | None:
    return ROLES.get(stage)


def critic_prompt() -> str:
    return _read("critic")
