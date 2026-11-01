"""Skills: reviewed, versioned procedures that the repair agent loads on demand.

A skill is a directory containing SKILL.md (YAML frontmatter + instructions), plus optional
reference files (scripts, examples). The agent sees only the catalog (name + one-line description of
the skills for its stage) in its system prompt, and pulls the full text with the `load_skill` tool when
a skill matches the situation. This keeps context small and the system prompt byte-stable per role,
so the prompt cache keeps hitting.

Definitions are validated at load time: a broken skill fails the worker at startup, not mid-repair.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

from ...core.errors import GsForgeError

BUILTIN_DIR = Path(__file__).parent
KNOWN_STAGES = frozenset({"split", "geometry", "align", "texture", "material", "physics"})
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
RESOURCE_SUFFIXES = frozenset({".py", ".md", ".txt", ".json", ".yaml", ".yml"})
MAX_RESOURCE_CHARS = 20000


class SkillDefinitionError(GsForgeError):
    """A SKILL.md is missing, malformed, or inconsistent with its directory."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    stages: tuple[str, ...]
    version: str
    body: str
    root: Path

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def resources(self) -> dict[str, str]:
        """Text files shipped with the skill (reference scripts, examples), keyed by relative path."""
        out = {}
        for p in sorted(self.root.rglob("*")):
            if p.is_file() and p.name != "SKILL.md" and p.suffix in RESOURCE_SUFFIXES and "__pycache__" not in p.parts:
                text = p.read_text(encoding="utf-8")
                if len(text) > MAX_RESOURCE_CHARS:
                    text = text[:MAX_RESOURCE_CHARS] + "\n# ... truncated"
                out[p.relative_to(self.root).as_posix()] = text
        return out


def parse_skill(skill_md: Path) -> Skill:
    text = skill_md.read_text(encoding="utf-8").replace("\r\n", "\n")
    where = skill_md.parent.name
    if not text.startswith("---\n"):
        raise SkillDefinitionError(f"{where}: SKILL.md must start with a '---' frontmatter block")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise SkillDefinitionError(f"{where}: unterminated frontmatter")
    try:
        meta = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError as e:
        raise SkillDefinitionError(f"{where}: invalid YAML frontmatter: {e}") from None
    body = text[end + 5:].strip()

    name = meta.get("name")
    if name != where:
        raise SkillDefinitionError(f"{where}: frontmatter name '{name}' must equal the directory name")
    if not NAME_RE.match(name):
        raise SkillDefinitionError(f"{where}: name must be lowercase words joined by '-'")
    description = str(meta.get("description") or "").strip()
    if not description or len(description) > 300 or "\n" in description:
        raise SkillDefinitionError(f"{where}: description must be one line of 1-300 characters")
    stages = meta.get("stages")
    if not isinstance(stages, list) or not stages or not set(stages) <= KNOWN_STAGES:
        raise SkillDefinitionError(f"{where}: stages must be a non-empty list from {sorted(KNOWN_STAGES)}")
    version = str(meta.get("version") or "")
    if not VERSION_RE.match(version):
        raise SkillDefinitionError(f"{where}: version must look like 1.2.3")
    if not body:
        raise SkillDefinitionError(f"{where}: SKILL.md has no instructions after the frontmatter")
    return Skill(name=name, description=description, stages=tuple(sorted(stages)), version=version, body=body,
                 root=skill_md.parent)


class SkillRegistry:
    def __init__(self, dirs: Iterable[Path]):
        self._skills: dict[str, Skill] = {}
        for d in dirs:
            d = Path(d)
            if not d.is_dir():
                raise SkillDefinitionError(f"skill directory not found: {d}")
            for sub in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))):
                skill_md = sub / "SKILL.md"
                if not skill_md.exists():
                    continue
                skill = parse_skill(skill_md)
                if skill.name in self._skills:  # explicit beats clever: no silent overrides
                    raise SkillDefinitionError(f"duplicate skill '{skill.name}' in {d} and "
                                               f"{self._skills[skill.name].root.parent}")
                self._skills[skill.name] = skill

    def __len__(self) -> int:
        return len(self._skills)

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError:
            raise KeyError(f"unknown skill '{name}'") from None

    def for_stages(self, stages: Iterable[str]) -> list[Skill]:
        wanted = set(stages)
        return sorted((s for s in self._skills.values() if wanted & set(s.stages)), key=lambda s: s.name)

    def catalog(self, stages: Iterable[str]) -> str | None:
        """System-prompt section listing the skills available to a role (deterministic order)."""
        skills = self.for_stages(stages)
        if not skills:
            return None
        lines = ["## Skills", "",
                 "Reviewed procedures for recurring problems. When one matches what you observe, load it with "
                 "load_skill before acting and follow it, unless the evidence in front of you contradicts it "
                 "(then say why in your summary).", ""]
        lines += [f"- {s.name}: {s.description}" for s in skills]
        return "\n".join(lines)


@cache
def _load(extra_dirs: tuple[str, ...]) -> SkillRegistry:
    return SkillRegistry([BUILTIN_DIR, *(Path(d) for d in extra_dirs)])


def load_skills(extra_dirs: Iterable[str] = ()) -> SkillRegistry:
    """Built-in skills (this package) plus deployment-specific directories (agent.skill_dirs)."""
    return _load(tuple(extra_dirs))
