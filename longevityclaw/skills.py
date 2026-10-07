"""
LongevityClaw skills: user-defined, repeatable procedures.

A *skill* captures a reusable workflow the agent has performed, so it can be
replayed later by name instead of re-explaining it. The format follows the
convention used by Claude Code, OpenClaw and ClawBio: each skill is a directory
with a ``SKILL.md`` manifest.

    skills/<slug>/SKILL.md

``SKILL.md`` has a small frontmatter block and a free-text body of instructions::

    ---
    name: methylation-triage
    description: Run all clocks on a methylation file, then flag accelerated organs.
    keywords: methylation, triage, acceleration
    created: 2026-06-08T12:00:00
    ---

    1. Call predict_age_from_file on the user's CSV.
    2. For any clock with age gap > 3 years, call interpret_individual ...

These are *prompt skills*: ``run_skill`` returns the recorded instructions back
into the conversation and the agent follows them with the current inputs, rather
than replaying a frozen sequence of tool calls. That keeps skills robust when
data or tool signatures change.
"""

import re
from datetime import datetime
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Turn a skill name into a filesystem-safe slug."""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or "skill"


def _parse_skill_md(text: str) -> dict:
    """Parse a SKILL.md document into its frontmatter fields and body."""
    meta: dict[str, str] = {}
    body = text
    if (m := _FRONTMATTER_RE.match(text)):
        front, body = m.group(1), m.group(2)
        for line in front.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip().lower()] = value.strip()
    keywords = [k.strip() for k in meta.get("keywords", "").split(",") if k.strip()]
    return {
        "name": meta.get("name", ""),
        "description": meta.get("description", ""),
        "keywords": keywords,
        "created": meta.get("created", ""),
        "instructions": body.strip(),
    }


def _render_skill_md(name: str, description: str, instructions: str,
                     keywords: list[str] | None, created: str) -> str:
    kw = ", ".join(keywords) if keywords else ""
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        f"keywords: {kw}\n"
        f"created: {created}\n"
        "---\n\n"
        f"{instructions.strip()}\n"
    )


def list_skills() -> list[dict]:
    """Return metadata for every saved skill (sorted by name)."""
    if not SKILLS_DIR.exists():
        return []
    skills = []
    for skill_md in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        parsed = _parse_skill_md(skill_md.read_text(encoding="utf-8"))
        parsed["slug"] = skill_md.parent.name
        if not parsed["name"]:
            parsed["name"] = parsed["slug"]
        skills.append(parsed)
    return skills


def get_skill(name: str) -> dict | None:
    """Look up a single skill by name or slug (case-insensitive)."""
    target = slugify(name)
    for skill in list_skills():
        if skill["slug"] == target or slugify(skill["name"]) == target:
            return skill
    return None


def save_skill(name: str, description: str, instructions: str,
               keywords: list[str] | None = None) -> dict:
    """Create or overwrite a skill. Returns the saved skill's metadata."""
    if not name or not name.strip():
        raise ValueError("A skill needs a name.")
    if not instructions or not instructions.strip():
        raise ValueError("A skill needs instructions describing what to do.")

    slug = slugify(name)
    skill_dir = SKILLS_DIR / slug
    skill_dir.mkdir(parents=True, exist_ok=True)
    created = datetime.now().isoformat(timespec="seconds")
    content = _render_skill_md(name.strip(), description.strip(),
                               instructions, keywords, created)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return {
        "slug": slug,
        "name": name.strip(),
        "description": description.strip(),
        "keywords": keywords or [],
        "created": created,
        "invoke_with": f"/{slug}",
    }


def delete_skill(name: str) -> bool:
    """Delete a skill by name or slug. Returns True if one was removed."""
    skill = get_skill(name)
    if not skill:
        return False
    skill_dir = SKILLS_DIR / skill["slug"]
    (skill_dir / "SKILL.md").unlink(missing_ok=True)
    try:
        skill_dir.rmdir()
    except OSError:
        pass  # directory not empty (user added files) — leave it
    return True
