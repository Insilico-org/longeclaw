"""
Filesystem access policy for LongevityClaw tools.

All agent file I/O is confined to a small set of allowed directories and gated by
a policy tier, so the same code path is safe whether LongevityClaw runs as a
local single-user CLI (read/write) or behind the multi-user web server (off).

Policy is read from the environment variable ``LONGEVITYCLAW_FS``:

    readwrite   read + list + write + edit     (default for the local CLI)
    readonly    read + list only
    off         no filesystem tools at all     (the web server sets this)

By default the agent may only touch two directories inside the repo:

    workspace/   the user's scratch/output area (relative paths resolve here)
    skills/      saved skill definitions

Additional locations can be opened up two ways:
  * statically, via ``LONGEVITYCLAW_ALLOWED`` (os.pathsep-separated paths), or
  * at runtime, via :func:`grant_path` — the hook a future interactive
    "grant access to this folder?" prompt (Claude-Code style) will call.

Every path the agent supplies is resolved (collapsing ``..`` and symlinks) and
verified to live under one of the allowed roots before any file is touched.
"""

import os
from pathlib import Path

# Policy tiers, ordered by capability.
READWRITE = "readwrite"
READONLY = "readonly"
OFF = "off"
_VALID_POLICIES = (READWRITE, READONLY, OFF)

# Guard rails for read/write operations.
MAX_READ_BYTES = 1_000_000      # refuse to inline files larger than ~1 MB
MAX_WRITE_BYTES = 5_000_000     # refuse to write more than ~5 MB in one call
MAX_LIST_ENTRIES = 500          # cap directory listings

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Extra paths (files or directories) opened up at runtime by an interactive
# grant. Persisted only for the lifetime of the process.
_granted_paths: set[Path] = set()


class FsAccessError(Exception):
    """Raised when a filesystem operation is denied by policy or confinement."""


def get_policy() -> str:
    """Return the active filesystem policy tier (defaults to ``readwrite``)."""
    policy = os.environ.get("LONGEVITYCLAW_FS", READWRITE).strip().lower()
    return policy if policy in _VALID_POLICIES else READWRITE


def can_read() -> bool:
    return get_policy() in (READWRITE, READONLY)


def can_write() -> bool:
    return get_policy() == READWRITE


def get_workspace_root() -> Path:
    """Return the primary workspace directory (relative paths resolve here).

    Defaults to ``<repo>/workspace`` and is created on demand. Overridable with
    ``LONGEVITYCLAW_WORKSPACE`` (e.g. to point the agent at a project dir).
    """
    raw = os.environ.get("LONGEVITYCLAW_WORKSPACE")
    root = (Path(raw).expanduser() if raw else _REPO_ROOT / "workspace").resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_skills_root() -> Path:
    """Return the skills directory (always within the allowed roots)."""
    return (_REPO_ROOT / "skills").resolve()


def grant_path(path: str | Path) -> Path:
    """Open up access to an additional file or directory at runtime.

    This is the hook an interactive permission prompt will call once the user
    approves access to a location outside the default workspace.
    """
    resolved = Path(path).expanduser().resolve()
    _granted_paths.add(resolved)
    return resolved


def revoke_path(path: str | Path) -> None:
    """Withdraw a previously granted path."""
    _granted_paths.discard(Path(path).expanduser().resolve())


def get_allowed_roots() -> list[Path]:
    """Return every directory/file the agent is currently allowed to access."""
    roots = [get_workspace_root(), get_skills_root()]
    for entry in os.environ.get("LONGEVITYCLAW_ALLOWED", "").split(os.pathsep):
        if entry.strip():
            roots.append(Path(entry.strip()).expanduser().resolve())
    roots.extend(sorted(_granted_paths))
    return roots


def _is_allowed(resolved: Path, roots: list[Path]) -> bool:
    """True if ``resolved`` is one of the roots or lives under a root directory."""
    for root in roots:
        if resolved == root:
            return True
        if root.is_dir() and root in resolved.parents:
            return True
    return False


def resolve_in_workspace(path: str) -> Path:
    """Resolve ``path`` (absolute or workspace-relative) and confine it.

    Returns the fully resolved :class:`Path`. Raises :class:`FsAccessError`
    if the resolved target falls outside every allowed root (via ``..`` or
    symlinks, or simply because it was never granted).
    """
    if not path or not str(path).strip():
        raise FsAccessError("Empty path.")

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = get_workspace_root() / candidate

    # resolve() collapses ".." and follows symlinks, so the containment check
    # below cannot be fooled by traversal or links pointing outside the roots.
    resolved = candidate.resolve()
    roots = get_allowed_roots()

    if not _is_allowed(resolved, roots):
        allowed = ", ".join(str(r) for r in roots)
        raise FsAccessError(
            f"Path '{path}' is outside the allowed directories. "
            f"Access is currently limited to: {allowed}. "
            "Ask the user to grant access to this location (e.g. /grant <path>) "
            "if you need it."
        )
    return resolved


def relative_to_workspace(resolved: Path) -> str:
    """Return ``resolved`` relative to whichever default root contains it."""
    for base in (get_workspace_root(), get_skills_root()):
        try:
            return str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(resolved)
