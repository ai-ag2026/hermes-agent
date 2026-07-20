"""``fs_workspace_grant``: typed, descriptor-enforced filesystem capability.

Why this is not a path check
----------------------------
The obvious implementation is ``realpath(candidate).startswith(root)``. It is
wrong, and the way it is wrong is not theoretical: between the check and the
open, any component of the path can be replaced with a symlink pointing
outside the root. The check passes, the open lands elsewhere. That is a
time-of-check/time-of-use race, and a capability that loses it is decoration.

ADR-7 therefore permits exactly two enforcement strategies, both
**descriptor-based**:

1. Linux ``openat2`` with ``RESOLVE_BENEATH|RESOLVE_NO_SYMLINKS``; or
2. a complete component-wise FD walk -- open the root once, reject ``..``,
   empty and absolute components, open every intermediate component relative
   to the previous descriptor with ``O_NOFOLLOW|O_DIRECTORY``, and perform the
   final operation only through ``*at`` calls against the final descriptor.

``O_NOFOLLOW`` on the final component plus a ``st_dev``/``st_ino`` comparison
is explicitly **not** an acceptable fallback: it says nothing about the
ancestry of the intermediate components. If neither strategy is available the
grant is refused. ``realpath`` survives only as a cheap pre-filter, never as
enforcement.

This module implements strategy 2, which needs nothing beyond ``os.open`` with
``dir_fd`` and works on every Linux Python. ``openat2`` would require a raw
syscall via ctypes for a marginal gain.

Scope in D0
-----------
Deliberately small: ``write`` for the six existing Tier-1 file verbs
(``rm mkdir mv cp touch rmdir``) plus ``write_file``/``patch``, with roots
from a static configuration list. No policy control plane, no ``principal_id``,
no step-up, no revocation generation -- those arrive with H1. **No grant for
code execution at all**: the 2026-07-19 finding that ``pytest`` writes files
is exactly why tool classes are separated rather than covered by one root
flag.

Reads are ungated *as a named transitional exception* for today's
single-tenant personal path. That exception ends at F2/S4: once tenant workers
exist, a customer core could otherwise read foreign workspaces while writes
are already properly separated.

``follow_symlinks`` is fixed at false and is not configurable. An option that
can be set wrongly is not an option when it comes to containment.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# The six Tier-1 file verbs the grant covers today, plus the two Hermes file
# tools that write. Terminal, interpreters and test runners are NOT here --
# they go through operation classes on the sandbox instead.
GRANTED_WRITE_VERBS = frozenset({"rm", "mkdir", "mv", "cp", "touch", "rmdir"})
GRANTED_WRITE_TOOLS = frozenset({"write_file", "patch"})

GRANT_CONFIG_ENV = "HERMES_FS_GRANTS"


class GrantDenied(Exception):
    """Raised when an operation is not covered by an active grant.

    Carries a reason so the caller can route or repair. ADR-7 is explicit that
    a missing capability triggers routing/repair, not a blanket abort -- the
    work may still be doable, just not here or not by this principal.
    """

    def __init__(self, reason: str, *, path: Optional[str] = None):
        super().__init__(reason)
        self.reason = reason
        self.path = path


@dataclass(frozen=True)
class FsWorkspaceGrant:
    grant_id: str
    issued_by: str
    issued_at: float
    roots: Tuple[str, ...]
    expires_at: Optional[float] = None
    review_after: Optional[float] = None
    state: str = "active"
    read: bool = False
    write: bool = False
    max_bytes_per_write: Optional[int] = None
    # Fixed by contract. Present as a field only so that a config trying to
    # set it is visibly rejected rather than silently ignored.
    follow_symlinks: bool = False

    def is_usable(self, now: Optional[float] = None) -> bool:
        now = time.time() if now is None else now
        if self.state != "active":
            return False
        if self.expires_at is not None and now >= self.expires_at:
            return False
        return True

    def unusable_reason(self, now: Optional[float] = None) -> Optional[str]:
        now = time.time() if now is None else now
        if self.state != "active":
            return f"grant {self.grant_id} is {self.state}"
        if self.expires_at is not None and now >= self.expires_at:
            return f"grant {self.grant_id} expired"
        return None


def _parse_grant(raw: dict) -> FsWorkspaceGrant:
    if raw.get("follow_symlinks"):
        # Fail loudly. Silently overriding it would let an operator believe
        # symlink following is on when it is not -- or worse, ship a config
        # that reads as permissive.
        raise ValueError(
            "follow_symlinks is fixed at false by ADR-7 and cannot be enabled"
        )
    roots = tuple(str(Path(r).expanduser()) for r in raw.get("roots") or ())
    if not roots:
        raise ValueError("a grant without roots grants nothing; refusing")
    return FsWorkspaceGrant(
        grant_id=str(raw["grant_id"]),
        issued_by=str(raw.get("issued_by") or "unknown"),
        issued_at=float(raw.get("issued_at") or time.time()),
        roots=roots,
        expires_at=raw.get("expires_at"),
        review_after=raw.get("review_after"),
        state=str(raw.get("state") or "active"),
        read=bool(raw.get("read")),
        write=bool(raw.get("write")),
        max_bytes_per_write=raw.get("max_bytes_per_write"),
    )


def load_grants() -> List[FsWorkspaceGrant]:
    """Load the static grant list (D0: config only, no control plane).

    Resolved at call time -- see ``process_ledger`` for why nothing in this
    programme freezes a path at import.
    """
    raw = os.environ.get(GRANT_CONFIG_ENV, "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        # A malformed grant config must not silently degrade to "no grants",
        # because "no grants" looks identical to "correctly configured with
        # nothing granted".
        raise GrantDenied(f"grant configuration is not valid JSON: {exc}")
    if isinstance(data, dict):
        data = [data]
    return [_parse_grant(item) for item in data]


# --------------------------------------------------------------- FD walk


def _split_relative(root: Path, candidate: Path) -> Sequence[str]:
    """Return the path components of ``candidate`` below ``root``.

    Pure string work and therefore only a **pre-filter**: it establishes what
    we intend to open, not what we will actually reach. The FD walk does the
    proving.
    """
    root_s = os.path.normpath(str(root))
    cand_s = str(candidate)
    if not os.path.isabs(cand_s):
        cand_s = os.path.join(root_s, cand_s)
    cand_s = os.path.normpath(cand_s)
    if cand_s == root_s:
        return ()
    prefix = root_s.rstrip(os.sep) + os.sep
    if not cand_s.startswith(prefix):
        raise GrantDenied(
            f"path is not below the granted root {root_s}", path=str(candidate)
        )
    return tuple(p for p in cand_s[len(prefix):].split(os.sep) if p)


def _walk_to_parent(root: Path, components: Sequence[str]) -> Tuple[int, str]:
    """Open each intermediate component with ``O_NOFOLLOW|O_DIRECTORY``.

    Returns ``(parent_fd, final_name)``. The caller owns ``parent_fd`` and must
    close it. Every ``*at`` operation must go through it -- re-resolving the
    path as a string afterwards would reintroduce the race this walk exists to
    close, including for create, rename and unlink.
    """
    if not components:
        raise GrantDenied("refusing to operate on the root itself", path=str(root))
    for part in components:
        if part in ("", ".", ".."):
            # `..` is rejected rather than normalised: normalising it would
            # make the string form disagree with what the kernel resolves once
            # symlinks are involved.
            raise GrantDenied(f"illegal path component {part!r}", path=str(root))
        if os.path.isabs(part):
            raise GrantDenied("absolute component inside a relative path")

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(str(root), flags)
    except OSError as exc:
        raise GrantDenied(f"granted root is not an openable directory: {exc}",
                          path=str(root))

    try:
        for part in components[:-1]:
            try:
                nxt = os.open(part, flags, dir_fd=fd)
            except OSError as exc:
                os.close(fd)
                if exc.errno in (errno.ELOOP, errno.EMLINK):
                    # A symlink in the middle of the path. This is the exact
                    # attack the walk defends against, so it is reported as a
                    # containment failure, not as a missing file.
                    raise GrantDenied(
                        f"symlinked path component {part!r} is not traversable",
                        path=str(root),
                    )
                raise GrantDenied(f"cannot traverse {part!r}: {exc}", path=str(root))
            os.close(fd)
            fd = nxt
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    return fd, components[-1]


def check_write_access(path: os.PathLike | str, *,
                       grants: Optional[Iterable[FsWorkspaceGrant]] = None,
                       ) -> FsWorkspaceGrant:
    """Prove that ``path`` is writable under an active grant.

    Raises ``GrantDenied`` otherwise. Success means the path was reached
    through a descriptor chain that never followed a symlink -- not merely
    that its string form looked contained.
    """
    grants = list(load_grants() if grants is None else grants)
    if not grants:
        raise GrantDenied("no fs_workspace_grant configured", path=str(path))

    candidate = Path(path).expanduser()
    reasons: List[str] = []
    for grant in grants:
        why_not = grant.unusable_reason()
        if why_not:
            reasons.append(why_not)
            continue
        if not grant.write:
            reasons.append(f"grant {grant.grant_id} does not carry write")
            continue
        for root in grant.roots:
            root_path = Path(root)
            try:
                components = _split_relative(root_path, candidate)
            except GrantDenied as exc:
                reasons.append(exc.reason)
                continue
            fd, _name = _walk_to_parent(root_path, components)
            os.close(fd)
            return grant
    raise GrantDenied(
        "no active grant covers this path: " + "; ".join(reasons[:4]),
        path=str(candidate),
    )


def open_for_write(path: os.PathLike | str, *, flags: int = os.O_WRONLY,
                   mode: int = 0o600,
                   grants: Optional[Iterable[FsWorkspaceGrant]] = None) -> int:
    """Open a file for writing through the descriptor chain.

    The returned descriptor is the *only* safe handle: reopening by path
    afterwards would discard the containment proof.
    """
    grants = list(load_grants() if grants is None else grants)
    if not grants:
        raise GrantDenied("no fs_workspace_grant configured", path=str(path))
    candidate = Path(path).expanduser()
    reasons: List[str] = []
    for grant in grants:
        if not grant.is_usable() or not grant.write:
            reasons.append(grant.unusable_reason() or
                           f"grant {grant.grant_id} does not carry write")
            continue
        for root in grant.roots:
            try:
                components = _split_relative(Path(root), candidate)
            except GrantDenied as exc:
                reasons.append(exc.reason)
                continue
            parent_fd, name = _walk_to_parent(Path(root), components)
            try:
                open_flags = flags | os.O_NOFOLLOW
                if hasattr(os, "O_CLOEXEC"):
                    open_flags |= os.O_CLOEXEC
                return os.open(name, open_flags, mode, dir_fd=parent_fd)
            except OSError as exc:
                if exc.errno == errno.ELOOP:
                    raise GrantDenied(
                        "refusing to write through a symlink", path=str(candidate))
                raise
            finally:
                os.close(parent_fd)
    raise GrantDenied("no active grant covers this path: " + "; ".join(reasons[:4]),
                      path=str(candidate))


def covers_verb(verb: str) -> bool:
    """Whether a shell verb falls into the granted write class at all."""
    return verb in GRANTED_WRITE_VERBS


def describe_denial(exc: GrantDenied) -> dict:
    """Structured denial for routing/repair rather than a blanket abort."""
    return {
        "capability": "fs_workspace_grant",
        "outcome": "denied",
        "reason": exc.reason,
        "path": exc.path,
        "remedy": "issue or widen a grant, or route the work to a principal "
                  "that already holds one",
    }
