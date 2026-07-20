"""Resolve ``work_uid`` to a board database -- fail-closed on ambiguity.

D0 gave every board an immutable ``board_uuid`` and made ``work_uid`` look
like ``kanban:<board_uuid>:<task_id>``. That made references *mintable*. It
did not make them *resolvable*: nothing mapped a uuid back to a file. This
module closes that half.

The uncomfortable case, carried over from D0
--------------------------------------------
A file copy duplicates the uuid. That is **correct** for a restore -- the
restored board is the same logical board and its references must keep
working. It is **wrong** for a clone meant to run alongside the original.
From inside the file the two are indistinguishable, which is why ADR-1 puts
re-identification in an explicit pre-mount tool step.

The resolver therefore cannot decide which of two identical uuids is "the
real one", and it must not try. Picking either is a coin flip that silently
routes work to the wrong board; picking by mtime or path order is the same
coin flip with a plausible-sounding rule attached. So on a duplicate the
resolver **refuses both write paths** and raises an attention. Reads are
refused too: a read that silently picks one board tells you something true
about a board you did not ask about.

That is a deliberate availability-for-correctness trade. A duplicate uuid
means the operator's mental model and the filesystem disagree, and continuing
would bury that disagreement under work that lands in the wrong place.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

WORK_UID_RE = re.compile(r"^kanban:(?P<board>[0-9a-fA-F-]{36}):(?P<task>t_[0-9a-f]+)$")


class WorkUidError(Exception):
    """Base for resolution failures."""


class MalformedWorkUid(WorkUidError):
    pass


class UnknownBoard(WorkUidError):
    pass


class AmbiguousBoard(WorkUidError):
    """The same board_uuid was found at more than one path.

    Deliberately fatal for both reads and writes -- see the module docstring.
    """

    def __init__(self, board_uuid: str, paths: Sequence[Path]):
        super().__init__(
            f"board_uuid {board_uuid} found at {len(paths)} paths: "
            + ", ".join(str(p) for p in sorted(paths))
        )
        self.board_uuid = board_uuid
        self.paths = list(paths)


@dataclass(frozen=True)
class ResolvedWork:
    work_uid: str
    board_uuid: str
    task_id: str
    db_path: Path


def parse_work_uid(work_uid: str) -> Tuple[str, str]:
    """Split a work_uid into ``(board_uuid, task_id)``.

    Strict on purpose. A permissive parser that accepted
    ``kanban::t_abc`` or a bare task id would let a caller construct a
    reference that looks resolvable and is not.
    """
    match = WORK_UID_RE.match(work_uid or "")
    if not match:
        raise MalformedWorkUid(f"not a work_uid: {work_uid!r}")
    return match.group("board"), match.group("task")


def _read_board_uuid(db_path: Path) -> Optional[str]:
    """Read a board's uuid without initialising or migrating it.

    Read-only and failure-tolerant: scanning a directory must not mutate
    anything, and one unreadable file must not abort the whole scan. A board
    that predates D0 simply has no ``board_meta`` and is skipped.
    """
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            "SELECT value FROM board_meta WHERE key='board_uuid'"
        ).fetchone()
        return row[0] if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def scan_boards(roots: Sequence[Path]) -> Dict[str, List[Path]]:
    """Map ``board_uuid -> [paths]`` across the given roots.

    The value is a **list**, not a path. Collapsing it to a single path here
    would hide exactly the duplicate this module has to detect.
    """
    found: Dict[str, List[Path]] = {}
    for root in roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        candidates = [root] if root.is_file() else sorted(root.rglob("kanban.db"))
        for db_path in candidates:
            uuid_value = _read_board_uuid(db_path)
            if uuid_value:
                found.setdefault(uuid_value, []).append(db_path)
    return found


def default_board_roots() -> List[Path]:
    from hermes_cli.config import get_hermes_home

    home = get_hermes_home()
    roots = [home / "kanban", home]
    extra = os.environ.get("HERMES_KANBAN_BOARD_ROOTS", "").strip()
    if extra:
        roots.extend(Path(p) for p in extra.split(os.pathsep) if p)
    return roots


def resolve(work_uid: str, *, roots: Optional[Sequence[Path]] = None,
            for_write: bool = False) -> ResolvedWork:
    """Resolve a work_uid to a concrete board database.

    ``for_write`` only affects the log message. Both reads and writes are
    refused on ambiguity -- a read that silently picks one of two boards
    answers a question the caller did not ask.
    """
    board_uuid, task_id = parse_work_uid(work_uid)
    index = scan_boards(roots if roots is not None else default_board_roots())
    paths = index.get(board_uuid) or []

    if not paths:
        raise UnknownBoard(f"no board with uuid {board_uuid} under the search roots")
    if len(paths) > 1:
        logger.error(
            "board_uuid %s resolves to %d paths; refusing %s access. This means "
            "a board file was copied and both copies are mounted. Re-identify "
            "the clone via the audited path, or unmount it.",
            board_uuid, len(paths), "write" if for_write else "read",
        )
        raise AmbiguousBoard(board_uuid, paths)

    return ResolvedWork(work_uid=work_uid, board_uuid=board_uuid,
                        task_id=task_id, db_path=paths[0])


def work_uid_for(conn: sqlite3.Connection, task_id: str) -> Optional[str]:
    """Mint the work_uid for a task on an already-open board."""
    from hermes_cli.kanban_db import build_work_uid, get_board_uuid

    board_uuid = get_board_uuid(conn)
    if not board_uuid:
        return None
    return build_work_uid(board_uuid, task_id)
