"""Read-only Linux process and cgroup-v2 probes for kanban workers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional


def current_cgroup_path(*, proc_root: Path = Path("/proc")) -> Optional[Path]:
    """Resolve this process' unified cgroup-v2 directory."""
    try:
        lines = (proc_root / "self" / "cgroup").read_text(encoding="utf-8").splitlines()
        for line in lines:
            hierarchy, controllers, rel = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                return Path("/sys/fs/cgroup") / rel.lstrip("/")
    except (OSError, ValueError):
        return None
    return None


def _read_int(path: Path) -> Optional[int]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if raw == "max":
            return None
        return int(raw)
    except (OSError, ValueError):
        return None


def process_start_ticks(pid: int, *, proc_root: Path = Path("/proc")) -> Optional[int]:
    """Return Linux /proc stat starttime (field 22), or None."""
    try:
        stat = (proc_root / str(int(pid)) / "stat").read_text(encoding="utf-8")
        tail = stat.rsplit(")", 1)[1].strip().split()
        return int(tail[19])
    except (OSError, ValueError, IndexError):
        return None


def probe_process(
    pid: int,
    *,
    expected_start_ticks: Optional[int] = None,
    proc_root: Path = Path("/proc"),
    platform: Optional[str] = None,
) -> dict[str, Any]:
    """Read process identity/state without signalling it."""
    if (platform or sys.platform) != "linux":
        return {"supported": False, "reason": "unsupported_platform"}
    base = proc_root / str(int(pid))
    start_ticks = process_start_ticks(pid, proc_root=proc_root)
    if start_ticks is None:
        return {"supported": True, "exists": False, "identity_matches": False}
    state = None
    errors: list[str] = []
    try:
        for line in (base / "status").read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                value = line.split(":", 1)[1].strip()
                state = value[:1] or None
                break
    except OSError as exc:
        errors.append(f"status:{type(exc).__name__}")
    try:
        wchan = (base / "wchan").read_text(encoding="utf-8").strip()[:160] or None
    except OSError as exc:
        wchan = None
        errors.append(f"wchan:{type(exc).__name__}")
    return {
        "supported": True,
        "exists": True,
        "start_ticks": start_ticks,
        "identity_matches": (
            expected_start_ticks is None or int(expected_start_ticks) == start_ticks
        ),
        "state": state,
        "wchan": wchan,
        "errors": errors,
    }


def _parse_events(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        parts = line.split()
        if len(parts) == 2:
            try:
                values[parts[0]] = int(parts[1])
            except ValueError:
                continue
    return values


def _parse_pressure(path: Path) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        prefix = parts[0]
        for item in parts[1:]:
            if "=" not in item:
                continue
            key, raw = item.split("=", 1)
            try:
                values[f"{prefix}_{key}"] = int(raw) if key == "total" else float(raw)
            except ValueError:
                continue
    return values


def read_cgroup_snapshot(
    cgroup_path: str | Path, *, platform: Optional[str] = None
) -> dict[str, Any]:
    """Read absolute cgroup-v2 memory/pressure counters (no delta calc).

    Split out of ``probe_cgroup`` so a dispatcher tick that checks many
    tasks against the SAME cgroup (see the aggregate-cgroup note on
    ``probe_cgroup``) can do the filesystem reads once and reuse the
    snapshot; call ``cgroup_deltas_since`` per task to fold in that task's
    own previous-sample baseline.
    """
    if (platform or sys.platform) != "linux":
        return {"supported": False, "reason": "unsupported_platform"}
    base = Path(cgroup_path)
    if not base.is_dir():
        return {"supported": False, "reason": "cgroup_unavailable"}
    current = _read_int(base / "memory.current")
    high = _read_int(base / "memory.high")
    maximum = _read_int(base / "memory.max")
    swap_current = _read_int(base / "memory.swap.current")
    swap_max = _read_int(base / "memory.swap.max")
    events = _parse_events(base / "memory.events")
    pressure = _parse_pressure(base / "memory.pressure")
    ratio = None
    if current is not None and high not in (None, 0):
        ratio = min(1000.0, current / high)
    swap_ratio = None
    if swap_current is not None and swap_max not in (None, 0):
        swap_ratio = min(1000.0, swap_current / swap_max)
    return {
        "supported": True,
        "current": current,
        "high": high,
        "max": maximum,
        "high_ratio": ratio,
        "swap_current": swap_current,
        "swap_max": swap_max,
        "swap_ratio": swap_ratio,
        "events": events,
        "pressure": pressure,
    }


def cgroup_deltas_since(
    snapshot: Mapping[str, Any], previous: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    """Fold a stored ``previous`` sample into monotonic deltas over ``snapshot``.

    ``snapshot`` is the shared, tick-level absolute reading from
    ``read_cgroup_snapshot``; ``previous`` is the per-task baseline (this
    task's own last recorded sample), so the resulting deltas stay
    per-task even though the underlying counters are cgroup-wide.
    """
    if not snapshot.get("supported"):
        return dict(snapshot)
    events = snapshot.get("events") or {}
    pressure = snapshot.get("pressure") or {}
    prev_events = dict((previous or {}).get("events") or {})
    prev_pressure = dict((previous or {}).get("pressure") or {})
    events_delta = {
        key: max(0, value - int(prev_events.get(key, value)))
        for key, value in events.items()
    }
    pressure_delta = {
        key: max(0, int(value) - int(prev_pressure.get(key, value)))
        for key, value in pressure.items()
        if key.endswith("_total")
    }
    return {**snapshot, "events_delta": events_delta, "pressure_delta": pressure_delta}


def probe_cgroup(
    cgroup_path: str | Path,
    *,
    previous: Optional[Mapping[str, Any]] = None,
    platform: Optional[str] = None,
) -> dict[str, Any]:
    """Read bounded cgroup-v2 memory counters and monotonic deltas.

    CROSS-TASK AGGREGATE, NOT PER-WORKER: kanban workers are spawned via
    ``Popen(start_new_session=True)`` and inherit the caller's cgroup, so in
    the live gateway deployment ``cgroup_path`` is ``hermes-gateway.service``
    — the SAME cgroup for the gateway process itself and every worker it has
    spawned. A sample from this function therefore measures memory pressure
    for the whole fleet, not the one task whose PID triggered the probe. A
    memory-hungry neighbour task (or the gateway) can push another task's
    ``resource_stalled`` reading past threshold even though that task's own
    process is healthy — a cross-task false positive.

    Mitigation in ``detect_resource_stalls``: this reading is only ever ANDed
    with a sustained ``D`` (uninterruptible sleep) state observed on the
    SPECIFIC worker PID via ``probe_process`` — a neighbour's memory pressure
    alone never blocks a task. The real fix is per-worker systemd scopes (one
    cgroup per spawned worker) so this function can attribute pressure to a
    single task; that is future work, not implemented here.
    """
    snapshot = read_cgroup_snapshot(cgroup_path, platform=platform)
    return cgroup_deltas_since(snapshot, previous)
