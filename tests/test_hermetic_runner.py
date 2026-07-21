"""Negative tests for the hermetic runner's isolation plan.

The runner is a script, not a package module; it is loaded here via
importlib so the environment scrub, the protected-root discovery and the
bwrap mount plan are pinned by tests instead of by review-time reading.
These exist because the 2026-07-21 TARS review demonstrated concrete gaps:
inherited store pins (``HERMES_KANBAN_DB`` …) surviving into the child env,
``--no-sandbox`` still attesting hermeticity, and a host-writable mount
plan. Each of those must stay dead.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_spec = importlib.util.spec_from_file_location(
    "hermetic_runner", PROJECT_ROOT / "scripts" / "run_tests_hermetic.py")
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)


def _sandbox_dirs(tmp_path: Path) -> dict:
    sub = {name: tmp_path / name for name in
           ("home", "hermes", "xdg-config", "xdg-data", "xdg-state",
            "xdg-cache", "xdg-runtime", "tmp", "basetemp")}
    for p in sub.values():
        p.mkdir(exist_ok=True)
    return sub


INHERITED_PINS = {
    # The exact leak set from the TARS review §3.3 …
    "HERMES_KANBAN_DB": "/real/kanban.db",
    "HERMES_KANBAN_HOME": "/real/kanban",
    "HERMES_KANBAN_WORKSPACES_ROOT": "/real/workspaces",
    "HERMES_MANAGED_DIR": "/real/managed",
    "HERMES_REAL_HOME": "/real/home",
    "XDG_RUNTIME_DIR": "/run/user/1000",
    # … plus credential-shaped things that must never reach a test tree.
    "GITEA_TOKEN": "secret",
    "HERMES_GATEWAY_TOKEN": "secret",
    "PYTEST_ADDOPTS": "-p evil_plugin",
}


def test_child_env_drops_inherited_pins(tmp_path):
    sub = _sandbox_dirs(tmp_path)
    base = {"PATH": "/usr/bin", "LANG": "de_DE.UTF-8", "LC_ALL": "C.UTF-8",
            **INHERITED_PINS}
    env, _ = runner.build_child_env(sub, base_env=base)
    for pin in INHERITED_PINS:
        assert pin not in env or env[pin] != INHERITED_PINS[pin], pin
    assert env["PATH"] == "/usr/bin"
    assert env["LANG"] == "de_DE.UTF-8"
    assert env["LC_ALL"] == "C.UTF-8"


def test_child_env_redirects_every_path_variable(tmp_path):
    sub = _sandbox_dirs(tmp_path)
    env, overrides = runner.build_child_env(sub, base_env={"PATH": "/usr/bin"})
    for var, key in [("HOME", "home"), ("HERMES_HOME", "hermes"),
                     ("XDG_CONFIG_HOME", "xdg-config"),
                     ("XDG_DATA_HOME", "xdg-data"),
                     ("XDG_STATE_HOME", "xdg-state"),
                     ("XDG_CACHE_HOME", "xdg-cache"),
                     ("XDG_RUNTIME_DIR", "xdg-runtime"),
                     ("TMPDIR", "tmp")]:
        assert env[var] == str(sub[key]), var
        assert overrides[var] == str(sub[key]), var


def test_child_env_never_attests(tmp_path):
    """The runner PRODUCES hermeticity; it must never declare it."""
    sub = _sandbox_dirs(tmp_path)
    env, _ = runner.build_child_env(sub, base_env={"HERMES_HERMETIC": "1"})
    assert "HERMES_HERMETIC" not in env


def test_child_env_hands_protected_roots_to_the_tripwire(tmp_path):
    sub = _sandbox_dirs(tmp_path)
    roots = [tmp_path / "a", tmp_path / "b"]
    env, _ = runner.build_child_env(sub, base_env={}, protected=roots)
    assert env["HERMES_HERMETIC_PROTECT"] == os.pathsep.join(map(str, roots))


def test_protected_roots_include_custom_hermes_home(tmp_path):
    custom = tmp_path / "custom-store"
    custom.mkdir()
    roots = runner.protected_roots({"HERMES_HOME": str(custom)})
    assert custom.resolve() in roots


def test_live_writable_roots_detect_only_writable_stores(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    (live / "state.db").touch()
    empty = tmp_path / "empty"
    empty.mkdir()
    ro = tmp_path / "ro"
    ro.mkdir()
    (ro / "state.db").touch()
    ro.chmod(0o555)
    try:
        assert runner.live_writable_roots([live, empty, ro]) == [live]
    finally:
        ro.chmod(0o755)


def test_bwrap_plan_is_default_deny(tmp_path):
    sandbox = tmp_path / "sb"
    sandbox.mkdir()
    protected = [Path("/real/.hermes"), Path("/mnt/hermes-datastore")]
    cmd = runner.build_bwrap_command(
        "bwrap", sandbox, protected, ["python", "-m", "pytest"])
    joined = " ".join(cmd)
    # Host root read-only, fresh /tmp, no --dev-bind of the whole root.
    assert cmd[1:3] == ["--ro-bind", "/"]
    assert "--dev-bind / /" not in joined
    assert "--tmpfs /tmp" in joined
    # Writable: exactly sandbox and repo.
    rw = [cmd[i + 1] for i, a in enumerate(cmd) if a == "--bind"]
    assert rw == [str(sandbox), str(runner.REPO)]
    # Protected roots are bound read-only AFTER the writable binds (later
    # binds win), so they stay read-only even where they overlap.
    last_rw = max(i for i, a in enumerate(cmd) if a == "--bind")
    for root in protected:
        idx = cmd.index(str(root))
        assert cmd[idx - 1] == "--ro-bind" and idx > last_rw, root
    assert "--die-with-parent" in cmd


def test_inventory_uses_nanoseconds_and_content_hashes(tmp_path):
    (tmp_path / "cron").mkdir()
    jobs = tmp_path / "cron" / "jobs.json"
    jobs.write_text('{"jobs": []}')
    plain = tmp_path / "plain.txt"
    plain.write_text("x")
    inv = runner.inventory(tmp_path)
    assert inv["plain.txt"][1] == plain.lstat().st_mtime_ns
    assert len(inv["plain.txt"]) == 2
    # Critical manifests additionally carry a content hash: a same-size,
    # same-mtime rewrite must still show up.
    assert len(inv["cron/jobs.json"]) == 3
    before = inv["cron/jobs.json"]
    st = jobs.lstat()
    jobs.write_text('{"jobs": {}}')  # same size, different content
    os.utime(jobs, ns=(st.st_atime_ns, st.st_mtime_ns))
    after = runner.inventory(tmp_path)["cron/jobs.json"]
    assert before != after
