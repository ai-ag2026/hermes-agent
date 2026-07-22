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
import sys
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


def test_live_writable_roots_use_the_shared_policy(tmp_path):
    """Same classification as the tripwire: canonical markers decide liveness,
    and only a read-only MOUNT counts as protection — a 0555 directory on a
    writable filesystem does not (TARS review §3.2)."""
    live = tmp_path / "live"
    live.mkdir()
    (live / "state.db").touch()
    partial = tmp_path / "partial"      # no state.db, still live state
    partial.mkdir()
    (partial / "config.yaml").touch()
    empty = tmp_path / "empty"
    empty.mkdir()
    chmod_only = tmp_path / "chmod-only"
    chmod_only.mkdir()
    (chmod_only / "state.db").touch()
    chmod_only.chmod(0o555)
    try:
        exposed = runner.live_writable_roots([live, partial, empty, chmod_only])
        assert exposed == [live, partial, chmod_only]
    finally:
        chmod_only.chmod(0o755)


# ---- CLI boundaries (TARS review §3.5: test them through main()) ----------


def test_cli_refuses_no_sandbox_with_writable_live_root(tmp_path, monkeypatch, capsys):
    store = tmp_path / "live-store"
    store.mkdir()
    (store / "state.db").touch()
    monkeypatch.setenv("HERMES_HERMETIC_PROTECT", str(store))
    monkeypatch.setattr(sys, "argv", ["run_tests_hermetic.py", "--no-sandbox"])
    assert runner.main() == 2
    assert "VERWEIGERT" in capsys.readouterr().err


def test_cli_refuses_log_dir_inside_a_protected_root(tmp_path, monkeypatch, capsys):
    protected = tmp_path / "store"
    protected.mkdir()
    (protected / "state.db").touch()
    monkeypatch.setenv("HERMES_HERMETIC_PROTECT", str(protected))
    monkeypatch.setattr(sys, "argv", [
        "run_tests_hermetic.py", "--log-dir", str(protected / "logs")])
    assert runner.main() == 2
    err = capsys.readouterr().err
    assert "VERWEIGERT" in err and "Bracket" in err


def test_runner_identity_covers_the_whole_trust_set():
    ident = runner.runner_identity()
    assert set(ident) == set(runner.RUNNER_TRUST_SET)
    for rel in ("tests/hermetic_policy.py", "tests/conftest.py",
                "tests/store_guard.py"):
        assert rel in ident, f"{rel} decides isolation and must be pinned"
    assert all(len(h) == 64 for h in ident.values())


def test_runner_identity_changes_with_content(tmp_path):
    fake = tmp_path / "repo"
    for rel in runner.RUNNER_TRUST_SET:
        p = fake / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("v1")
    before = runner.runner_identity(fake)
    (fake / runner.RUNNER_TRUST_SET[0]).write_text("v2")
    assert runner.runner_identity(fake) != before


# ---- worktree attestation (TARS P1-RUN-3) --------------------------------


def _git_repo(tmp_path: Path) -> Path:
    import subprocess as sp
    repo = tmp_path / "wt"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for args in (["init", "-q"], ["add", "."], ):
        sp.run(["git", *args], cwd=repo, env=env, check=True)
    (repo / "f.txt").write_text("v1")
    sp.run(["git", "add", "."], cwd=repo, env=env, check=True)
    sp.run(["git", "commit", "-qm", "init"], cwd=repo, env=env, check=True)
    return repo


def test_worktree_state_reports_clean(tmp_path):
    repo = _git_repo(tmp_path)
    st = runner.worktree_state(repo)
    assert st["state"] == "clean"
    assert st["porcelain"] == ""
    assert len(st["tree_hash"]) == 40


def test_worktree_state_flags_dirty_and_changes_tree_hash(tmp_path):
    repo = _git_repo(tmp_path)
    clean = runner.worktree_state(repo)
    (repo / "f.txt").write_text("v2-uncommitted")
    dirty = runner.worktree_state(repo)
    assert dirty["state"] == "DIRTY"
    assert "f.txt" in dirty["dirty"]
    # The dirty tree hash must differ from the clean one: an uncommitted
    # change under the same HEAD is exactly what P1-RUN-3 is about.
    assert dirty["tree_hash"] != clean["tree_hash"]
    assert dirty["tree_hash"] != "unbekannt (stash create fehlgeschlagen)"


# ---- untracked/ignored attestation (TARS R9, P1-RUN-6) -------------------


def test_untracked_file_changes_the_attestation(tmp_path):
    """An untracked test file runs, but was invisible in the tracked tree.

    ``git stash create`` builds its tree from the index — a dropped-in
    ``tests/test_evil.py`` left ``Tree-Hash`` looking like a clean checkout.
    """
    repo = _git_repo(tmp_path)
    clean = runner.worktree_state(repo)
    assert clean["state"] == "clean"

    evil = repo / "test_evil.py"
    evil.write_text("def test_x():\n    assert True\n")
    evil.chmod(0o755)
    with_evil = runner.worktree_state(repo)

    assert "test_evil.py" in with_evil["untracked"]
    assert with_evil["untracked_hash"] != clean["untracked_hash"]
    # The load-bearing property: the combined attestation must move.
    assert with_evil["attest_hash"] != clean["attest_hash"]


def test_untracked_content_change_changes_the_attestation(tmp_path):
    """Same path, different bytes — the hash is over content, not names."""
    repo = _git_repo(tmp_path)
    (repo / "test_evil.py").write_text("def test_x():\n    assert True\n")
    first = runner.worktree_state(repo)
    (repo / "test_evil.py").write_text("def test_x():\n    raise SystemExit\n")
    second = runner.worktree_state(repo)
    assert second["untracked"] == first["untracked"]  # identical name set
    assert second["attest_hash"] != first["attest_hash"]


def test_ignored_executable_code_is_hashed_but_venvs_are_not(tmp_path):
    """Ignored ``.py`` is attested; an ignored venv is excluded *visibly*."""
    repo = _git_repo(tmp_path)
    (repo / ".gitignore").write_text("ignored_tool.py\n.venv/\n")
    import subprocess as sp
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    sp.run(["git", "add", ".gitignore"], cwd=repo, env=env, check=True)
    sp.run(["git", "commit", "-qm", "ignore"], cwd=repo, env=env, check=True)

    base = runner.worktree_state(repo)
    (repo / "ignored_tool.py").write_text("import os\n")
    with_ignored = runner.worktree_state(repo)
    assert "ignored_tool.py" in with_ignored["ignored_code"]
    assert with_ignored["attest_hash"] != base["attest_hash"]

    # A venv full of .py files must NOT be walked — that is the documented
    # blind spot, and it must stay a *named* exclusion, not a silent one.
    venv = repo / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "whatever.py").write_text("x = 1\n")
    with_venv = runner.worktree_state(repo)
    assert not any(p.startswith(".venv/") for p in with_venv["ignored_code"])
    assert with_venv["attest_hash"] == with_ignored["attest_hash"]
    assert ".venv/" in runner._UNHASHED_IGNORED_PREFIXES


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
