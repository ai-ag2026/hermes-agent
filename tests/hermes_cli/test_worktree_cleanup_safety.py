"""Regression tests for the worktree auto-cleanup safety gates.

Guards the data-loss vector found in the 2026-07-07 re-audit: a worktree in a repo
with NO remote-tracking refs was treated as "fully pushed" and force-reaped, losing
committed-but-unpushed local-only work. The fix makes ``_worktree_has_unpushed_commits``
fail SAFE (return True = keep) when it cannot prove the work is pushed.
"""
from __future__ import annotations

import subprocess

import pytest

from hermes_cli import kanban_db as k


def _git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.email=a@b", "-c", "user.name=a", *args],
        cwd=str(cwd), capture_output=True, text=True, check=True,
    )


def test_no_remote_repo_is_kept(tmp_path):
    """A repo with local commits but no remote cannot be proven pushed -> keep."""
    repo = tmp_path / "noremote"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "commit", "--allow-empty", "-m", "local only")
    assert k._worktree_has_unpushed_commits(str(repo)) is True


def test_pushed_and_clean_is_reapable(tmp_path):
    """A repo whose HEAD is on a remote and clean is safe to reap."""
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(bare)], capture_output=True, check=True)
    repo = tmp_path / "clone"
    subprocess.run(["git", "clone", str(bare), str(repo)], capture_output=True, check=True)
    _git(repo, "commit", "--allow-empty", "-m", "c1")
    _git(repo, "push", "origin", "HEAD")
    assert k._worktree_has_unpushed_commits(str(repo)) is False
    assert k._worktree_is_dirty(str(repo)) is False


def test_remote_but_unpushed_commit_is_kept(tmp_path):
    """A commit not reachable from any remote -> keep."""
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(bare)], capture_output=True, check=True)
    repo = tmp_path / "clone"
    subprocess.run(["git", "clone", str(bare), str(repo)], capture_output=True, check=True)
    _git(repo, "commit", "--allow-empty", "-m", "c1")
    _git(repo, "push", "origin", "HEAD")
    _git(repo, "commit", "--allow-empty", "-m", "not pushed")
    assert k._worktree_has_unpushed_commits(str(repo)) is True


def test_uncommitted_changes_are_dirty(tmp_path):
    repo = tmp_path / "dirty"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "commit", "--allow-empty", "-m", "c1")
    (repo / "scratch.txt").write_text("wip")
    assert k._worktree_is_dirty(str(repo)) is True


def test_bad_path_fails_safe(tmp_path):
    """An unreadable path must fail SAFE on both gates (never reap on uncertainty)."""
    missing = str(tmp_path / "does-not-exist")
    assert k._worktree_has_unpushed_commits(missing) is True
    assert k._worktree_is_dirty(missing) is True
