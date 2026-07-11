from __future__ import annotations

from tools.environments.docker import _worker_mount_is_control_plane


def _worker_env(monkeypatch, home, workspace, task_id="t_test"):
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(workspace))


def test_worker_rejects_mount_of_control_plane(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    workspace = home / "kanban" / "workspaces" / "t_test"
    workspace.mkdir(parents=True)
    _worker_env(monkeypatch, home, workspace)
    assert _worker_mount_is_control_plane(str(home)) is True
    assert _worker_mount_is_control_plane(str(tmp_path)) is True


def test_worker_allows_only_exact_task_workspace(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    workspace = home / "kanban" / "workspaces" / "t_test"
    workspace.mkdir(parents=True)
    _worker_env(monkeypatch, home, workspace)
    assert _worker_mount_is_control_plane(str(workspace)) is False
    assert _worker_mount_is_control_plane(str(workspace / "child")) is True
    assert _worker_mount_is_control_plane(str(workspace.parent)) is True


def test_worker_rejects_sensitive_descendants_and_sibling_workspaces(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    workspace = home / "kanban" / "workspaces" / "t_test"
    sibling = home / "kanban" / "workspaces" / "t_other"
    profile = home / "profiles" / "reviewer"
    credentials = home / ".env"
    workspace.mkdir(parents=True)
    sibling.mkdir()
    profile.mkdir(parents=True)
    credentials.touch()
    _worker_env(monkeypatch, home, workspace)
    assert _worker_mount_is_control_plane(str(profile)) is True
    assert _worker_mount_is_control_plane(str(credentials)) is True
    assert _worker_mount_is_control_plane(str(sibling)) is True


def test_worker_rejects_symlink_to_sensitive_descendant(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    workspace = home / "kanban" / "workspaces" / "t_test"
    profile = home / "profiles" / "reviewer"
    workspace.mkdir(parents=True)
    profile.mkdir(parents=True)
    link = tmp_path / "looks-safe"
    link.symlink_to(profile, target_is_directory=True)
    _worker_env(monkeypatch, home, workspace)
    assert _worker_mount_is_control_plane(str(link)) is True


def test_worker_fails_closed_without_workspace(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACE", raising=False)
    assert _worker_mount_is_control_plane(str(tmp_path)) is True


def test_interactive_session_keeps_existing_mount_behavior(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _worker_mount_is_control_plane(str(tmp_path)) is False
