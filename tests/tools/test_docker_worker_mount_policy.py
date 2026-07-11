from __future__ import annotations

from tools.environments.docker import _worker_mount_is_control_plane


def test_worker_rejects_mount_of_control_plane(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    assert _worker_mount_is_control_plane(str(home)) is True
    assert _worker_mount_is_control_plane(str(tmp_path)) is True


def test_worker_allows_narrow_task_workspace(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    workspace = home / "kanban" / "workspaces" / "t_test"
    workspace.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test")
    assert _worker_mount_is_control_plane(str(workspace)) is False


def test_interactive_session_keeps_existing_mount_behavior(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    assert _worker_mount_is_control_plane(str(tmp_path)) is False
