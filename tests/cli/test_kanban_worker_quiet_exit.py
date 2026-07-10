from __future__ import annotations

import atexit
import os
import signal
from types import SimpleNamespace

import pytest

# Importing ``cli`` runs load_hermes_dotenv() at module level (cli.py), which
# merges the developer's real ~/.hermes/.env into os.environ for the whole
# pytest process — before any fixture (including the hermetic-environment
# conftest) can intervene. Snapshot the environment, import, then roll back
# the pollution: these tests monkeypatch HermesCLI away and need none of it.
_pre_import_env = dict(os.environ)
import cli  # noqa: E402

for _k in set(os.environ) - set(_pre_import_env):
    del os.environ[_k]
for _k, _v in _pre_import_env.items():
    if os.environ.get(_k) != _v:
        os.environ[_k] = _v
del _pre_import_env

from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_platform_registry():
    """cli.main() triggers real plugin discovery, which registers platform
    adapters (e.g. ntfy) in the process-global platform_registry. Later test
    modules that read the live GatewayConfig (dashboard home-channel tests)
    would then see platforms from the developer's real config.yaml. Snapshot
    and restore the registry so this file leaves no cross-file state behind."""
    from gateway.config import Platform
    from gateway.platform_registry import platform_registry

    saved_entries = dict(platform_registry._entries)
    saved_deferred = dict(platform_registry._deferred)
    # Platform._missing_ memoizes runtime-registered platforms as pseudo
    # members directly in the enum's lookup maps; those outlive the registry
    # restore and must be rolled back too.
    saved_value_map = dict(Platform._value2member_map_)
    saved_member_map = dict(Platform._member_map_)
    saved_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved_env)
    platform_registry._entries.clear()
    platform_registry._entries.update(saved_entries)
    platform_registry._deferred.clear()
    platform_registry._deferred.update(saved_deferred)
    Platform._value2member_map_.clear()
    Platform._value2member_map_.update(saved_value_map)
    Platform._member_map_.clear()
    Platform._member_map_.update(saved_member_map)


class _FakeKanbanWorkerCLI:
    def __init__(self, result: dict):
        self.session_id = "worker-session"
        self.system_prompt = ""
        self.preloaded_skills = []
        self.conversation_history = []
        self.provider = "test-provider"
        self.model = "test-model"
        self._active_agent_route_signature = "same"
        self.agent = SimpleNamespace(
            session_id="worker-session",
            quiet_mode=False,
            suppress_status_output=False,
            stream_delta_callback=object(),
            tool_gen_callback=object(),
            run_conversation=lambda **_kwargs: result,
        )

    def _claim_active_session(self, *_args, **_kwargs):
        return True

    def _ensure_runtime_credentials(self):
        return True

    def _resolve_turn_agent_config(self, _message):
        return {
            "signature": "same",
            "model": self.model,
            "runtime": None,
            "request_overrides": None,
        }

    def _init_agent(self, **_kwargs):
        return True


def _run_quiet_worker(monkeypatch, *, reason: str, kanban: bool) -> int:
    result = {
        "final_response": "provider unavailable",
        "completed": False,
        "failed": True,
        "error": "quota wall",
        "failure_reason": reason,
    }
    fake_cli = _FakeKanbanWorkerCLI(result)

    monkeypatch.setattr(cli, "HermesCLI", lambda **_kwargs: fake_cli)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)
    if kanban:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "t_rate_limit")
        # Body/image enrichment is best-effort and irrelevant to this producer
        # contract; force its DB lookup to fail closed without touching a board.
        from hermes_cli import kanban_db as kb

        monkeypatch.setattr(kb, "connect", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("isolated test")))
    else:
        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)

    with pytest.raises(SystemExit) as exc:
        cli.main(query="work kanban task t_rate_limit", toolsets="kanban", quiet=True)
    assert isinstance(exc.value.code, int)
    return exc.value.code


@pytest.mark.parametrize("reason", ["rate_limit", "billing"])
def test_quiet_kanban_worker_maps_provider_quota_to_tempfail(monkeypatch, reason):
    assert _run_quiet_worker(monkeypatch, reason=reason, kanban=True) == KANBAN_RATE_LIMIT_EXIT_CODE


def test_quiet_non_kanban_failure_keeps_generic_exit(monkeypatch):
    assert _run_quiet_worker(monkeypatch, reason="rate_limit", kanban=False) == 1


# ---------------------------------------------------------------------------
# S4: the goal-mode ``_run_turn`` closure must classify a mid-loop provider
# failure instead of silently swallowing it into an empty/ordinary turn (the
# "externe Barrieren/Cooldowns verbrauchen Turns" defect). This exercises the
# real cli.py wiring (not goals.py's own injected-callback unit tests) by
# capturing the ``run_turn`` callback handed to ``goals.run_kanban_goal_loop``
# and driving it directly against a stateful fake agent.
# ---------------------------------------------------------------------------

@pytest.fixture
def _goal_mode_task(tmp_path, monkeypatch):
    """A real, temp kanban DB with one goal_mode task — cli.py's
    ``_run_kanban_goal_loop_q`` does its own DB round-trips (get_task,
    task_status, block_task), so a fake connect() would have to reimplement
    too much of kanban_db to be worth it."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)

    from hermes_cli import kanban_db as kb

    kb.init_db()
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="goal task", body="do the thing", assignee="default",
            goal_mode=True, goal_max_turns=10,
        )
        kb.claim_task(conn, tid)
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    return tid


def _capture_run_turn(monkeypatch):
    """Monkeypatch goals.run_kanban_goal_loop to a stub that captures the
    ``run_turn`` kwarg cli.py wires in, without running the real loop."""
    from hermes_cli import goals

    captured = {}

    def _fake_loop(**kwargs):
        captured["run_turn"] = kwargs["run_turn"]
        return {"outcome": "stopped", "turns_used": 1, "reason": "test stub"}

    monkeypatch.setattr(goals, "run_kanban_goal_loop", _fake_loop)
    return captured


def test_goal_mode_run_turn_raises_transient_on_rate_limit(monkeypatch, _goal_mode_task):
    from hermes_cli.goals import KanbanTransientTurnError

    captured = _capture_run_turn(monkeypatch)
    result = {"final_response": "first turn ok", "completed": True, "failed": False}
    fake_cli = _FakeKanbanWorkerCLI(result)
    monkeypatch.setattr(cli, "HermesCLI", lambda **_kwargs: fake_cli)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit):
        cli.main(query="work kanban task", toolsets="kanban", quiet=True)

    assert "run_turn" in captured
    fake_cli.agent.run_conversation = lambda **_kwargs: {
        "final_response": "",
        "completed": False,
        "failed": True,
        "error": "provider returned 429",
        "failure_reason": "rate_limit",
    }
    with pytest.raises(KanbanTransientTurnError):
        captured["run_turn"]("continue please")


def test_goal_mode_run_turn_raises_deterministic_on_format_error(monkeypatch, _goal_mode_task):
    from hermes_cli.goals import KanbanTurnError, KanbanTransientTurnError

    captured = _capture_run_turn(monkeypatch)
    result = {"final_response": "first turn ok", "completed": True, "failed": False}
    fake_cli = _FakeKanbanWorkerCLI(result)
    monkeypatch.setattr(cli, "HermesCLI", lambda **_kwargs: fake_cli)
    monkeypatch.setattr(cli, "_finalize_single_query", lambda _cli: None)
    monkeypatch.setattr(atexit, "register", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(signal, "signal", lambda *_args, **_kwargs: None)

    with pytest.raises(SystemExit):
        cli.main(query="work kanban task", toolsets="kanban", quiet=True)

    assert "run_turn" in captured
    fake_cli.agent.run_conversation = lambda **_kwargs: {
        "final_response": "",
        "completed": False,
        "failed": True,
        "error": "bad request shape",
        "failure_reason": "format_error",
    }
    with pytest.raises(KanbanTurnError) as exc:
        captured["run_turn"]("continue please")
    assert not isinstance(exc.value, KanbanTransientTurnError)
