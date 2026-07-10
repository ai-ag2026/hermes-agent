from __future__ import annotations

import atexit
import signal
from types import SimpleNamespace

import pytest

import cli
from hermes_cli.kanban_db import KANBAN_RATE_LIMIT_EXIT_CODE


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
