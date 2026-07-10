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
