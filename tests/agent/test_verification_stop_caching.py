"""Verification-loop synthetic scaffolding must never reach durable session state.

verify_on_stop / pre_verify inject a synthetic user nudge to keep the agent
going one more turn before it can claim completion. The assistant response is
real content that persists and is emitted to the UI as an interim message.
Only the nudge (the synthetic user message) is flagged, so only the nudge
gets stripped from the durable transcript. This test file verifies:

  - The verification-loop flags remain registered in
    ``_EPHEMERAL_SCAFFOLDING_FLAGS`` (so nudges are stripped).
  - The DB flush drops only the nudge, keeping the assistant candidate.
  - The JSON log drops only the nudge, keeping the assistant candidate.
"""

import json
import os
import sys
import textwrap
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_child(tmp_path, body: str) -> None:
    """Kopplungs-Backlog 31.07.: Frisch-Import im KINDPROZESS statt Purge.

    Das alte ``_fresh_run_agent`` löschte ``run_agent``/``agent.*``/
    ``tools.*``/``hermes_*`` aus ``sys.modules`` des geteilten Workers —
    nachweislich ein Kopplungs-Täter (Paar-Beweis im Strang
    ``26-test-kopplung-20260730``), und der Rücktausch der Originale hat es
    verdoppelt statt behoben. Ein frischer Prozess hat den frischen
    Importgraphen per Definition; alle Assertions laufen im Kind, ein
    Fehlschlag kommt als Exitcode + stderr zurück.
    """
    import subprocess

    script = (
        "import json, os, sys\n"
        f"os.environ['HERMES_HOME'] = {str(tmp_path / '.hermes')!r}\n"
        f"sys.path.insert(0, {_REPO_ROOT!r})\n"
        "from unittest.mock import MagicMock\n"
        "import run_agent as ra\n"
        f"tmp_path = {str(tmp_path)!r}\n"
        "from pathlib import Path\n"
        "tmp_path = Path(tmp_path)\n"
    ) + body
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert proc.returncode == 0, (
        f"child failed rc={proc.returncode}\n"
        f"stdout tail: {proc.stdout[-1500:]}\nstderr tail: {proc.stderr[-2500:]}"
    )


_MAKE_AGENT_SNIPPET = """
def _make_agent(session_id):
    agent = ra.AIAgent(
        session_id=session_id,
        api_key="test-key",
        base_url="http://127.0.0.1:8000/v1",
        provider="openai-compat",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._session_db = MagicMock()
    agent._session_db_created = True
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path / "logs"
    agent.logs_dir.mkdir(parents=True, exist_ok=True)
    return agent
"""


def test_verification_flags_registered_as_ephemeral(tmp_path):
    _run_child(tmp_path, """
assert "_verification_stop_synthetic" in ra._EPHEMERAL_SCAFFOLDING_FLAGS
assert "_pre_verify_synthetic" in ra._EPHEMERAL_SCAFFOLDING_FLAGS
assert ra._is_ephemeral_scaffolding(
    {"role": "user", "content": "[System: run tests]", "_pre_verify_synthetic": True}
)
assert ra._is_ephemeral_scaffolding(
    {"role": "user", "content": "[System: run tests]", "_verification_stop_synthetic": True}
)
assert not ra._is_ephemeral_scaffolding({"role": "user", "content": "hi"})
assert not ra._is_ephemeral_scaffolding({"role": "assistant", "content": "premature done"})
""")


def test_db_flush_drops_only_nudge_keeps_candidate(tmp_path):
    """The assistant candidate is NOT flagged synthetic, so it persists.
    Only the nudge (flagged synthetic) is dropped from the DB flush."""
    _run_child(tmp_path, textwrap.dedent(_MAKE_AGENT_SNIPPET) + """
agent = _make_agent("sess_db")
messages = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "premature done"},
    {"role": "user", "content": "[System: run tests]", "_verification_stop_synthetic": True},
    {"role": "assistant", "content": "verified and clean"},
]
agent._flush_messages_to_session_db(messages, conversation_history=[])
persisted = [
    kwargs.get("content")
    for _args, kwargs in agent._session_db.append_message.call_args_list
]
assert "hi" in persisted
assert "verified and clean" in persisted
assert "premature done" in persisted
assert "[System: run tests]" not in persisted
""")


def test_json_log_drops_only_nudge_keeps_candidate(tmp_path):
    """The assistant candidate is NOT flagged synthetic, so it persists in the
    JSON log. Only the nudge (flagged synthetic) is dropped."""
    _run_child(tmp_path, textwrap.dedent(_MAKE_AGENT_SNIPPET) + """
agent = _make_agent("sess_json")
messages = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "premature done"},
    {"role": "user", "content": "[System: run tests]", "_pre_verify_synthetic": True},
    {"role": "assistant", "content": "verified and clean"},
]
agent._save_session_log(messages)
log_file = agent.logs_dir / "session_sess_json.json"
assert log_file.exists()
data = json.loads(log_file.read_text(encoding="utf-8"))
contents = [m.get("content") for m in data["messages"]]
assert "premature done" in contents
assert "verified and clean" in contents
assert "hi" in contents
assert "[System: run tests]" not in contents
assert all(not m.get("_pre_verify_synthetic") for m in data["messages"])
""")
