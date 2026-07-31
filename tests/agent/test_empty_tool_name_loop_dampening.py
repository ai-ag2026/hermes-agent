"""Regression for #47967 — empty-name phantom tool calls.

Weak open models (mimo, nemotron-class) that see tool-call XML/JSON sitting in
file contents or tool output get *primed* and emit their own structured tool
calls that mimic the payload — usually with an empty/whitespace ``name``. Those
calls can't be fuzzy-repaired toward a real tool, so the dispatch loop returns an
error and the model retries. Before this fix, every empty-name error dumped the
full tool catalog back to the model, which fed the priming loop more names to
mimic and inflated context 3-4x across the retry budget.

The fix: a blank/whitespace-only tool name gets a terse anti-priming error that
tells the model in-context tool-call syntax is DATA, with NO catalog dump. A
genuinely-wrong-but-nonempty name (an actual typo) still gets the full catalog
so the model can self-correct.

These assert the *behavior contract* of the dispatch branch (what content goes
back to the model for each name shape), exercised end-to-end through
``AIAgent.run_conversation`` against an in-process mock provider — not a snapshot
of the message string.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

# Repo root = three levels up from tests/agent/<file>.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


class _MockHandler(BaseHTTPRequestHandler):
    # Set by the fixture before each request cycle.
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        is_stream = req.get("stream") is True
        # Kopplungs-Backlog 31.07.: der Agent im KINDPROZESS initialisiert sich
        # erst beim Testaufruf — sein Init-Probe-POST kommt mit LEEREN
        # ``messages`` und darf die vom Test vorbereitete Antwort-Queue nicht
        # auffressen (im alten In-Prozess-Aufbau entstand der Agent vor dem
        # Befüllen der Queue, da war die Reihenfolge zufällig richtig). Die
        # Queue wird nur für echte Konversations-Requests bedient.
        has_user = any(
            isinstance(m, dict) and m.get("role") == "user"
            for m in (req.get("messages") or [])
        )
        if type(self).response_queue and has_user:
            resp = type(self).response_queue.pop(0)
        else:
            resp = _text_resp("DONE")
        msg = resp["choices"][0]["message"]
        if is_stream:
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"id": "m", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}]
            if content:
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
            if tcs:
                for ti, tc in enumerate(tcs):
                    chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"tool_calls": [{
                        "index": ti, "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}]}, "finish_reason": None}]})
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls" if tcs else "stop"}]})
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):  # silence the default stderr logging
        pass


def _tc_resp(name: str, args: str = "{}") -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": name, "arguments": args}}]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _batch_tc_resp(calls: list[tuple[str, str]]) -> dict:
    """Multi-call batch response: calls = [(name, arguments), ...]."""
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {
            "role": "assistant", "content": "",
            "tool_calls": [
                {"id": f"call_{i}", "type": "function",
                 "function": {"name": name, "arguments": args}}
                for i, (name, args) in enumerate(calls)
            ]},
            "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


def _text_resp(text: str) -> dict:
    return {
        "id": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
    }


@pytest.fixture()
def agent_env():
    """Spin up the mock provider + an isolated HERMES_HOME, yield (agent, helpers)."""
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    test_home = tempfile.mkdtemp(prefix="hermes_e2e_47967_")
    os.makedirs(os.path.join(test_home, ".hermes"))

    # Kopplungs-Backlog 31.07.: der Agent läuft in einem KINDPROZESS statt
    # nach einem sys.modules-Purge im geteilten Worker. Der Purge vergiftete
    # nachweislich Nachbardateien (Paar mit test_skill_commands: 6 failed auf
    # unberührtem Baum), und der Versuch, die Originalmodule danach
    # zurückzutauschen, hat es verdoppelt (12 failed) — Modul-Rücktausch
    # repariert kein Split-Brain. Ein frischer Prozess hat den frischen
    # Importgraphen per Definition; der Mock-Server bleibt hier im Eltern-
    # prozess, das Kind spricht ihn über HTTP an und liefert das
    # run_conversation-Ergebnis als JSON über stdout zurück.
    class _SubprocessAgent:
        def __init__(self):
            self.valid_tool_names = {
                "terminal", "read_file", "write_file", "execute_code", "session_search"
            }

        def run_conversation(self, prompt, conversation_history=None, task_id="t"):
            import subprocess
            script = (
                "import json, os, sys\n"
                f"os.environ['HERMES_HOME'] = {os.path.join(test_home, '.hermes')!r}\n"
                f"sys.path.insert(0, {_REPO_ROOT!r})\n"
                "from run_agent import AIAgent\n"
                "agent = AIAgent(\n"
                f"    api_key='test-key', base_url='http://127.0.0.1:{port}/v1',\n"
                "    provider='openai-compat', model='test-model',\n"
                "    max_iterations=10, enabled_toolsets=[],\n"
                "    quiet_mode=True, skip_context_files=True, skip_memory=True,\n"
                "    save_trajectories=False, platform='cli',\n"
                ")\n"
                f"agent.valid_tool_names = set({sorted(self.valid_tool_names)!r})\n"
                f"result = agent.run_conversation({prompt!r}, conversation_history=[], task_id={task_id!r})\n"
                "print('\\n__RESULT__' + json.dumps("
                "{'partial': bool(result.get('partial', False)), 'messages': result.get('messages') or [], 'err': str(result.get('error') or '')[:300]}"
                ", default=str))\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=180,
            )
            assert proc.returncode == 0, (
                f"agent child failed rc={proc.returncode}\n"
                f"stdout tail: {proc.stdout[-2000:]}\nstderr tail: {proc.stderr[-2000:]}"
            )
            marker = proc.stdout.rsplit("__RESULT__", 1)
            assert len(marker) == 2, f"no result marker in child stdout: {proc.stdout[-500:]}"
            return json.loads(marker[1])

    agent = _SubprocessAgent()

    try:
        yield agent, _MockHandler
    finally:
        srv.shutdown()
        shutil.rmtree(test_home, ignore_errors=True)


def _tool_results(handler) -> list[str]:
    out = []
    for req in handler.captured_requests:
        for m in req.get("messages", []):
            if m.get("role") == "tool":
                out.append(m.get("content", ""))
    return out


@pytest.mark.parametrize("blank", ["", "   ", "\n", "\t "])
def test_empty_tool_name_gets_terse_error_no_catalog(agent_env, blank):
    """A blank/whitespace tool name must NOT trigger a full tool-catalog dump."""
    agent, handler = agent_env
    handler.response_queue.append(_tc_resp(blank, "{}"))
    handler.response_queue.append(_text_resp("Recovered in plain text."))

    agent.run_conversation("read ./payload and report", conversation_history=[], task_id="t")

    joined = " ".join(_tool_results(handler))
    assert "tool name was empty" in joined
    # The whole point: do not feed the priming loop the catalog of names.
    assert "Available tools:" not in joined




# ── Mixed batches: valid calls execute, invalid calls get error results ──
#
# Degrading models (observed with gpt-5.6 past ~350K input; jonny's July 2026
# report) emit batches like 6 named calls + 1 blank-name call. Before the fix,
# the whole turn was voided ("Skipped: another tool call in this turn used an
# invalid name") and three such batches halted the session as partial even
# though most of the model's work was coherent.




def test_mixed_batch_preserves_tool_call_result_pairing(agent_env):
    """Every emitted tool_call keeps a matching tool result (provider invariant)."""
    agent, handler = agent_env
    agent.valid_tool_names = agent.valid_tool_names | {"todo"}
    handler.response_queue.append(_batch_tc_resp([("todo", "{}"), ("", "{}")]))
    handler.response_queue.append(_text_resp("done"))

    result = agent.run_conversation("track work", conversation_history=[], task_id="t")

    msgs = result["messages"]
    tc_ids = []
    for m in msgs:
        if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls"):
            tc_ids.extend(tc["id"] for tc in m["tool_calls"])
    result_ids = [
        m.get("tool_call_id") or "" for m in msgs
        if isinstance(m, dict) and m.get("role") == "tool"
    ]
    # Both the valid and blank call must appear in the assistant message,
    # and each must have exactly one matching tool result.
    assert set(tc_ids) == {"call_0", "call_1"}
    assert sorted(result_ids) == sorted(tc_ids)






def test_invalid_tool_exhaustion_closes_tool_tail(agent_env):
    """Invalid-tool 3-strike partial must not leave a durable tool→user tail (#48879 class).

    Retries <3 append assistant+error tool rows, so the transcript already ends
    on ``tool`` before the exhaustion early-return. That return must close the
    sequence (same contract as interrupt aborts) so the next user turn is not
    ``tool → user`` for strict providers.
    """
    agent, handler = agent_env
    for _ in range(3):
        handler.response_queue.append(_tc_resp("frobnicate_xyz", "{}"))

    result = agent.run_conversation("degenerate", conversation_history=[], task_id="t")

    assert result.get("partial", False)
    msgs = result.get("messages") or []
    assert msgs, "expected persisted conversation messages"
    assert msgs[-1].get("role") == "assistant"
    assert "invalid tool call" in (msgs[-1].get("content") or "").lower()


