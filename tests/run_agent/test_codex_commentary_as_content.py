"""Opt-in: Codex commentary/analysis text as visible assistant content.

Upstream routes commentary-phase text to the reasoning channel (ea125dd62 +
b3b1e58ad) — correct for a coding agent, fatal for a conversational assistant:
the mid-turn progress narration IS what the operator wants to see, and its
suppression turned long tool runs into silence (measured 2026-07-16: replies
per user turn fell 4.65 -> 1.06 exactly when that routing was deployed; both
gpt-5.5 and gpt-5.6 emit identical ``phase`` on the wire, so this is
model-agnostic).

``codex_commentary_as_content()`` (env ``HERMES_CODEX_COMMENTARY_AS_CONTENT``,
config ``model.codex_commentary_as_content``) flips ONLY where that text
lands. Replay items keep their ``phase``; finish_reason/continuation semantics
stay untouched.
"""

import sys
import types
from types import SimpleNamespace

import pytest

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    monkeypatch.setenv("HERMES_CODEX_COMMENTARY_AS_CONTENT", "1")


class _FakeCreateStream:
    def __init__(self, events):
        self._events = list(events)

    def __iter__(self):
        return iter(self._events)


def _commentary_then_tool_events(text):
    commentary_item = SimpleNamespace(
        type="message",
        phase="commentary",
        status="completed",
        content=[SimpleNamespace(type="output_text", text=text)],
    )
    function_item = SimpleNamespace(
        type="function_call", id="fc_1", call_id="call_1",
        name="terminal", arguments="{}",
    )
    return [
        SimpleNamespace(type="response.created"),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="message", phase="commentary"),
        ),
        SimpleNamespace(type="response.output_text.delta", delta=text),
        SimpleNamespace(type="response.output_item.done", item=commentary_item),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="function_call"),
        ),
        SimpleNamespace(type="response.output_item.done", item=function_item),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed"),
        ),
    ]


def test_flag_resolution_env_and_config(monkeypatch):
    from agent.codex_responses_adapter import codex_commentary_as_content

    monkeypatch.setenv("HERMES_CODEX_COMMENTARY_AS_CONTENT", "1")
    assert codex_commentary_as_content() is True
    monkeypatch.setenv("HERMES_CODEX_COMMENTARY_AS_CONTENT", "0")
    assert codex_commentary_as_content() is False

    # Env absent -> config decides.
    monkeypatch.delenv("HERMES_CODEX_COMMENTARY_AS_CONTENT", raising=False)
    from hermes_cli import config as hermes_config
    monkeypatch.setattr(
        hermes_config, "load_config_readonly",
        lambda: {"model": {"codex_commentary_as_content": True}},
    )
    assert codex_commentary_as_content() is True
    monkeypatch.setattr(hermes_config, "load_config_readonly", lambda: {})
    assert codex_commentary_as_content() is False


def test_consume_stream_collects_commentary_deltas_as_text():
    from agent.codex_runtime import _consume_codex_event_stream

    streamed, reasoning_streamed = [], []
    response = _consume_codex_event_stream(
        _FakeCreateStream(_commentary_then_tool_events("Ich lese zuerst den Report.")),
        model="gpt-5-codex",
        on_text_delta=streamed.append,
        on_reasoning_delta=reasoning_streamed.append,
        commentary_as_text=True,
    )

    # Live streaming before the first tool call is the pre-gate behaviour;
    # the delta must NOT go to the reasoning channel anymore.
    assert reasoning_streamed == []
    assert streamed == ["Ich lese zuerst den Report."]
    assert response.output_text == "Ich lese zuerst den Report."


def test_consume_stream_default_still_routes_to_reasoning():
    """The parameter default is off — callers must opt in explicitly."""
    from agent.codex_runtime import _consume_codex_event_stream

    streamed, reasoning_streamed = [], []
    response = _consume_codex_event_stream(
        _FakeCreateStream(_commentary_then_tool_events("preamble")),
        model="gpt-5-codex",
        on_text_delta=streamed.append,
        on_reasoning_delta=reasoning_streamed.append,
    )

    assert streamed == []
    assert reasoning_streamed == ["preamble"]
    assert response.output_text == ""


def test_normalize_puts_commentary_into_content_and_keeps_replay_phase():
    from agent.codex_responses_adapter import _normalize_codex_response

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                phase="commentary",
                status="completed",
                content=[SimpleNamespace(
                    type="output_text",
                    text="Ich prüfe Änderungszeiten rein lesend.",
                )],
            ),
            SimpleNamespace(
                type="function_call", id="fc_1", call_id="call_1",
                name="terminal", arguments="{}", status="completed",
            ),
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5-codex",
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "tool_calls"
    assert "Ich prüfe Änderungszeiten" in (assistant_message.content or "")
    assert not (assistant_message.reasoning or "")
    # Replay continuity: the exact item keeps its phase for the API.
    assert assistant_message.codex_message_items[0]["phase"] == "commentary"


def test_normalize_commentary_only_stays_incomplete_with_flag_on():
    """The flag moves text, not turn semantics: commentary without a final
    answer and without tool calls is still an unfinished turn."""
    from agent.codex_responses_adapter import _normalize_codex_response

    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                phase="commentary",
                status="completed",
                content=[SimpleNamespace(type="output_text", text="Moment, ich schaue nach.")],
            )
        ],
        usage=SimpleNamespace(input_tokens=4, output_tokens=2, total_tokens=6),
        status="completed",
        model="gpt-5-codex",
    )

    assistant_message, finish_reason = _normalize_codex_response(response)

    assert finish_reason == "incomplete"
    assert "Moment, ich schaue nach." in (assistant_message.content or "")
