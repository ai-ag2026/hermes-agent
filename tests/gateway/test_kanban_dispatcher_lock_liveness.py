"""The dispatcher singleton lock must never outlive the dispatcher.

Two liveness edges found in the TARS re-review 2026-07-27 (P2):

1. The lock is taken BEFORE the 5 s startup grace, and that grace was not
   covered by any handler. A cancellation there (gateway restart, shutdown
   during boot) left the flock held by a watcher that no longer runs — no
   other gateway could take over, and this one stayed dark until the process
   exited. ``_finish_dispatcher`` cannot help: it is defined much later, so
   for the whole early window there was nothing to call.

2. A restart landing inside the maintenance watchdog's release window got one
   "contended" answer and stood down for good, although the lock frees up
   seconds later.

These tests drive the real watcher coroutine; they mock only the lock
primitives, the config and ``asyncio.sleep``.
"""

from __future__ import annotations

import asyncio

import pytest


def _make_runner(monkeypatch, lock_answers, *, cancel_on_sleep):
    """Wire a GatewayRunner whose lock handshake is scripted.

    ``lock_answers`` is consumed per ``_acquire_singleton_lock`` call.
    ``cancel_on_sleep`` is the 1-based index of the ``asyncio.sleep`` call that
    raises ``CancelledError`` — the stand-in for a gateway shutdown.
    """
    from gateway.run import GatewayRunner
    import gateway.kanban_watchers as watchers
    import hermes_cli.config as config

    runner = object.__new__(GatewayRunner)
    runner._running = True

    monkeypatch.setattr(
        config, "load_config",
        lambda: {"kanban": {"dispatch_in_gateway": True,
                            "dispatch_interval_seconds": 1}},
    )
    monkeypatch.setattr(watchers, "_root_dispatch_frozen", lambda: False)

    handle = object()
    acquired: list[str] = []

    def fake_acquire(_path):
        state = lock_answers[min(len(acquired), len(lock_answers) - 1)]
        acquired.append(state)
        return (handle if state == "held" else None), state

    released: list[object] = []
    monkeypatch.setattr(watchers, "_acquire_singleton_lock", fake_acquire)
    monkeypatch.setattr(watchers, "_release_singleton_lock", released.append)

    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        slept.append(delay)
        if len(slept) == cancel_on_sleep:
            raise asyncio.CancelledError()
        await real_sleep(0)

    monkeypatch.setattr(watchers.asyncio, "sleep", fake_sleep)
    return runner, handle, acquired, released, slept


def test_cancel_in_startup_grace_hands_the_lock_back(monkeypatch):
    """The decisive one: cancelled during the 5 s grace → lock released."""
    runner, handle, acquired, released, slept = _make_runner(
        monkeypatch, ["held"], cancel_on_sleep=1,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner._kanban_dispatcher_watcher())

    assert slept == [5], "the cancellation must hit the startup grace itself"
    assert released == [handle], (
        "a dispatcher that never started must not keep the singleton lock"
    )
    assert runner._kanban_dispatcher_lock_handle is None


def test_contended_lock_is_retried_before_standing_down(monkeypatch):
    """A watchdog release window must not cost this gateway its dispatcher."""
    runner, handle, acquired, released, slept = _make_runner(
        monkeypatch, ["contended", "contended", "held"], cancel_on_sleep=3,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner._kanban_dispatcher_watcher())

    assert acquired == ["contended", "contended", "held"], (
        "one contended answer must not end the attempt"
    )
    assert slept[:2] == [15.0, 15.0], "retries wait between attempts"
    assert slept[2] == 5, "after acquiring, the normal startup grace runs"
    assert released == [handle]


def test_contended_forever_stands_down_without_holding_anything(monkeypatch):
    """The retry budget is bounded — and giving up releases nothing wrongly."""
    runner, _handle, acquired, released, slept = _make_runner(
        monkeypatch, ["contended"], cancel_on_sleep=0,  # never cancel
    )

    asyncio.run(runner._kanban_dispatcher_watcher())

    assert len(acquired) == 21, "300 s budget in 15 s steps, plus the first try"
    assert all(state == "contended" for state in acquired)
    assert released == [], "nothing was ever acquired, so nothing is released"
    assert runner._kanban_dispatcher_lock_handle is None


def test_unavailable_lock_does_not_retry(monkeypatch):
    """'unavailable' means locking cannot be performed at all — no point retrying."""
    runner, _handle, acquired, released, slept = _make_runner(
        monkeypatch, ["unavailable"], cancel_on_sleep=1,
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner._kanban_dispatcher_watcher())

    assert acquired == ["unavailable"], "no retry loop on a non-lockable filesystem"
    assert slept == [5], "straight to the startup grace"
    assert released == [], "there is no handle to release"
