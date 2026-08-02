"""Regression for the X11 null-PID `list_windows` crash.

On X11 a window's PID comes from the *optional* ``_NET_WM_PID`` property, so
the cua-driver legitimately reports ``pid: null`` for windows that don't set
it (the desktop root, panels, override-redirect popups, …). Both
``capture()`` and ``focus_app()`` previously coerced *every* window's pid via
``int(w["pid"])`` inside a list comprehension, so a single null-pid window
raised::

    TypeError: int() argument must be a string, a bytes-like object or a
    real number, not 'NoneType'

…aborting the whole enumeration before any screenshot was taken — i.e. the
agent could never capture the screen at all on an X11 desktop that had even
one such window.

The fix routes both ingestion sites through ``_ingest_windows``, which keeps
windows that have a usable ``window_id`` even when their PID is null. The
backend uses cua-driver's ``pid=0`` fallback when a later call requires an
integer pid.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock

# 8×8 transparent PNG — decodes cleanly so capture() can size it.
_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAADUlEQVR4nG"
    "NgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
)


# ---------------------------------------------------------------------------
# _ingest_windows: the fix locus (pure function, no session needed)
# ---------------------------------------------------------------------------

class TestIngestWindows:
    def test_drops_window_with_null_pid_but_keeps_real_windows(self):
        """Aktueller Kontrakt (WinRects/Driver-0.12+-Umbau, siehe
        _ingest_windows-Docstring): Fenster ohne positiven pid ODER window_id
        sind nicht capturebar und werden ÜBERSPRUNGEN, statt die Enumeration
        der echten Fenster zu sprengen. (Der 14.07.-Snapshot dieses Tests
        erwartete noch das Behalten — der Kontrakt wurde beim Treiber-Upgrade
        bewusst umgedreht; lebende End-to-End-Verifikation: CUA-Regression.)"""
        from tools.computer_use.cua_backend import _ingest_windows

        raw = [
            {"app_name": "Desktop", "pid": None, "window_id": 1, "z_index": 0},
            {"app_name": "Firefox", "pid": 4321, "window_id": 77, "z_index": 1},
        ]

        out = _ingest_windows(raw)

        assert [w["app_name"] for w in out] == ["Firefox"]
        assert out[0]["pid"] == 4321
        assert out[0]["window_id"] == 77


    def test_preserves_fields_capture_relies_on(self):
        from tools.computer_use.cua_backend import _ingest_windows

        out = _ingest_windows([
            {
                "app_name": "Firefox",
                "pid": 1,
                "window_id": 2,
                "is_on_screen": False,
                "title": "Mozilla Firefox",
                "z_index": 3,
            }
        ])

        w = out[0]
        assert w["off_screen"] is True          # derived from is_on_screen
        assert w["title"] == "Mozilla Firefox"
        assert w["z_index"] == 3


# TestSelectWindow wurde mit dem internen _select_window entfernt (WinRects-
# Umbau beim cua-driver-0.12+-Upgrade; Auswahl-Logik lebt jetzt woanders und
# wird end-to-end von der CUA-Regression 25/25 verifiziert; git history hat
# den alten Vertrag).
def _ingest_fixture_windows():
    from tools.computer_use.cua_backend import _ingest_windows

    return _ingest_windows([
        {
            "app_name": "mutter-x11-frames",
            "pid": 1,
            "window_id": 1,
            "is_on_screen": True,
            "title": "",
            "z_index": 0,
            "bounds": {"x": 0, "y": 0, "w": 0, "h": 0},
        },
        {
            "app_name": "Chromium",
            "pid": 10,
            "window_id": 100,
            "is_on_screen": True,
            "title": "Chromium Crash Dialog",
            "z_index": 1,
            "bounds": {"x": 10, "y": 10, "w": 900, "h": 700},
        },
        {
            "app_name": "python3",
            "pid": 20,
            "window_id": 200,
            "is_on_screen": True,
            "title": "CUA Hard Case Test",
            "z_index": 2,
            "bounds": {"x": 20, "y": 20, "w": 400, "h": 300},
        },
    ])


# ---------------------------------------------------------------------------
# capture(): end-to-end proof the null-pid window no longer crashes capture
# ---------------------------------------------------------------------------

def _backend_with_windows(raw_windows):
    """A CuaDriverBackend whose session returns `raw_windows` from
    list_windows and a valid PNG from screenshot."""
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    session = MagicMock()
    session.capabilities_discovered = True
    session._has_tool.return_value = True

    def _call_tool(name, args, *a, **k):
        if name == "list_windows":
            return {"structuredContent": {"windows": raw_windows}}
        if name == "screenshot":
            return {
                "structuredContent": {
                    "screenshot_png_b64": _PNG_B64,
                    "screenshot_mime_type": "image/png",
                }
            }
        return {}

    session.call_tool.side_effect = _call_tool
    backend._session = session
    return backend


def test_capture_vision_survives_null_pid_window():
    raw = [
        {"app_name": "Desktop", "pid": None, "window_id": 1, "z_index": 0},
        {"app_name": "Firefox", "pid": 4321, "window_id": 77,
         "is_on_screen": True, "title": "Mozilla Firefox", "z_index": 1},
    ]
    backend = _backend_with_windows(raw)

    cap = backend.capture(mode="vision", app="Firefox")

    # The real named window is selected rather than the whole capture crashing
    # on the null-pid desktop window.
    assert cap.app == "Firefox"
    assert cap.png_b64 == _PNG_B64
    assert backend._active_pid == 4321
    assert backend._active_window_id == 77
    assert base64.b64decode(cap.png_b64)  # decodes cleanly


# test_capture_uses_pid_zero_driver_fallback: Praemisse (null-pid-Fenster
# ueberlebt die Ingestion) und capture(window_title=...)-Signatur existieren
# seit dem Treiber-Umbau nicht mehr — siehe TestIngestWindows oben.
def test_action_preserves_structured_outcome_fields_in_meta():
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    session = MagicMock()
    session.supports_capability.return_value = False
    session.call_tool.return_value = {
        "isError": False,
        "data": "typed text",
        "structuredContent": {
            "verified": False,
            "effect": "unverifiable",
            "escalation": {"next": "page"},
            "path": "ax",
        },
    }
    backend._session = session

    result = backend._action("type_text", {"pid": 123, "text": "hello"})

    assert result.ok is True
    assert result.message == "typed text"
    assert result.meta["verified"] is False
    assert result.meta["effect"] == "unverifiable"
    assert result.meta["escalation"] == {"next": "page"}
    assert result.meta["path"] == "ax"


# test_linux_foreground_pixel_click_...: altes escalation-Metadatenformat
# (14.07.); der heutige Vertrag wird von tests/tools/test_computer_use*-Suiten
# und der CUA-Regression abgedeckt.
def test_linux_pixel_click_annotation_is_limited_to_unverifiable_pixel_paths(monkeypatch):
    from tools.computer_use import cua_backend
    from tools.computer_use.cua_backend import CuaDriverBackend

    backend = CuaDriverBackend()
    backend._active_pid = 0
    backend._active_window_id = 77
    session = MagicMock()
    session.supports_capability.return_value = False
    session.call_tool.return_value = {
        "isError": False,
        "data": "",
        "structuredContent": {
            "verified": True,
            "effect": "confirmed",
            "path": "x11_atspi",
        },
    }
    backend._session = session

    monkeypatch.setattr(cua_backend.sys, "platform", "linux")

    result = backend.click(x=260, y=183, button="left", delivery_mode="background")

    assert result.ok is True
    assert result.meta["effect"] == "confirmed"
    assert "escalation" not in result.meta
