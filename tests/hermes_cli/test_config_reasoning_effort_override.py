"""CC-PARITY-A3: HERMES_REASONING_EFFORT override folded in at config load."""

from __future__ import annotations

from hermes_cli import config as cfg


def _home_with_config(tmp_path):
    # A real config.yaml so the load cache engages (poison test is meaningful).
    (tmp_path / "config.yaml").write_text("agent:\n  max_turns: 40\n", encoding="utf-8")
    return tmp_path


def test_env_override_injects_valid_both_wrappers(tmp_path, monkeypatch):
    _home_with_config(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REASONING_EFFORT", "high")
    assert cfg.load_config()["agent"]["reasoning_effort"] == "high"
    assert cfg.load_config_readonly()["agent"]["reasoning_effort"] == "high"


def test_env_override_normalises_case(tmp_path, monkeypatch):
    _home_with_config(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_REASONING_EFFORT", "XHigh")
    assert cfg.load_config()["agent"]["reasoning_effort"] == "xhigh"


def test_invalid_override_ignored(tmp_path, monkeypatch):
    _home_with_config(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REASONING_EFFORT", raising=False)
    base = cfg.load_config().get("agent", {}).get("reasoning_effort", "")
    monkeypatch.setenv("HERMES_REASONING_EFFORT", "turbo")
    assert cfg.load_config().get("agent", {}).get("reasoning_effort", "") == base


def test_readonly_does_not_poison_cache(tmp_path, monkeypatch):
    _home_with_config(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_REASONING_EFFORT", raising=False)
    base = cfg.load_config_readonly().get("agent", {}).get("reasoning_effort", "")
    # active override via the shared-cache (readonly) path...
    monkeypatch.setenv("HERMES_REASONING_EFFORT", "high")
    assert cfg.load_config_readonly()["agent"]["reasoning_effort"] == "high"
    # ...must not leak into a subsequent no-override read.
    monkeypatch.delenv("HERMES_REASONING_EFFORT", raising=False)
    assert cfg.load_config_readonly().get("agent", {}).get("reasoning_effort", "") == base
