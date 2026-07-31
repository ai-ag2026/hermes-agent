from __future__ import annotations

from hermes_cli import config as config_module


def test_save_config_emits_fallback_guidance_once(tmp_path, monkeypatch) -> None:
    """Fallback guidance is appended by save_config, not duplicated in the base block."""
    config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(config_module, "get_config_path", lambda: config_path)
    monkeypatch.setattr(config_module, "ensure_hermes_home", lambda: None)
    monkeypatch.setattr(config_module, "is_managed", lambda: False)

    config_module.save_config({"model": {"provider": "test", "model": "test/model"}})

    rendered = config_path.read_text(encoding="utf-8")
    assert rendered.count("# ── Fallback Model") == 1
