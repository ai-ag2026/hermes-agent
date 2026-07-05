"""local(tars) carry: config-driven DISPLAY/XAUTHORITY wiring for cua-driver children.

Covers the headless-Linux path where gateway/worker processes have no DISPLAY
and ``computer_use.display`` in config.yaml pins the X target for cua-driver
spawns only (never the whole agent process).
"""
from unittest.mock import patch

from tools.computer_use.cua_backend import cua_driver_child_env


def _cfg(**cu):
    return {"computer_use": cu}


class TestDisplayInjection:
    def test_display_injected_when_absent(self):
        with patch("hermes_cli.config.load_config", return_value=_cfg(display=":0")):
            env = cua_driver_child_env(base_env={})
        assert env["DISPLAY"] == ":0"

    def test_existing_display_not_overridden(self):
        with patch("hermes_cli.config.load_config", return_value=_cfg(display=":0")):
            env = cua_driver_child_env(base_env={"DISPLAY": ":7"})
        assert env["DISPLAY"] == ":7"

    def test_no_config_key_no_injection(self):
        with patch("hermes_cli.config.load_config", return_value=_cfg()):
            env = cua_driver_child_env(base_env={})
        assert "DISPLAY" not in env

    def test_config_failure_is_safe(self):
        with patch("hermes_cli.config.load_config", side_effect=RuntimeError("boom")):
            env = cua_driver_child_env(base_env={})
        assert "DISPLAY" not in env


class TestXauthorityGlob:
    def test_glob_picks_latest_match(self, tmp_path):
        (tmp_path / ".mutter-Xwaylandauth.AAA").write_text("a")
        (tmp_path / ".mutter-Xwaylandauth.BBB").write_text("b")
        pattern = str(tmp_path / ".mutter-Xwaylandauth.*")
        with patch("hermes_cli.config.load_config", return_value=_cfg(display=":0", xauthority=pattern)):
            env = cua_driver_child_env(base_env={})
        assert env["XAUTHORITY"].endswith(".mutter-Xwaylandauth.BBB")

    def test_no_match_no_xauthority(self, tmp_path):
        pattern = str(tmp_path / "does-not-exist.*")
        with patch("hermes_cli.config.load_config", return_value=_cfg(display=":0", xauthority=pattern)):
            env = cua_driver_child_env(base_env={})
        assert "XAUTHORITY" not in env

    def test_existing_xauthority_not_overridden(self, tmp_path):
        (tmp_path / "cookie").write_text("x")
        with patch("hermes_cli.config.load_config", return_value=_cfg(display=":0", xauthority=str(tmp_path / "*"))):
            env = cua_driver_child_env(base_env={"XAUTHORITY": "/keep/me"})
        assert env["XAUTHORITY"] == "/keep/me"
