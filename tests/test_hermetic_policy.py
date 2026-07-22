"""The isolation policy itself — tests/hermetic_policy.py.

These pin the two P0s from the second TARS review (2026-07-21):

* directory permissions are not protection (a ``0555`` directory still lets
  an existing ``state.db`` inside it be truncated), so only a read-only
  *mount* may be accepted;
* an ambient custom ``HERMES_HOME`` must be found without anyone passing
  ``HERMES_HERMETIC_PROTECT``.

Plus the fail-open the same review found: liveness must not hinge on
``state.db`` alone, or a partially damaged store — exactly when the tripwire
matters most — would be classified as safe.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests import hermetic_policy as policy


def _store(root: Path, marker: str = "state.db") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    target = root / marker
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    return root


# ---- root discovery ------------------------------------------------------


def test_ambient_hermes_home_is_found_without_protect_var(tmp_path):
    custom = _store(tmp_path / "profile-store")
    roots = policy.candidate_roots({"HERMES_HOME": str(custom)},
                                   home=tmp_path / "empty-home")
    assert custom.resolve() in roots


def test_protect_var_is_additive_not_a_precondition(tmp_path):
    ambient = _store(tmp_path / "ambient")
    extra = _store(tmp_path / "extra")
    roots = policy.candidate_roots(
        {"HERMES_HOME": str(ambient), policy.PROTECT_ENV: str(extra)},
        home=tmp_path / "empty-home")
    assert ambient.resolve() in roots and extra.resolve() in roots


def test_passwd_home_is_always_a_candidate(tmp_path):
    _store(tmp_path / ".hermes")
    roots = policy.candidate_roots({}, home=tmp_path)
    assert (tmp_path / ".hermes").resolve() in roots


def test_real_home_comes_from_passwd_not_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    try:
        import pwd
        expected = Path(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):
        pytest.skip("no passwd database on this platform")
    assert policy.passwd_home() == expected != tmp_path


# ---- liveness ------------------------------------------------------------


@pytest.mark.parametrize("marker", policy.LIVE_MARKERS)
def test_any_canonical_marker_makes_a_root_live(tmp_path, marker):
    assert policy.is_live(_store(tmp_path / marker.replace("/", "_"), marker))


def test_store_without_state_db_is_still_live(tmp_path):
    """The fail-open: a partial store keeps profiles/config and must be
    protected even though state.db is gone."""
    root = _store(tmp_path / "damaged", "config.yaml")
    _store(root, "profiles")
    assert not (root / "state.db").exists()
    assert policy.is_live(root)


def test_empty_root_is_not_live(tmp_path):
    (tmp_path / "fresh-sandbox").mkdir()
    assert not policy.is_live(tmp_path / "fresh-sandbox")


# ---- protection ----------------------------------------------------------


def test_chmod_0555_directory_is_not_protection(tmp_path):
    """TARS' counter-evidence, encoded: the directory refuses new entries but
    the database inside it stays writable and truncatable."""
    root = _store(tmp_path / "chmod-only")
    db = root / "state.db"
    db.write_bytes(b"xx")
    root.chmod(0o555)
    try:
        with db.open("ab") as fh:  # the damage the tripwire must prevent
            fh.write(b"y")
        assert db.stat().st_size == 3, "file inside a 0555 dir stayed writable"
        assert not policy.is_readonly_mount(root)
        assert policy.refusal_reason({policy.PROTECT_ENV: str(root)},
                                     home=tmp_path / "empty") is not None
    finally:
        root.chmod(0o755)


def test_readonly_mount_is_the_accepted_protection():
    """The real thing: under the runner's bwrap --ro-bind the canonical store
    is a read-only mount, and that alone makes the run acceptable."""
    real = policy.passwd_home() / ".hermes"
    if not policy.is_live(real):
        pytest.skip("no live store on this machine")
    if not policy.is_readonly_mount(real):
        pytest.skip("not running under a read-only bind (i.e. not via the runner)")
    assert policy.refusal_reason({}, home=policy.passwd_home()) is None


def test_readonly_mount_actually_rejects_writes_with_erofs():
    """P1-RUN-2: prove the accepted protection is REAL kernel read-only, not
    just a flag reading True. Under the runner's bwrap --ro-bind, an append
    to an existing file in the canonical store must fail with EROFS.

    This targets the live store on purpose: the write is guaranteed to fail
    (that is the whole point), so it cannot damage anything. Outside the
    runner (store writable) it skips — it must never run against a writable
    real store."""
    import errno
    real = policy.passwd_home() / ".hermes"
    if not policy.is_readonly_mount(real):
        pytest.skip("not under a read-only bind — nothing to prove here")
    victim = real / "config.yaml"
    if not victim.exists():
        pytest.skip("no existing file to attempt a write against")
    with pytest.raises(OSError) as ei:
        with victim.open("ab") as fh:
            fh.write(b"\0")
            fh.flush()
    assert ei.value.errno == errno.EROFS, (
        f"expected EROFS, got {errno.errorcode.get(ei.value.errno)}")


# ---- refusal decision ----------------------------------------------------


def test_writable_live_root_is_refused(tmp_path):
    _store(tmp_path / ".hermes")
    reason = policy.refusal_reason({}, home=tmp_path)
    assert reason is not None and "ABGEBROCHEN" in reason


def test_ambient_custom_store_is_refused_without_protect_var(tmp_path):
    """P0 §3.3: passwd home clean, ambient HERMES_HOME live and writable."""
    custom = _store(tmp_path / "custom")
    reason = policy.refusal_reason({"HERMES_HOME": str(custom)},
                                   home=tmp_path / "no-home")
    assert reason is not None and str(custom) in reason


def test_no_live_root_means_no_refusal(tmp_path):
    assert policy.refusal_reason({}, home=tmp_path) is None


def test_attestation_flag_is_dead(tmp_path):
    _store(tmp_path / ".hermes")
    assert policy.refusal_reason({"HERMES_HERMETIC": "1"}, home=tmp_path) is not None


def test_legacy_allow_unhermetic_override_is_dead(tmp_path):
    """P0-RUN-1: the free-settable escape hatch is gone. The old magic string
    must NOT excuse a writable live store — it is no different from the
    abolished HERMES_HERMETIC=1."""
    _store(tmp_path / ".hermes")
    env = {"HERMES_ALLOW_UNHERMETIC": "yes-i-accept-the-risk"}
    reason = policy.refusal_reason(env, home=tmp_path)
    assert reason is not None and "ABGEBROCHEN" in reason


def test_no_override_env_names_remain_in_the_module():
    """Guard against a quiet reintroduction of any override constant."""
    assert not hasattr(policy, "OVERRIDE_ENV")
    assert not hasattr(policy, "OVERRIDE_VALUE")


# ---- symlink / unresolvable roots (P2-RUN-4) -----------------------------


def test_symlink_to_live_root_is_still_protected(tmp_path):
    real = _store(tmp_path / "real-store")
    link = tmp_path / "link-store"
    link.symlink_to(real)
    reason = policy.refusal_reason({"HERMES_HOME": str(link)},
                                   home=tmp_path / "empty")
    assert reason is not None and "ABGEBROCHEN" in reason


def test_dangling_root_is_simply_absent_not_a_refusal(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert policy.refusal_reason({"HERMES_HOME": str(missing)},
                                 home=tmp_path / "empty") is None


def test_symlink_cycle_is_rejected_deterministically(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.symlink_to(b)
    b.symlink_to(a)
    assert policy.unresolvable_candidates({"HERMES_HOME": str(a)},
                                          home=tmp_path / "empty")
    reason = policy.refusal_reason({"HERMES_HOME": str(a)},
                                   home=tmp_path / "empty")
    assert reason is not None and "nicht auflösbar" in reason
