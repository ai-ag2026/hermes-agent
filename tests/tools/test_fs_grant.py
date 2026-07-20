"""``fs_workspace_grant``: containment must survive an active attacker.

The interesting tests here are not "an allowed path is allowed" but the
escape attempts. Measured against the symlink case below:

* a **string prefix** check allows it -- straightforwardly broken;
* ``O_NOFOLLOW`` on the **final component only** allows it, because the link
  sits in the middle and the last component is a perfectly ordinary file;
* ``realpath`` on the parent **does** catch this particular case.

So these tests do not, on their own, prove the FD walk beats ``realpath``.
What they prove is that the two genuinely naive strategies fail. The case
``realpath`` cannot handle is the *racing* one: an attacker who swaps a
component between the check and the open. That is not expressible as a
deterministic unit test -- it is closed by construction, because the walk
resolves the path exactly once into a descriptor chain and every subsequent
operation goes through ``*at`` calls against that chain, never through the
path string again.

Stated plainly so a later reader does not over-read this file: ADR-7 mandates
descriptor enforcement for the race, and the tests below cover the static
escapes plus the grant lifecycle.
"""

from __future__ import annotations

import json
import os

import pytest

from tools import fs_grant


def _grant(root, **kw):
    payload = {
        "grant_id": "g1",
        "issued_by": "test",
        "roots": [str(root)],
        "write": True,
    }
    payload.update(kw)
    return json.dumps([payload])


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "file.txt").write_text("inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("must stay unreachable")
    monkeypatch.setenv(fs_grant.GRANT_CONFIG_ENV, _grant(root))
    return root, outside


# ------------------------------------------------------------- the basics


def test_path_inside_the_root_is_allowed(workspace):
    root, _ = workspace
    assert fs_grant.check_write_access(root / "sub" / "file.txt").grant_id == "g1"


def test_path_outside_the_root_is_denied(workspace):
    _, outside = workspace
    with pytest.raises(fs_grant.GrantDenied):
        fs_grant.check_write_access(outside / "secret.txt")


def test_no_grant_configured_denies(tmp_path, monkeypatch):
    monkeypatch.delenv(fs_grant.GRANT_CONFIG_ENV, raising=False)
    with pytest.raises(fs_grant.GrantDenied):
        fs_grant.check_write_access(tmp_path / "anything")


def test_a_read_only_grant_does_not_authorise_writes(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    (root / "a").mkdir(parents=True)
    monkeypatch.setenv(fs_grant.GRANT_CONFIG_ENV,
                       _grant(root, write=False, read=True))
    with pytest.raises(fs_grant.GrantDenied):
        fs_grant.check_write_access(root / "a" / "f.txt")


# ------------------------------------------------- the attacks that matter


def test_symlink_inside_the_root_cannot_escape_it(workspace):
    """The TOCTOU case, in its simplest form.

    ``realpath`` on the *parent* would say "inside the root". Opening through
    the link would land outside. The FD walk refuses to traverse it at all.
    """
    root, outside = workspace
    (root / "escape").symlink_to(outside)

    with pytest.raises(fs_grant.GrantDenied) as exc:
        fs_grant.check_write_access(root / "escape" / "secret.txt")
    assert "symlink" in str(exc.value).lower() or "traverse" in str(exc.value).lower()


def test_symlinked_intermediate_component_is_refused(workspace):
    """Why ``O_NOFOLLOW`` on the final component alone is not enough.

    The link sits in the middle of the path. A check that only guards the last
    component -- even with a ``st_dev``/``st_ino`` comparison -- proves nothing
    about how it got there.
    """
    root, outside = workspace
    (outside / "deep").mkdir()
    (outside / "deep" / "target.txt").write_text("out of bounds")
    (root / "sub" / "hop").symlink_to(outside / "deep")

    with pytest.raises(fs_grant.GrantDenied):
        fs_grant.check_write_access(root / "sub" / "hop" / "target.txt")


def test_dotdot_traversal_is_refused(workspace):
    root, _ = workspace
    with pytest.raises(fs_grant.GrantDenied):
        fs_grant.check_write_access(root / "sub" / ".." / ".." / "outside" / "s.txt")


def test_writing_through_a_symlinked_file_is_refused(workspace):
    """A link at the final component must not become a write to its target."""
    root, outside = workspace
    (root / "sub" / "link.txt").symlink_to(outside / "secret.txt")

    with pytest.raises(fs_grant.GrantDenied):
        fd = fs_grant.open_for_write(root / "sub" / "link.txt")
        os.close(fd)
    assert (outside / "secret.txt").read_text() == "must stay unreachable"


def test_open_for_write_reaches_a_real_file_inside_the_root(workspace):
    root, _ = workspace
    fd = fs_grant.open_for_write(root / "sub" / "file.txt", flags=os.O_WRONLY)
    try:
        os.write(fd, b"ok")
    finally:
        os.close(fd)
    assert (root / "sub" / "file.txt").read_text().startswith("ok")


# ------------------------------------------------------- grant lifecycle


def test_expired_grant_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    (root / "a").mkdir(parents=True)
    monkeypatch.setenv(fs_grant.GRANT_CONFIG_ENV,
                       _grant(root, expires_at=1.0))
    with pytest.raises(fs_grant.GrantDenied):
        fs_grant.check_write_access(root / "a")


def test_revoked_grant_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    (root / "a").mkdir(parents=True)
    monkeypatch.setenv(fs_grant.GRANT_CONFIG_ENV, _grant(root, state="revoked"))
    with pytest.raises(fs_grant.GrantDenied):
        fs_grant.check_write_access(root / "a")


def test_follow_symlinks_true_is_rejected_loudly(tmp_path, monkeypatch):
    """An option you can set wrongly is not an option.

    Silently ignoring it would be worse than refusing: the operator would
    believe symlink following is enabled and reason about the system on that
    basis.
    """
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setenv(fs_grant.GRANT_CONFIG_ENV,
                       _grant(root, follow_symlinks=True))
    with pytest.raises(ValueError, match="follow_symlinks"):
        fs_grant.load_grants()


def test_malformed_config_does_not_degrade_to_no_grants(tmp_path, monkeypatch):
    """"Broken config" must not be indistinguishable from "nothing granted"."""
    monkeypatch.setenv(fs_grant.GRANT_CONFIG_ENV, "{not json")
    with pytest.raises(fs_grant.GrantDenied, match="valid JSON"):
        fs_grant.load_grants()


def test_a_grant_without_roots_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv(fs_grant.GRANT_CONFIG_ENV, json.dumps(
        [{"grant_id": "empty", "issued_by": "test", "roots": [], "write": True}]))
    with pytest.raises(ValueError, match="roots"):
        fs_grant.load_grants()


def test_denial_is_structured_for_routing_not_a_blanket_abort(workspace):
    """ADR-7: a missing capability routes or repairs; it does not just fail."""
    _, outside = workspace
    try:
        fs_grant.check_write_access(outside / "secret.txt")
    except fs_grant.GrantDenied as exc:
        described = fs_grant.describe_denial(exc)
    assert described["capability"] == "fs_workspace_grant"
    assert described["outcome"] == "denied"
    assert described["remedy"]


def test_only_the_six_tier1_verbs_are_covered(workspace):
    """Execution is deliberately out of scope.

    The 2026-07-19 finding that ``pytest`` writes files is why tool classes
    are separated instead of covered by one root flag.
    """
    for verb in ("rm", "mkdir", "mv", "cp", "touch", "rmdir"):
        assert fs_grant.covers_verb(verb)
    for verb in ("pytest", "python", "bash", "make", "npm"):
        assert not fs_grant.covers_verb(verb)
