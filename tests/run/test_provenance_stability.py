"""A COMMIT FROM ANOTHER SEAT MUST NOT BE ABLE TO TURN THE SUITE RED.

WHAT HAPPENED. ``provenance._git_commit`` shells out to ``git rev-parse HEAD`` at the moment each
artifact is stamped, and ``git_commit`` is an identity key. A session-scoped fixture that emits 25
eval artifacts therefore stamps them all at whatever HEAD was current *at that instant* — so when a
second seat committed while a 10m42s suite was running, the early artifacts carried one revision
and the later ones another, and ``load_cell_evals`` refused the set. The refusal was CORRECT: on
the money path, two shards on two revisions inside one paired comparison is exactly the silent
wrong number the identity surface exists to stop. But in a test run it is a false red, and a false
red on a shared branch is how a real one gets waved through.

THE GUARD IS NOT WEAKENED — THE INPUT IS. ``conftest`` pins ``TRIKAAL_GIT_COMMIT`` once per
session; ``_git_commit`` already prefers it, so every stamp in one run agrees no matter what the
repository does underneath. Production behaviour is untouched: a run with no env var still reads
live git, which is what the money path wants.

EVERY TEST BELOW IS PAIRED WITH ITS NEGATIVE CONTROL. Asserting "the stamp did not move" proves
nothing on its own — it would also pass if the perturbation never reached the code. So each case
also runs with the pin removed and requires the stamp to MOVE, and the end-to-end case requires
the verdict to be REFUSED. If the controls stopped failing, these tests would be decoration.
"""

from __future__ import annotations

import os
import subprocess
from unittest import mock

import pytest

from trikaal.eval import verdict as V
from trikaal.utils import provenance as PROV

OTHER_HEAD = "b" * 40


def _git_returns(sha: str):
    """Patch the shell-out itself, so the perturbation enters where the real hazard enters.

    ``provenance`` imports ``subprocess`` INSIDE ``_git_commit``, so the name it resolves at call
    time is the module global — patching ``subprocess.run`` is therefore the correct target, and
    the paired negative controls below are what prove the patch actually lands.
    """
    return mock.patch.object(
        subprocess,
        "run",
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=sha + "\n"),
    )


@pytest.fixture
def _unpinned(monkeypatch):
    """Drop the session pin for one test — the negative-control arm."""
    monkeypatch.delenv("TRIKAAL_GIT_COMMIT", raising=False)


# ── the pin is actually in force ──────────────────────────────────────────────────────────────
def test_the_session_pin_is_active(_stable_git_commit_for_the_whole_run) -> None:
    """Anti-vacuity: every test here is meaningless if the fixture did not run."""
    assert os.environ.get("TRIKAAL_GIT_COMMIT") == _stable_git_commit_for_the_whole_run
    assert _stable_git_commit_for_the_whole_run


# ── a moving HEAD cannot move a stamp ─────────────────────────────────────────────────────────
def test_a_moving_head_cannot_change_a_stamp_during_a_run() -> None:
    before = PROV._git_commit()
    with _git_returns(OTHER_HEAD):
        after = PROV._git_commit()
    assert after == before, "the pin did not hold — a concurrent commit would split the run"
    assert after != OTHER_HEAD


def test_NEGATIVE_CONTROL_without_the_pin_the_stamp_does_move(_unpinned) -> None:
    """The perturbation reaches the code. Without this, the test above proves nothing."""
    with _git_returns(OTHER_HEAD):
        assert PROV._git_commit() == OTHER_HEAD


# ── the property that actually matters: a 25-artifact set still assembles ─────────────────────
def _assemble_across_a_commit(tmp_path):
    """Emit the eval set with the shell-out returning a DIFFERENT sha partway through."""
    from tests.eval.test_pinned_surface import _assemble

    real = subprocess.run
    calls = {"n": 0}

    def moving(*a, **kw):
        calls["n"] += 1
        if a and a[0] and a[0][:2] == ["git", "rev-parse"]:
            sha = ("a" if calls["n"] < 8 else "b") * 40
            return subprocess.CompletedProcess(args=a[0], returncode=0, stdout=sha + "\n")
        return real(*a, **kw)

    with mock.patch.object(subprocess, "run", side_effect=moving):
        return _assemble(tmp_path)


def test_the_verdict_still_assembles_when_a_commit_lands_mid_run(tmp_path) -> None:
    m = _assemble_across_a_commit(tmp_path)
    assert m["clauses"], "the manifest assembled but carries no clauses"


def test_NEGATIVE_CONTROL_unpinned_the_same_run_is_refused(tmp_path, _unpinned) -> None:
    """The identity guard is intact: strip the pin and the split set is rejected, as designed."""
    with pytest.raises(V.VerdictInputError):
        _assemble_across_a_commit(tmp_path)


# ── the one attribute that made safe loading impossible ───────────────────────────────────────
def test_the_torch_version_stamp_is_a_plain_string() -> None:
    """``torch.__version__`` is a ``TorchVersion`` OBJECT. Stamped into a checkpoint it becomes a
    pickle GLOBAL that ``torch.load(weights_only=True)`` refuses, so every published checkpoint
    could only be opened by executing arbitrary pickle from a 127 MB download. One missing
    ``str()`` removed the safe path for every reader of these weights."""
    prov = PROV.run_provenance(device="cpu", attention_mode="sdpa_deterministic")
    assert type(prov["torch"]) is str, type(prov["torch"])
    assert prov["torch"] == str(__import__("torch").__version__)


def test_a_checkpoint_carrying_the_stamp_opens_under_weights_only(tmp_path) -> None:
    """The property that actually matters, asserted end to end rather than by type inspection."""
    import torch

    prov = PROV.run_provenance(device="cpu", attention_mode="sdpa_deterministic")
    f = tmp_path / "ckpt.pt"
    torch.save({"meta": prov, "state_dict": {"w": torch.zeros(2)}}, f)
    got = torch.load(f, map_location="cpu", weights_only=True)
    assert got["meta"]["torch"] == prov["torch"]


def test_NEGATIVE_CONTROL_the_object_form_is_what_weights_only_refuses(tmp_path) -> None:
    """Fixture discrimination: prove ``weights_only=True`` really does reject the un-str()'d form,
    so the test above is not passing for some unrelated reason."""
    import torch

    f = tmp_path / "bad.pt"
    torch.save({"meta": {"torch": torch.__version__}}, f)  # the OBJECT, deliberately
    with pytest.raises(Exception, match="TorchVersion"):
        torch.load(f, map_location="cpu", weights_only=True)
