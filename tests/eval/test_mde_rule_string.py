"""Clause 2's persisted rule string must describe the recipe that produced its own number.

THE DEFECT THIS FILE EXISTS FOR. ``verdict.py``'s clause-2 rule read
``MDE_paired = (z0.95+z0.80)·SE_boot`` — the pre-v1.5 formula — while ``paired_bootstrap`` computed
t-quantiles at the Welch–Satterthwaite ν over SE_total, which is the signed-off §7 v1.5 item-B
amendment and is what the paper's §4 prose describes. The string ships inside every emitted verdict
manifest. No manifest was ever emitted, so nothing false reached a reader, but the string was in
the released artifact and it asserted a recipe the code does not run.

WHY IT SURVIVED. Clause 5 had already been rebuilt from its pins with a comment saying, in as many
words, that "a rule string that cannot disagree with the recipe is the fix". Clause 2 was left
restated. That is the class rule failing two lines from where it is written down — so the fix is
not "correct the sentence", it is to make the sentence a FUNCTION of the realized terms, and then
to assert here that its arithmetic closes.

WHAT IS ASSERTED, and why the product identity rather than the wording. A test that checks the
string contains "t-quantiles" passes the moment someone types those words, whatever the code does.
The tail of the string is pinned to the shape ``A x B = C``; this file parses A, B and C back out
and requires A·B == C == the ``mde_paired`` the bootstrap returned. A rule string can then only be
wrong by being arithmetically wrong, which is the property clause 5 already has.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from trikaal.eval.paired_bootstrap import (
    BOOT_ALPHA,
    BOOT_POWER,
    mde_rule_string,
    mde_terms,
    paired_delta_ir_bootstrap,
)

# ":: A x B = C" — the pinned, checkable tail. NOT anchored to end-of-string: clause 2 appends its
# no-ceiling-appeal sentence after it, and anchoring here silently found nothing in the manifest.
_TAIL = re.compile(r"::\s*([-\d.eE+]+)\s*x\s*([-\d.eE+]+)\s*=\s*([-\d.eE+]+)")

# The exact literal that shipped for a month. Named so that reintroducing it fails loudly.
SUPERSEDED = "(z0.95+z0.80)·SE_boot"


def _series(seed: int, t: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(2e-4, 4e-3, size=t)
    b = rng.normal(0.0, 4e-3, size=t)
    return a, b


def _pb(*, with_seeds: bool):
    a, b = _series(11)
    psd = np.array([0.31, 0.19, 0.44]) if with_seeds else None
    return paired_delta_ir_bootstrap(a, b, h=15, b=512, per_seed_deltas=psd)


@pytest.mark.parametrize("with_seeds", [True, False])
def test_the_string_arithmetic_closes_on_the_returned_mde(with_seeds: bool) -> None:
    pb = _pb(with_seeds=with_seeds)
    m = _TAIL.search(mde_rule_string(pb))
    assert m is not None, f"rule string lost its checkable tail: {mde_rule_string(pb)!r}"
    mult, se, stated = (float(g) for g in m.groups())
    assert mult * se == pytest.approx(stated, rel=1e-9)
    assert stated == pytest.approx(pb.mde_paired, rel=1e-9)
    # and the SE quoted is the one the recipe actually applied, not the other one
    assert se == pytest.approx(pb.se_total if with_seeds else pb.se_boot, rel=1e-9)


def test_the_two_branches_are_named_correctly_and_differ() -> None:
    """The t leg must be strictly harder than the z leg — the amendment's whole point."""
    t_pb, z_pb = _pb(with_seeds=True), _pb(with_seeds=False)
    t_s, z_s = mde_rule_string(t_pb), mde_rule_string(z_pb)
    assert "t_0.95" in t_s and "Welch-Satterthwaite" in t_s and "SE_total" in t_s
    assert "z_0.95" in z_s and "Welch-Satterthwaite" not in z_s
    assert t_pb.mde_paired > z_pb.mde_paired
    assert t_pb.se_total > t_pb.se_boot


def test_the_amended_recipe_never_describes_itself_with_the_superseded_formula() -> None:
    s = mde_rule_string(_pb(with_seeds=True))
    assert SUPERSEDED not in s
    assert "SE_boot" not in s.split("::")[0].replace("SE_boot^2", "")


def test_the_check_would_have_caught_the_shipped_defect() -> None:
    """FIXTURE DISCRIMINATION — run the same assertion against the string that actually shipped.

    Without this, every assertion above passes against a correct string and would ALSO have passed
    against nothing at all. The superseded literal is fed through the same parse; it must fail.
    """
    pb = _pb(with_seeds=True)
    stale = (
        "ΔIR(4-5) ≥ MDE_paired = (z0.95+z0.80)·SE_boot; no ceiling appeal — the "
        "operative threshold is the realized MDE_paired in either direction vs §2's table."
    )
    assert _TAIL.search(stale) is None, "the stale string carries no checkable arithmetic at all"
    # and had it been given one built from the WRONG leg, the product would miss mde_paired
    wrong = mde_terms(
        se_boot=pb.se_boot, se_train=0.0, n_seeds=0, b=pb.b, alpha=pb.alpha, power=pb.power
    )
    assert wrong.mde != pytest.approx(pb.mde_paired, rel=1e-9)


def test_clause_2_in_an_emitted_manifest_is_built_not_restated(tmp_path) -> None:
    """The manifest carries the terms as DATA beside the prose; both must agree with the number."""
    from tests.eval.test_verdict import MU_PLANTED, _assemble, _build_fixture

    m = _assemble(_build_fixture(tmp_path / "art", MU_PLANTED, fixture_seed=101))
    c2 = m["clauses"]["2_mde_paired"]
    assert SUPERSEDED not in c2["rule"]
    terms = c2["mde_terms"]
    assert terms["multiplier"] * terms["se"] == pytest.approx(c2["mde_paired"], rel=1e-9)
    tail = _TAIL.search(c2["rule"])
    assert tail is not None
    assert float(tail.group(3)) == pytest.approx(c2["mde_paired"], rel=1e-9)
    # the emitted recipe must be the amended one whenever per-seed contrasts exist
    assert terms["quantile_family"] in ("t", "z")
    assert BOOT_ALPHA == 0.05 and BOOT_POWER == 0.80
