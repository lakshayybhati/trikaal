"""§7 v1.6.25 (RE-AUDIT R2 + R3) — THE CLAUSE-5 DE-ANNUALIZATION BASIS, PINNED AND RE-DERIVED.

Two findings, one quantity. The C-3 amendment (v1.6.22, pre-data) moved the clause-5 trial set from
per-**own-h** Sharpes to per-**PRIMARY_H** Sharpes, because SR₀ is compared against the Sharpe of
the headline series at PRIMARY_H. Mixing h ∈ {5, 15, 60} spans √12 = 3.4641× and inflates
``var_sr``; the amendment's own witness moved DSR 0.9868 PASS → 0.8577 FAIL.

* **R2** — the amendment's signed package specified a pin key and a revert-mutation KAT. Neither
  shipped. Reverting one identifier (`PRIMARY_H` → `h`) passed all 601 tests and silently restored
  the EASIER basis. The pin is `PINNED_DSR["de_annualization_horizon"]`, and
  ``dsr_unit_convention_failures`` re-derives every trial value from the artifacts ON THE MONEY
  PATH — so the revert now fails the run, not merely a test.
* **R3** — the ``dual_specification`` v1.2 leg read the AMENDED trial set while calling itself a
  faithful reconstruction of the ORIGINAL spec: {all-arms basis} × {PRIMARY_H units}, a
  combination no specification ever had. The bias has a direction — the faithful (own-h) basis has
  the LARGER variance, hence larger SR₀ and LOWER DSR — so the hybrid agreed with the v1.5 primary
  more readily than v1.2 would have, under-reporting disagreement exactly when the primary says
  SURVIVES. That is the one direction a dual-specification report exists to guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval.conformance import PINNED_DSR, ConformanceError
from trikaal.eval.metrics import periods_per_year
from trikaal.eval.verdict import (
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_SEEDS,
    PRIMARY_H,
    assemble_verdict,
    dsr_unit_convention_failures,
    enumerate_dsr_trials,
    load_cell_evals,
    own_h_trial_values,
    write_cell_eval_artifact,
    write_eval_index,
)

T = 300
CODEBOOK = {"coarse": {"utilization": 0.99}, "fine": {"utilization": 0.98}}
MU = {"frac_negative": 0.5, "activity_decisions": 0.5}
RECON = {"ohlcv_recon_mae": 0.10, "ohlcv_recon_mae_by_dim": [0.1] * 7, "n_ohlcv_dims": 7}
DECODE = {"sign_agreement_dim0": 0.95, "variance_ratio_dim0": 1.0}


def _fixture(out):
    """25 artifacts whose VAL IRs differ ACROSS HORIZONS — otherwise the two conventions could
    not be told apart and every assertion here would be vacuous."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    common = rng.standard_normal(T) * 0.02
    entries = {}
    for cell in (1, 2, 3, 4, 5):
        for seed in DSR_SEEDS:
            # deliberately horizon-dependent levels: h=5 near 0.2, h=15 near 0.6, h=60 near 1.4
            val = {
                h: {k: float(rng.normal(0.2 + 0.02 * h, 0.15)) for k in DSR_KAPPAS}
                for h in DSR_HORIZONS
            }
            name, sha = write_cell_eval_artifact(
                out,
                cell_id=cell,
                seed=seed,
                grid={"h": PRIMARY_H, "start_ms": 1_704_132_000_000, "n_periods": T},
                headline_series=common + rng.standard_normal(T) * 0.01,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h=val,
                mu_diag=MU,
                ohlcv_recon=RECON,
                decode_agreement=DECODE,
                codebook=CODEBOOK,
                meta={"fixture": True},
            )
            entries[name] = sha
    write_eval_index(out, entries)
    return out


# ------------------------------------------------------------------ R2: the pin
def test_the_unit_convention_is_PINNED_and_agrees_with_PRIMARY_H():
    assert PINNED_DSR["de_annualization_horizon"] == PRIMARY_H == 15


def test_every_trial_is_de_annualized_at_PRIMARY_H_regardless_of_its_own_horizon(tmp_path):
    """The amendment in one assertion: two trials with the SAME annualized VAL IR but different
    horizons must map to the SAME de-annualized value."""
    evals, _ = load_cell_evals(_fixture(tmp_path / "a"))
    trials = enumerate_dsr_trials(evals)
    denom = float(np.sqrt(periods_per_year(PRIMARY_H)))
    for (cid, seed, h, k), got in trials.items():
        ir = float(evals[(cid, seed)]["val_ir_by_kappa_by_h"][str(h)][f"{k:g}"])
        assert got == ir / denom
        if h != PRIMARY_H:  # and it is NOT the superseded own-h value
            assert got != ir / float(np.sqrt(periods_per_year(h)))


def test_the_money_path_REFUSES_a_trial_set_on_the_own_h_basis(tmp_path):
    """The production check, not a test-only one. A hand-built own-h trial set must be rejected
    by the same function ``assemble_verdict`` calls."""
    evals, _ = load_cell_evals(_fixture(tmp_path / "b"))
    good = enumerate_dsr_trials(evals)
    assert dsr_unit_convention_failures(evals, good) == []

    keys = sorted(good)
    reverted = dict(zip(keys, own_h_trial_values(evals, keys), strict=True))
    fails = dsr_unit_convention_failures(evals, reverted)
    assert fails and any("OWN horizon" in f for f in fails)


def test_a_reverted_trial_set_cannot_reach_a_verdict(tmp_path, monkeypatch):
    """End-to-end: with ``enumerate_dsr_trials`` reverted to the pre-amendment convention,
    ``assemble_verdict`` raises instead of quietly scoring clause 5 on the easier basis."""
    import trikaal.eval.verdict as V

    evals, shas = load_cell_evals(_fixture(tmp_path / "c"))
    assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)  # discrimination

    def reverted(ev):
        keys = sorted(enumerate_dsr_trials(ev))
        return dict(zip(keys, own_h_trial_values(ev, keys), strict=True))

    monkeypatch.setattr(V, "enumerate_dsr_trials", reverted)
    with pytest.raises(ConformanceError, match="de-annualization"):
        assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)


# ------------------------------------------------------------------ R3: the v1.2 leg
def test_the_v1_2_leg_uses_the_OWN_H_units_v1_2_actually_had(tmp_path):
    evals, shas = load_cell_evals(_fixture(tmp_path / "d"))
    man = assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)
    trials = enumerate_dsr_trials(evals)
    keys = sorted(trials)

    faithful = float(np.var(own_h_trial_values(evals, keys)))
    hybrid = float(np.var(np.array([trials[k] for k in keys], dtype=np.float64)))
    assert faithful != hybrid, "the fixture cannot separate the two conventions"

    got = man["dual_specification"]["v1_2_original"]["var_sr"]
    assert got == faithful
    assert got != hybrid


def test_the_hybrid_leg_was_biased_TOWARD_agreement(tmp_path):
    """Not an opinion — the direction, measured. Mixing horizons inflates the dispersion, so the
    faithful basis gives a LARGER var_sr, a LARGER SR0 and a LOWER DSR than the hybrid did. A
    lower DSR is more likely to fail clause 5, i.e. more likely to DISAGREE with a SURVIVES
    primary — which is the disagreement the hybrid was suppressing."""
    from trikaal.eval.dsr import expected_max_sharpe

    evals, _ = load_cell_evals(_fixture(tmp_path / "e"))
    trials = enumerate_dsr_trials(evals)
    keys = sorted(trials)
    n = len(keys)
    faithful = float(np.var(own_h_trial_values(evals, keys)))
    hybrid = float(np.var(np.array([trials[k] for k in keys], dtype=np.float64)))
    assert faithful > hybrid
    assert expected_max_sharpe(faithful, n) > expected_max_sharpe(hybrid, n)


def test_the_three_dual_specification_legs_are_the_three_real_conventions(tmp_path):
    """{all-arms, cell-5} × {own-h, PRIMARY_H} has four corners; three of them are specifications
    this project actually held, and each is reported. The fourth — all-arms × PRIMARY_H — is the
    hybrid R3 removed, and it must not reappear as a leg."""
    evals, shas = load_cell_evals(_fixture(tmp_path / "f"))
    ds = assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)[
        "dual_specification"
    ]
    assert "OWN horizon" in ds["v1_2_original"]["var_sr_basis"]
    assert "OWN horizon" in ds["v1_5_mixed_unit_basis_superseded"]["var_sr_basis"]
    assert ds["v1_5_amended"]["var_sr_basis"] == "cell5_placebo"
    assert isinstance(ds["agree"], bool)
