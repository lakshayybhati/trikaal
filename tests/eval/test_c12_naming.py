"""§7 v1.6.27 (C-12) — THE MANIFEST MUST NOT NAME ΔIR(4-5) AS THE INFORMATION EFFECT.

Clauses 1, 2 and 4 all test `pb45.delta_ir` = **ΔIR(4-5)**, which C-12 established carries
**(information) + (capacity handicap)**. All three rule strings called it `ΔIR_info`, and those
strings are **persisted into the verdict manifest**. A reader of a SURVIVES manifest would conclude
the INFORMATION effect cleared the 0.5 floor. It did not — only the sum did, and a large enough
handicap can carry the sum over the floor with no information at all.

That is the C-17 class (prose asserting something the code does not do) landing on the **headline
clause**, and the existing disclosure cannot neutralise it: `ohlcv_recon` is a reconstruction-MAE
ratio and limitation A7 states plainly that **no reconstruction-to-IR mapping exists**.

The ruling named clause 4. Clauses 1 and 2 carried the identical string; fixing only the named
instance is the class-rule failure, so all three are asserted here.
"""

from __future__ import annotations

import numpy as np

from trikaal.eval.verdict import (
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_SEEDS,
    PRIMARY_H,
    assemble_verdict,
    load_cell_evals,
    write_cell_eval_artifact,
    write_eval_index,
)

T = 256
CODEBOOK = {"coarse": {"utilization": 0.99}, "fine": {"utilization": 0.98}}
MU = {"frac_negative": 0.5, "activity_decisions": 0.5}
RECON = {"ohlcv_recon_mae": 0.10}
DECODE = {"sign_agreement_dim0": 0.95}


def _manifest(tmp_path):
    out = tmp_path / "art"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    common = rng.standard_normal(T) * 0.02
    entries = {}
    for cell in (1, 2, 3, 4, 5):
        for seed in DSR_SEEDS:
            name, sha = write_cell_eval_artifact(
                out,
                cell_id=cell,
                seed=seed,
                grid={"h": PRIMARY_H, "start_ms": 1_704_132_000_000, "n_periods": T},
                headline_series=common + rng.standard_normal(T) * 0.01,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h={
                    h: {k: float(rng.normal(0.5, 0.2)) for k in DSR_KAPPAS} for h in DSR_HORIZONS
                },
                mu_diag=MU,
                ohlcv_recon=RECON,
                decode_agreement=DECODE,
                codebook=CODEBOOK,
                meta={"fixture": True},
            )
            entries[name] = sha
    write_eval_index(out, entries)
    evals, shas = load_cell_evals(out)
    return assemble_verdict(evals, shas, tabled_mde_h15=3.518, money_verdict=False)


def test_NO_clause_rule_string_calls_the_tested_quantity_the_information_effect(tmp_path):
    man = _manifest(tmp_path)
    for name, clause in man["clauses"].items():
        rule = clause["rule"]
        assert "ΔIR_info" not in rule, (
            f"clause {name} names the tested quantity 'ΔIR_info'. It is ΔIR(4-5) = "
            "(information) + (capacity handicap), and this string ships in the manifest"
        )


def test_the_three_delta_clauses_SAY_the_quantity_carries_the_handicap(tmp_path):
    """Not merely renamed — the confound is stated where the number is read."""
    man = _manifest(tmp_path)
    for name in ("1_paired_ci", "2_mde_paired", "4_economic_floor"):
        rule = man["clauses"][name]["rule"]
        assert "ΔIR(4-5)" in rule, f"clause {name} does not name the actual tested contrast"
        assert "capacity handicap" in rule, (
            f"clause {name} states a threshold on ΔIR(4-5) without saying it carries the capacity "
            "handicap — that is the C-12 defect, and this string is what a reader sees"
        )


def test_the_placebo_disclosure_no_longer_claims_cell_4_is_tested_ON_ITS_OWN(tmp_path):
    """The false sentence was load-bearing: it was the reassurance that ΔIR(4-5) 'inflates rather
    than manufactures'. No clause tests cell 4 alone — both thresholds are on the DIFFERENCE."""
    import trikaal.eval.verdict as V

    doc = V.placebo_capacity_disclosure.__doc__ or ""
    assert "must still clear the MDE and the 0.5 economic floor on its own" not in doc
    assert "No clause tests Cell 4 on its own" in doc
