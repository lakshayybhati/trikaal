"""§7 v1.6 FINDING 1 — verify the pre-run MDE floor before any abstract quotes it.

    PYTHONPATH=src .venv/bin/python scripts/m6_mde_floor_verification.py

THE RULING'S DERIVATION, TO BE CHECKED NOT ASSUMED:

    MDE_v1.2 = 2.4865 × SE_boot = 3.518  ⇒  SE_boot = 1.4148
    v1.5 floor = 3.0728 × 1.4148 = 4.3475 annualized IR at S=5, "STRICTLY GREATER once SE_train>0"

Two things must be true for that to hold, and this script tests both against the live pinned
surface rather than against any prior statement:

  A. that 3.518 is ``(z_.95+z_.80) × SE_boot`` with SE_boot the bootstrap scoring SE — i.e. that
     inverting the multiplier recovers the same object v1.5 multiplies;
  B. that ``(t_.95,4 + t_.80,4) × SE_boot`` is a LOWER BOUND on the v1.5 MDE.

Both are checked numerically through the repo's OWN code — ``eval.tdist``, ``eval.dsr.norm_ppf``,
``eval.metrics.periods_per_year`` and the exact arithmetic in ``eval.paired_bootstrap`` — so the
answer cannot drift from what the instrument will actually compute on run day.

Writes ``runs_manifest/m6_mde_floor_verification.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from trikaal.eval.dsr import norm_ppf
from trikaal.eval.metrics import periods_per_year
from trikaal.eval.tdist import student_t_ppf, welch_satterthwaite_df

OUT = Path("runs_manifest/m6_mde_floor_verification.json")
MDE_INPUTS = Path("runs_manifest/m6_mde_inputs.json")

ALPHA, POWER = 0.05, 0.80
S_SEEDS = 5  # v1.5 PINNED_SEEDS
B_BOOT = 10_000  # v1.5 §3a pinned bootstrap resamples → nu_boot = B-1
H = 15  # the pinned money horizon
RULING_FLOOR = 4.3475


def mde_v15(se_boot: float, se_train: float) -> tuple[float, float, float]:
    """The v1.5 MDE, arithmetic-identical to eval.paired_bootstrap (nu_boot = B-1, nu_train = S-1).

    Returns ``(mde, nu_effective, multiplier)``. When ``se_train == 0`` the module falls back to
    the z-based scoring-only number, which is reproduced here exactly.
    """
    se_total = math.sqrt(se_boot * se_boot + se_train * se_train)
    if se_train <= 0.0:
        mult = norm_ppf(1.0 - ALPHA) + norm_ppf(POWER)
        return mult * se_total, float("inf"), mult
    nu = welch_satterthwaite_df(se_boot, float(B_BOOT - 1), se_train, float(S_SEEDS - 1))
    mult = student_t_ppf(1.0 - ALPHA, nu) + student_t_ppf(POWER, nu)
    return mult * se_total, nu, mult


def main() -> int:
    inputs = json.loads(MDE_INPUTS.read_text())
    tabled = float(inputs["h15_pooled"]["MDE_annualized_IR"])
    t_eff = float(inputs["h15_pooled"]["T_eff"])

    # ---- A. what IS the 3.518? reproduce it from the receipt's own inputs -------------------
    z_sum = norm_ppf(1.0 - ALPHA) + norm_ppf(POWER)
    analytic_se = math.sqrt(2.0 / t_eff) * math.sqrt(periods_per_year(H))
    reproduced = z_sum * analytic_se
    inverted_se = tabled / z_sum

    # ---- B. is (t_4 quantiles) × SE a floor? sweep the training term ------------------------
    t_mult_4 = student_t_ppf(1.0 - ALPHA, 4.0) + student_t_ppf(POWER, 4.0)
    se_boot = inverted_se
    sweep = []
    for ratio in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
        mde, nu, mult = mde_v15(se_boot, ratio * se_boot)
        sweep.append(
            {
                "se_train_over_se_boot": ratio,
                "mde": mde,
                "nu_effective": nu,
                "multiplier": mult,
                "above_ruling_floor": mde >= RULING_FLOOR,
            }
        )
    monotone = all(sweep[i]["mde"] <= sweep[i + 1]["mde"] + 1e-12 for i in range(len(sweep) - 1))
    infimum = sweep[0]["mde"]

    # where does the MDE actually cross the ruling's number? (bisection on the ratio)
    lo, hi = 0.0, 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mde_v15(se_boot, mid * se_boot)[0] < RULING_FLOOR:
            lo = mid
        else:
            hi = mid
    crossing = 0.5 * (lo + hi)

    out = {
        "receipt": "m6_mde_floor_verification",
        "amendment": "prereg §7 v1.6 Finding 1 (supervisor-ordered, local $0)",
        "pins_used": {"alpha": ALPHA, "power": POWER, "S": S_SEEDS, "B": B_BOOT, "h": H},
        "A_what_3518_is": {
            "tabled_mde_h15": tabled,
            "T_eff": t_eff,
            "z_sum": z_sum,
            "formula_in_repo": (
                "z_sum * sqrt(2/T_eff) * sqrt(periods_per_year(h))  [scripts/m6_prereg.py:74]"
            ),
            "analytic_se": analytic_se,
            "reproduced_mde": reproduced,
            "reproduces_to_3dp": abs(round(reproduced, 3) - tabled) < 5e-4,
            "inverted_se_from_tabled": inverted_se,
            "CORRECTION": (
                "The tabled 3.518 is NOT (z_.95+z_.80)·SE_boot with SE_boot the bootstrap SE. It "
                "is an ANALYTIC iid-normal MDE, z_sum·sqrt(2/T_eff)·sqrt(periods_per_year(h)) — "
                "reproduced here to 3dp from the receipt's own T_eff. Inverting the multiplier "
                "therefore yields the ANALYTIC SE, not the paired-bootstrap SE_boot that the v1.5 "
                "MDE multiplies. Substituting one for the other is an assumption about how well "
                "the iid approximation predicts the realized bootstrap SE — defensible as a "
                "pre-run estimate, but it is an ESTIMATE OF AN ESTIMATE, not an identity."
            ),
        },
        "B_is_the_ruling_floor_a_floor": {
            "ruling_floor": RULING_FLOOR,
            "t_multiplier_at_nu_4": t_mult_4,
            "product_t4_times_inverted_se": t_mult_4 * inverted_se,
            "sweep": sweep,
            "mde_is_monotone_increasing_in_se_train": monotone,
            "infimum_at_se_train_zero": infimum,
            "se_train_over_se_boot_where_mde_reaches_ruling_floor": crossing,
            "CORRECTION": (
                "NO — 4.3475 is not a floor. The 3.0728 multiplier is the nu->4 limit, which "
                "obtains only when the TRAINING term dominates; but the product is taken against "
                "an SE that IGNORES the training term. Those two limits are mutually exclusive. "
                "The MDE is monotone increasing in se_train (verified across the sweep), so its "
                "infimum is at se_train = 0, where Welch-Satterthwaite gives nu -> nu_boot = "
                "9999, the multiplier returns to z_sum = 2.4865, and the MDE returns to the v1.2 "
                "number itself. The pre-run computable LOWER BOUND is therefore 3.518, not "
                f"4.3475; the MDE only reaches 4.3475 once se_train exceeds ~{crossing:.2f}x "
                "se_boot, which is not knowable before the run."
            ),
        },
        "what_the_abstract_may_state": (
            "STRUCTURE, NOT A SECOND NUMBER. Pre-data, the only defensible quantitative floor is "
            "the v1.2 analytic value (3.518 annualized IR at h=15), attained only in the limit of "
            "zero across-seed training variance; the v1.5 MDE is >= that and strictly greater "
            "once training variance is non-zero, by an amount the run itself supplies. Quoting "
            "4.35 pre-data would assert a bound the arithmetic does not support."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True))
    print("=== MDE FLOOR VERIFICATION ===")
    print(
        f"  tabled 3.518 reproduced analytically: {reproduced:.4f} "
        f"(match={out['A_what_3518_is']['reproduces_to_3dp']})"
    )
    print(f"  inverted SE                         : {inverted_se:.4f}")
    print(
        f"  t_.95,4 + t_.80,4                   : {t_mult_4:.4f}  "
        f"-> x SE = {t_mult_4 * inverted_se:.4f}"
    )
    print(f"  MDE monotone in se_train            : {monotone}")
    print(f"  infimum (se_train=0)                : {infimum:.4f}  <-- the real floor")
    print(f"  reaches {RULING_FLOOR} only at se_train/se_boot >= {crossing:.3f}")
    print(f"[receipt] -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
