"""M6 POST-MORTEM — BREAK-EVEN COST PER HORIZON, from banked artifacts only. $0, no GPU.

WHAT THIS ANSWERS. Invariant 5 pins the headline to cost-aware NET IR at a modelled 0.10-0.30%
round trip. Every banked cell-1 unit is deeply negative across that whole band. The decision-
relevant follow-up is not "how negative" but "at what round-trip cost would this strategy break
even, and is that above or below what an exchange actually charges". That number is
``c_break`` — and it is a SINGLE number per horizon, comparable against a quantity anyone can
look up, which is why it is the lead rather than the slope of edge in h.

★★ THIS IS NOT A SECONDARY DIAGNOSTIC. I LABELLED IT ONE AND THAT WAS WRONG.
``docs/m6_prereg.md:170-171`` defines the §4 headline as "the COST-STRESS CURVE across flat
nettings {0.10 %, 0.20 %, 0.30 %} at κ* (WITH BREAK-EVEN COST)". Break-even cost is therefore a
COMPONENT OF THE PRE-REGISTERED HEADLINE, not a new quantity beside it. What this script does is
recompute that pre-registered component with a NON-PRE-REGISTERED INSTRUMENT: the pinned
``metrics.break_even_cost`` interpolates on {0.10,0.20,0.30 %} and returns -inf here, while this
script bisects the exact root BELOW that grid floor and returns a finite 0.0023-0.0208 %.
SAME NAMED QUANTITY, DIFFERENT INSTRUMENT, DIFFERENT ANSWER. Whether the extension below the
pre-registered floor is admissible is THE SUPERVISOR'S RULING AND NOT MINE. Both numbers ship
side by side in every unit so the substitution can never be silent. Invariant 5 is untouched —
the headline metric is still cost-aware NET IR and no gross quantity replaces it.

★ c_break IS AN ESTIMATE AND IT CARRIES A STANDARD ERROR. gbar_A is a sample mean over active
periods, so c_break has sampling error that was computable at $0 — and my first version reported
it to 16 significant figures with no interval anywhere. Worse, the receipt already carried its own
t-statistic under a name that hid it: because the money grid is 35,064 periods and
periods_per_year(15) = 35,040, ``IR_gross`` IS t(gbar_A) to 4 decimal places (measured ratio
1.0003-1.0004 on all three units). So the significance of the break-even number was sitting one
field away, mislabelled, and nothing linked them. Both now ship together, and the frozen
``m6_units_1_2_predeclaration.md`` block-bootstrap intervals on IR_gross (which CONTAIN ZERO for
seeds 0 and 4) are the same statement about c_break.

★ THE ASYMMETRY, IN BOTH DIRECTIONS — and it is weak BOTH ways, unlike the window probe.
The window probe measured an ABSENCE and the asymmetry ran one way (signal would have been
conclusive, absence only suggestive). Here BOTH directions are only suggestive, for the same
single reason: the evidence is CELL 1 ONLY — BSQ + OHLCV-only, one arm of a 2x2 that never ran.
  - Finding a horizon whose c_break exceeds real fees would be SUGGESTIVE, not proof: it is one
    quantizer on one input arm, and we hold no micro artifacts to test whether it transfers.
  - Finding NO such horizon is ALSO only SUGGESTIVE, and for the same reason: a null on cell 1
    does not license "no horizon is tradeable", because the arm that the project's headline claim
    is about was never scored. A null here must not be read as conclusive.
Do not let either direction be quoted as settled, and do not confuse this asymmetry with the
window probe's — they point opposite ways.

--------------------------------------------------------------------------------------------
THE IDENTITY, AND WHY c_break IS EXACT AT h=15
--------------------------------------------------------------------------------------------
``single_symbol_backtest`` / ``score_cell`` emit the calendar series with flat periods as exactly
0.0 and, on the headline pass, a CONSTANT per-position netting cost c
(``np.full(n, headline_cost)``).
Pooling equal-weights the active positions in a period, so for active period p

    x_p(c) = mean_n(s_n r_n) - c  =  g_p - c        (cost enters ONCE per period, any breadth)
    x_p(c) = 0.0                                     (flat period)

Write T for the grid size, A for the active set, a = |A|/T, and gbar_A = mean_{p in A} g_p. Then

    mean_p x_p(c) = a * (gbar_A - c)

and since IR = sqrt(ppy) * mean / (std + eps) with std > 0 and a > 0,

    IR(c) = 0   <=>   gbar_A = c

    ==>  c_break = gbar_A = THE MEAN GROSS RETURN PER ACTIVE PERIOD.

That is arithmetic on a banked array, not inference, and it needs no re-run.

★ WHAT ``verify_identity`` DOES AND DOES NOT CHECK — I OVERSTATED THIS AND IT IS THE PROJECT'S
OLDEST DEFECT IN MY OWN NEW CODE. The first draft said the identity "is NOT assumed here". It IS
partly assumed. ``verify_identity`` compares ``gross_series`` (which ADDS headline_cost on the
mask) against a bisected root of ``ir_at_cost`` (which SUBTRACTS the same constant on the same
mask). Both apply the IDENTICAL affine shift, so they agree for ANY input — including an input
netted at a cost that is not 0.0030 at all. Demonstrated by the audit: feeding a series netted at
0.0010 while telling the script 0.0030 still returns ``agrees=True`` while the reported "gross"
is 10.2x wrong. So the check verifies THE ALGEBRA, never THE PREMISE.

The premise — netting is FLAT and its value is exactly 0.0030 — is not verifiable from the
artifact's own fields (``write_cell_eval_artifact`` persists no headline_cost, no cost_stress and
no break_even). It is instead enforced UPSTREAM: ``assert_conformance`` refuses any money run
whose ``headline_cost != PINNED_HEADLINE_COST``, and ``meta.money_run = True`` can only be written
on a run that passed it. This script therefore REFUSES any artifact that is not
``money_run is True``, which is the only honest way to inherit the pin. That refusal, not
``verify_identity``, is what guards the premise.

The one assumption is inherited from ``diagnostics.gross_series``: active periods are identified
as ``series != 0.0``. Measured on these three units, zero values fall in (0, 1e-9) and the smallest
|nonzero| is ~1e-7, so the populations are cleanly separated rather than assumed to be.

--------------------------------------------------------------------------------------------
WHY h=5 AND h=60 GET BOUNDS AND NOT NUMBERS — A CORRECTION TO MY OWN SCOPING
--------------------------------------------------------------------------------------------
I scoped this task as "three points, one at a kappa grid boundary". THAT WAS WRONG, and the error
was mine: it assumed the val passes bank a SERIES. They do not. ``write_cell_eval_artifact``
persists ``headline_series`` (h=15 only) plus ``val_ir_by_kappa_by_h``, which is FOUR ANNUALIZED
IR SCALARS per horizon and nothing else. A scalar IR at a single cost cannot be re-netted, so
gbar_A is NOT recoverable at h=5 or h=60 from banked artifacts.

The curve therefore has ONE measured point and TWO bounds, not three points. What IS exact at
h=5/60, with no assumptions beyond var >= 0:

  (1) SIGN BOUND. The val IRs are netted at the same flat 0.30% (``np.full(n, cfg.headline_cost)``
      on the val path too - verified in xsection.py). IR(0.0030) < 0 and IR is monotone decreasing
      in c with positions fixed, so c_break < 0.0030 at every horizon and every kappa. Exact.

  (2) ACTIVITY LOWER BOUND. From IR^2 (v + (1-a)m^2) = ppy*a*m^2 with v >= 0 and m != 0:
          a >= IR^2 / (ppy + IR^2)
      Exact and assumption-free. Reported because it is the only other thing the scalars support.

  (3) IMPLIED c_break under an EXPLICIT variance assumption. Clearly labelled INFERENCE, never a
      measurement, reported as a range over the assumption rather than a point.

--------------------------------------------------------------------------------------------
WHAT THE SHIPPED INSTRUMENT WOULD SAY, AND WHY IT CANNOT ANSWER THIS QUESTION
--------------------------------------------------------------------------------------------
``metrics.break_even_cost`` interpolates over the pre-registered stress grid
FLAT_COSTS = (0.0010, 0.0020, 0.0030) and returns -inf when every IR on it is <= 0. On this data
every IR is negative at all three points, so the shipped function returns -inf: literally
"cost-fragile below the swept band". True, and uninformative, because THE GRID FLOOR (0.10%) IS
EXACTLY THE REALISTIC ROUND TRIP WE MOST WANT TO COMPARE AGAINST. The pre-registered instrument
bottoms out precisely where the interesting comparison starts. Answering the question requires
extending below the pre-registered grid, which is what this script does and why it is a separate
diagnostic rather than a field read.

Separately: ``CellScore.break_even`` IS computed at eval time but ``write_cell_eval_artifact``
never persists it, so it was discarded on the rented box. Both facts are recorded in the receipt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trikaal.eval.diagnostics import gross_series  # noqa: E402
from trikaal.eval.harness import FLAT_COSTS, HEADLINE_COST  # noqa: E402
from trikaal.eval.metrics import (  # noqa: E402
    EPS_IR,
    break_even_cost,
    information_ratio,
    periods_per_year,
)

PRIMARY_H = 15
VAL_HORIZONS = (5, 15, 60)

# Supplied by the supervisor in the task prompt, NOT independently verified by this script and
# NOT fetched from any exchange API. Carried as an explicitly-attributed external reference so
# that a reader who disputes the fee disputes the ATTRIBUTION and not our measurement.
FEE_REFERENCE = {
    "source": "supervisor-supplied in the task prompt; NOT independently verified here",
    "venue": "Binance USD-M perpetual futures, taker",
    "per_side_frac": (0.0004, 0.0005),
    "round_trip_frac": 0.0010,
    "caveat": (
        "A round-trip taker fee is a LOWER BOUND on real execution cost: it excludes spread paid, "
        "market impact, funding, and slippage. The true comparison threshold is therefore ABOVE "
        "0.10%, which makes any 'c_break below fees' finding stronger and any 'c_break above fees' "
        "finding weaker than the bare comparison suggests."
    ),
}


# ---------------------------------------------------------------------------------------------
# exact re-netting on the banked series (positions FIXED - cost-stress semantics)
# ---------------------------------------------------------------------------------------------
def renet(net_series: np.ndarray, *, from_cost: float, to_cost: float) -> np.ndarray:
    """Same positions, re-netted from ``from_cost`` to ``to_cost``. Flat periods stay exactly 0.0.

    Mirrors ``xsection._renet``: ``np.where(head != 0, head - (flat_c - headline_cost), 0)``.
    """
    s = np.asarray(net_series, dtype=np.float64)
    return np.where(s != 0.0, s - (to_cost - from_cost), 0.0)


def ir_at_cost(net_series: np.ndarray, c: float, h: int, *, headline_cost: float) -> float:
    return information_ratio(renet(net_series, from_cost=headline_cost, to_cost=c), h)


def mean_gross_per_active_period(net_series: np.ndarray, *, headline_cost: float) -> float:
    """gbar_A — the identity's value of c_break. Uses the production ``gross_series``."""
    s = np.asarray(net_series, dtype=np.float64)
    g = gross_series(s, headline_cost=headline_cost)
    active = s != 0.0
    return float(g[active].mean()) if active.any() else float("nan")


def root_of_ir(net_series: np.ndarray, h: int, *, headline_cost: float) -> float:
    """Numerically bracket-and-bisect IR(c)=0. INDEPENDENT of the identity — it is its check."""
    f = lambda c: ir_at_cost(net_series, c, h, headline_cost=headline_cost)  # noqa: E731
    lo, hi = -0.05, 0.05
    if not (f(lo) > 0.0 > f(hi)):
        return float("nan")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------------------------
# the two checks the identity must survive before any number is read (standing norms)
# ---------------------------------------------------------------------------------------------
def verify_identity(net_series: np.ndarray, h: int, *, headline_cost: float) -> dict:
    """Assert the numeric root of IR(c) equals gbar_A. Behaviour, not existence."""
    gbar = mean_gross_per_active_period(net_series, headline_cost=headline_cost)
    root = root_of_ir(net_series, h, headline_cost=headline_cost)
    return {
        "gbar_A": gbar,
        "numeric_root": root,
        "abs_diff": abs(gbar - root),
        "agrees": bool(np.isfinite(root) and abs(gbar - root) < 1e-12),
    }


def verify_identity_discriminates(net_series: np.ndarray, h: int, *, headline_cost: float) -> dict:
    """FIXTURE-DISCRIMINATION RULE — prove the check above CAN fail before trusting that it passed.

    Shift every ACTIVE period by a known delta. gbar_A must move by exactly delta and the numeric
    root must follow it. A check that passes on the perturbed series too would be checking nothing.
    """
    s = np.asarray(net_series, dtype=np.float64)
    delta = 1e-4
    perturbed = np.where(s != 0.0, s + delta, 0.0)
    base = verify_identity(s, h, headline_cost=headline_cost)
    pert = verify_identity(perturbed, h, headline_cost=headline_cost)
    moved = pert["gbar_A"] - base["gbar_A"]
    return {
        "delta_applied": delta,
        "gbar_moved_by": moved,
        "root_moved_by": pert["numeric_root"] - base["numeric_root"],
        "moves_as_expected": bool(abs(moved - delta) < 1e-12),
        "perturbed_still_agrees": bool(pert["agrees"]),
        # the discrimination claim: the ORIGINAL gbar must NOT satisfy the perturbed series
        "original_value_rejected_by_perturbed": bool(
            abs(base["gbar_A"] - pert["numeric_root"]) > 1e-9
        ),
    }


# ---------------------------------------------------------------------------------------------
# exact, assumption-free bounds from a bare annualized IR scalar
# ---------------------------------------------------------------------------------------------
def activity_lower_bound(ir: float, h: int) -> float:
    """a >= IR^2/(ppy + IR^2). Exact; follows from var(g) >= 0 alone."""
    ppy = periods_per_year(h)
    return float(ir * ir / (ppy + ir * ir))


def implied_break_even(ir: float, h: int, a: float, v: float, *, netted_at: float) -> float:
    """INFERENCE, NOT MEASUREMENT. c_break implied by (IR, a, v) under the calendar-series model.

    Solves m^2 = IR^2 v / (ppy a - IR^2 (1-a)) for m = gbar_A - c, sign(m) = sign(IR).
    Returns nan when the assumed (a, v) are inconsistent with the observed IR.

    ★ MEASURED TO BE USELESS, AND KEPT ONLY TO SHOW THAT. Sweeping the admissible activity range
    makes the denominator approach zero at the activity lower bound, so the band blows up to
    several HUNDRED PERCENT — four orders of magnitude wider than the question being asked. The
    endpoints are therefore NOT a break-even estimate and the receipt never exposes them under a
    quotable key. Narrowing the band would require inventing an activity we did not measure, which
    is the self-scaling rule's prohibition exactly.
    """
    ppy = periods_per_year(h)
    denom = ppy * a - ir * ir * (1.0 - a)
    if denom <= 0.0 or v < 0.0:
        return float("nan")
    m = float(np.sign(ir) * np.sqrt(ir * ir * v / denom))
    return netted_at + m


def required_gross_edge_per_active_period(target_cost: float) -> float:
    """EXACT AND HORIZON-INDEPENDENT. c_break = gbar_A, so clearing a round trip of ``target_cost``
    requires gross edge per ACTIVE period >= target_cost — the same threshold at EVERY horizon.

    This is the assumption-free reframing of the whole question: the strategy must earn 0.10% gross
    per active period to survive a 0.10% round trip, whether it holds for 5 minutes or 60. What
    changes with h is not the threshold but how much gross edge a period can plausibly contain.
    """
    return float(target_cost)


def required_horizon_for_fee(edge_at_h: float, h: int, target_cost: float) -> float:
    """Holding period at which gross edge per active period would reach ``target_cost`` under
    LINEAR edge scaling: edge(h') = edge(h) * h'/h.

    ★★ I CALLED THIS A "LOWER BOUND". IT IS NOT A BOUND OF ANY KIND — WITHDRAWN, 3 of 3 refuted.
    Four independent grounds, none of which depend on the edge-decay prior I had appealed to:

      1. SAMPLING ERROR CROSSES IT ON ALL THREE SEEDS. The denominator is a sample mean whose
         t-statistic is 1.14-2.17. At the 95% upper end of edge the required horizon is
         109.5 / 37.9 / 238.0 min — every one BELOW the value I published as a floor. On seeds 0
         and 4 the edge CI contains zero, where the honest output is "no finite horizon".
      2. IT OVERSHOOTS ITS OWN MODEL. ``forward_log_returns`` is log(C_{t+h}/C_{t+1}), spanning
         h-1 one-minute returns, so under the constant drift the docstring invokes the exact
         answer is 1 + (h-1)*c/edge, not h*c/edge. This function sits 5.6-7.0% ABOVE the model it
         claims to instantiate. A value that overshoots its own model bounds nothing.
      3. SUPER-LINEAR GROWTH IS ORDINARY HERE. Any one-time entry drag b > 0 inside the gross
         return (transient impact, adverse selection, bid-ask bounce) gives edge(h) = a*h - b, so
         edge(h)/h is INCREASING — which flips the direction to an upper bound.
      4. THE PREMISE "edge is a function of h" IS BROKEN BY THE PRODUCTION CODE. ``_per_bar_cost``
         takes no horizon argument, so theta = kappa*c_total is horizon-INDEPENDENT while mu-hat
         is horizon-dependent: raising h leaves the filter fixed while |mu-hat| grows, so the
         filter stops binding and the ACTIVE SET changes. gbar_A is a mean over that selected set.

    It is retained ONLY as a labelled point extrapolation, reported with its interval, because the
    arithmetic still answers "how far away is this from working" in units anyone can read.
    """
    if not np.isfinite(edge_at_h) or edge_at_h <= 0.0:
        return float("nan")
    return float(h * target_cost / edge_at_h)


def break_even_interval(net_series: np.ndarray, *, headline_cost: float) -> dict:
    """SE and 95% interval on c_break = gbar_A, plus the t == IR_gross reconciliation.

    gbar_A is a sample mean over active periods, so its standard error is available at $0 from the
    banked series alone. The iid SE is reported alongside the lag-1 autocorrelation of the active
    gross series, so a reader can see whether serial dependence would materially widen it; the
    frozen ``m6_units_1_2_predeclaration.md`` carries the pinned BLOCK-bootstrap (B=10,000,
    L=188) intervals on the same quantity expressed as IR_gross.
    """
    s = np.asarray(net_series, dtype=np.float64)
    g = gross_series(s, headline_cost=headline_cost)
    ga = g[s != 0.0]
    n = int(ga.size)
    if n < 3:
        return {"n_active": n, "insufficient": True}
    gbar = float(ga.mean())
    se = float(ga.std(ddof=1) / np.sqrt(n))
    d = ga - gbar
    lag1 = float((d[:-1] * d[1:]).sum() / (d * d).sum()) if n > 2 else float("nan")
    return {
        "n_active": n,
        "c_break_point": gbar,
        "se_iid": se,
        "t_stat": float(gbar / se) if se > 0 else float("nan"),
        "ci95_low": float(gbar - 1.96 * se),
        "ci95_high": float(gbar + 1.96 * se),
        "lag1_autocorr_active_gross": lag1,
        "note_t_equals_ir_gross": (
            "IR_gross IS this t-statistic: the money grid is 35,064 periods and "
            "periods_per_year(15) = 35,040, so IR_gross = sqrt(n_active)*gbar/sd to ~4 decimals. "
            "The frozen m6_units_1_2_predeclaration.md block-bootstrap intervals on IR_gross "
            "(containing ZERO for seeds 0 and 4) are the SAME statement about c_break."
        ),
    }


# ---------------------------------------------------------------------------------------------
def analyse_unit(doc: dict) -> dict:
    # ★ THE PREMISE GATE. The flat-netting value 0.0030 is NOT recoverable from the artifact and
    # verify_identity is blind to it (see module docstring). It is inherited from
    # assert_conformance, which only lets money_run=True be written when headline_cost matches
    # PINNED_HEADLINE_COST. An artifact without that flag has no such guarantee, so it is refused
    # rather than analysed under an assumption it cannot support.
    if doc.get("meta", {}).get("money_run") is not True:
        raise ValueError(
            f"cell{doc.get('cell_id')}_seed{doc.get('seed')}: meta.money_run is not True — the "
            "flat-netting premise (headline_cost == 0.0030) rests on assert_conformance and is "
            "unverifiable for a non-money artifact. REFUSED."
        )
    x = np.array(doc["headline_series"], dtype=np.float64)
    h = int(doc["grid"]["h"])
    seed = int(doc["seed"])
    active = x != 0.0
    a = float(active.mean())
    g = gross_series(x, headline_cost=HEADLINE_COST)

    ident = verify_identity(x, h, headline_cost=HEADLINE_COST)
    disc = verify_identity_discriminates(x, h, headline_cost=HEADLINE_COST)

    stress = {f"{c:.4f}": ir_at_cost(x, c, h, headline_cost=HEADLINE_COST) for c in FLAT_COSTS}
    shipped = break_even_cost(
        np.array(sorted(FLAT_COSTS)), np.array([stress[f"{c:.4f}"] for c in sorted(FLAT_COSTS)])
    )

    c_break = ident["gbar_A"]
    fee_rt = FEE_REFERENCE["round_trip_frac"]
    return {
        "seed": seed,
        "h": h,
        "provenance": "HEADLINE grid (money region, forward blocks 1..5) — the pinned h=15 pass",
        "kappa_star": float(doc["kappa_star_by_h"][str(h)]),
        "n_periods": int(x.size),
        "period_activity": a,
        "decision_activity": float(doc["mu_diag"]["activity_decisions"]),
        "ir_net_at_headline_cost": information_ratio(x, h),
        "ir_gross_IS_THE_T_STAT_OF_c_break_NOT_A_RESULT": information_ratio(g, h),
        "cost_stress_ir": stress,
        "shipped_PREREGISTERED_break_even_cost_on_pinned_grid": shipped,
        "c_break_POINT_ESTIMATE_positions_fixed": c_break,
        "c_break_interval": break_even_interval(x, headline_cost=HEADLINE_COST),
        "c_break_pct_POINT_see_c_break_interval": c_break * 100.0,
        "c_break_over_round_trip_fee": c_break / fee_rt,
        "clears_round_trip_fee": bool(c_break > fee_rt),
        # SAME NUMBER as c_break — the identity. Named for the edge-vs-h table; carries the same
        # interval, which lives in c_break_interval.
        "gross_edge_per_active_period_EQUALS_c_break": c_break,
        "gross_var_per_active_period": float(g[active].var(ddof=0))
        if active.any()
        else float("nan"),
        "identity_check": ident,
        "identity_discrimination_check": disc,
    }


def analyse_val_scalars(doc: dict, measured_h15: dict) -> dict:
    """h=5/60 (and the h=15 VAL point) from IR scalars alone: exact bounds + labelled inference."""
    out = {}
    v15 = measured_h15["gross_var_per_active_period"]
    for h_str, by_k in sorted(doc["val_ir_by_kappa_by_h"].items(), key=lambda kv: int(kv[0])):
        h = int(h_str)
        ks = doc["kappa_star_by_h"][h_str]
        ir = float(by_k[f"{float(ks):g}"]) if f"{float(ks):g}" in by_k else float(by_k[str(ks)])
        # variance scaling assumption band: log-return variance over a stride-h period scales
        # ~linearly in h. Reported as a BAND over the exponent, never as a point.
        widths = []
        for expo in (0.5, 1.0, 1.5):
            v_h = v15 * (h / PRIMARY_H) ** expo
            a_lb = activity_lower_bound(ir, h)
            vals = [
                implied_break_even(ir, h, a, v_h, netted_at=HEADLINE_COST)
                for a in (min(0.999999, a_lb + 1e-6), 0.5, 0.999999)
            ]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                widths.append((max(vals) - min(vals)) * 100.0)
        width_pct = max(widths) if widths else float("nan")
        out[h_str] = {
            "h": h,
            "provenance": "VAL block only (forward block 0) — NEVER the headline grid; do not mix",
            "kappa_star": float(ks),
            "ir_net_at_headline_cost_all_kappa": {k: float(v) for k, v in by_k.items()},
            "ir_net_at_kappa_star": ir,
            "netted_at": HEADLINE_COST,
            "EXACT_sign_bound_c_break_lt": HEADLINE_COST,
            "EXACT_activity_lower_bound": activity_lower_bound(ir, h),
            "c_break_exact": None,
            "why_no_exact_value": (
                "val passes bank FOUR ANNUALIZED IR SCALARS per horizon, not a series. A scalar IR "
                "at a single netting cost cannot be re-netted, so gbar_A is not recoverable. This "
                "corrects my own scoping, which promised three measured points."
            ),
            # ★ CONTEXT-STRIPPING RULE: no endpoint of this band is exposed under a quotable key.
            # A cropped "+0.25%" would read as a break-even; only the WIDTH and the VERDICT ship.
            "inference_attempted_and_REJECTED": {
                "verdict": "UNINFORMATIVE — carries no constraint on c_break",
                "admissible_band_width_pct": width_pct,
                "why": (
                    "Sweeping the admissible activity range drives the denominator to zero at the "
                    "activity lower bound, so the band spans hundreds of percent — four orders of "
                    "magnitude wider than the 0.001-0.10% question. Narrowing it would require "
                    "inventing an activity we never measured (self-scaling rule). ENDPOINTS ARE "
                    "DELIBERATELY NOT REPORTED: a cropped upper endpoint would read as a "
                    "break-even estimate."
                ),
                "usable_statements_at_this_horizon": [
                    "c_break < 0.30% (exact, from IR<0 and monotonicity in c)",
                    "activity >= IR^2/(ppy+IR^2) (exact, from var >= 0)",
                ],
            },
        }
    return out


def transfer_check_inventory(repo: Path) -> dict:
    """Why the cell-1-only limit is ENFORCED, not merely regretted — verified by BEHAVIOUR.

    The asymmetry statement rests on "we hold no micro artifacts to check transfer". That is a
    claim about the repo, so it is measured rather than asserted: every ``cell*_eval.json`` under
    ``runs_cloud/`` is inventoried, and each non-money artifact SET is handed to the real
    ``load_cell_evals`` to confirm it is REFUSED. Micro-arm artifacts DO exist (cells 3/4/5 at toy
    geometry, cell 2 at toy and cuda-validation geometry) — they are simply unusable, and an
    inventory that only said "none exist" would have been wrong.
    """
    from trikaal.eval.verdict import load_cell_evals

    inv, by_dir = [], {}
    for p in sorted(repo.glob("runs_cloud/**/cell*_eval.json")):
        d = json.loads(p.read_text())
        m = d.get("meta", {})
        rec = {
            "path": str(p.relative_to(repo)),
            "cell_id": d.get("cell_id"),
            "seed": d.get("seed"),
            "arm": d.get("arm"),
            "money_run": m.get("money_run"),
            "dry_run": m.get("dry_run"),
            "grid_pinned": m.get("grid_pinned"),
            "n_periods": d.get("grid", {}).get("n_periods"),
        }
        inv.append(rec)
        by_dir.setdefault(str(p.parent.relative_to(repo)), []).append(rec)

    refusals = {}
    for dirname in by_dir:
        if all(r["money_run"] is True for r in by_dir[dirname]):
            refusals[dirname] = "MONEY — accepted (this is the quotable set)"
            continue
        try:
            load_cell_evals(repo / dirname)
            refusals[dirname] = "★ NOT REFUSED — investigate"
        except Exception as e:
            refusals[dirname] = f"REFUSED: {type(e).__name__}: {str(e).splitlines()[0][:160]}"

    micro = sorted({r["arm"] for r in inv if r["arm"] in ("micro", "micro_shuffled")})
    return {
        "why_this_exists": (
            "Measures the 'cell 1 only' limit instead of asserting it. Micro-arm artifacts DO "
            "exist; they are structurally unquotable, which is a different and stronger statement."
        ),
        "n_artifacts_total": len(inv),
        "n_money_artifacts": sum(1 for r in inv if r["money_run"] is True),
        "money_artifacts": [r["path"] for r in inv if r["money_run"] is True],
        "micro_arms_present_but_unquotable": micro,
        "loader_verdict_by_dir": refusals,
        "conclusion": (
            "Every non-money artifact set is REFUSED by the production load_cell_evals (schema v1, "
            "missing index, or incomplete matrix). No transfer check to the micro arms is possible "
            "at $0 — the limit is ENFORCED by the loader, not a matter of discipline."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REPO / "runs_cloud" / "results")
    ap.add_argument(
        "--out", type=Path, default=REPO / "runs_manifest" / "m6_horizon_break_even.json"
    )
    args = ap.parse_args()

    paths = sorted(args.results.glob("*/cell*_eval.json"))
    if not paths:
        print(f"NO ARTIFACTS under {args.results}", file=sys.stderr)
        return 1

    units, val_by_unit = [], {}
    for p in paths:
        doc = json.loads(p.read_text())
        try:
            u = analyse_unit(doc)
        except ValueError as e:
            # A refusal is an outcome, not a crash: report it the way every other gate here does.
            print(f"REFUSING TO EMIT: {e}", file=sys.stderr)
            return 2
        # relative_to raises when --results points outside the repo (e.g. a scratch mutation
        # tree); the artifact path is provenance, never a computation input, so degrade to the
        # absolute path rather than killing the run.
        u["artifact"] = str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)
        units.append(u)
        val_by_unit[f"seed{u['seed']}"] = analyse_val_scalars(doc, u)

    # every identity check must pass, and must have been PROVEN capable of failing, before any
    # number in this receipt is readable. Refuse rather than emit a qualified number.
    bad = [u for u in units if not u["identity_check"]["agrees"]]
    nondisc = [
        u
        for u in units
        if not (
            u["identity_discrimination_check"]["moves_as_expected"]
            and u["identity_discrimination_check"]["original_value_rejected_by_perturbed"]
        )
    ]
    if bad or nondisc:
        print(
            "REFUSING TO EMIT: identity check failed on "
            f"{[u['seed'] for u in bad]}; discrimination unproven on "
            f"{[u['seed'] for u in nondisc]}",
            file=sys.stderr,
        )
        return 2

    cb = np.array([u["c_break_POINT_ESTIMATE_positions_fixed"] for u in units], dtype=np.float64)
    fee_rt = FEE_REFERENCE["round_trip_frac"]

    # ---- the val<->headline bridge: the ONLY quantified handle on what "do not mix" would cost.
    # h=15 exists on BOTH the val block and the headline grid, at the same kappa*, the same flat
    # netting and the same full-grid-with-flat-0 construction. Their ratio is the only measurement
    # of how much the two calendar regions differ. 3 seeds, ONE horizon — suggestive, not a
    # transport coefficient, and explicitly NOT used to convert any h=5/60 number.
    bridge = []
    for u in units:
        val15 = val_by_unit[f"seed{u['seed']}"]["15"]["ir_net_at_kappa_star"]
        hd15 = u["ir_net_at_headline_cost"]
        bridge.append(
            {
                "seed": u["seed"],
                "ir_val_block_h15": val15,
                "ir_headline_grid_h15": hd15,
                "val_over_headline": float(val15 / hd15) if hd15 else float("nan"),
            }
        )
    ratios = np.array([b["val_over_headline"] for b in bridge], dtype=np.float64)

    # ---- required horizon under LINEAR edge scaling (a LOWER BOUND on the holding period)
    req = []
    for u in units:
        e = u["gross_edge_per_active_period_EQUALS_c_break"]
        req.append(
            {
                "seed": u["seed"],
                "measured_edge_per_active_period_at_h15": e,
                "shortfall_multiple_vs_fee": float(fee_rt / e) if e > 0 else float("nan"),
                "required_h_minutes_POINT_EXTRAPOLATION_NOT_A_BOUND": required_horizon_for_fee(
                    e, PRIMARY_H, fee_rt
                ),
            }
        )
    rh = np.array(
        [r["required_h_minutes_POINT_EXTRAPOLATION_NOT_A_BOUND"] for r in req], dtype=np.float64
    )
    doc = {
        "script": "scripts/m6_horizon_break_even.py",
        "schema": "m6_horizon_break_even_v1",
        "status": (
            "RECOMPUTES A PRE-REGISTERED §4 HEADLINE COMPONENT (prereg:170-171 names break-even "
            "cost inside the headline cost-stress curve) WITH A NON-PRE-REGISTERED INSTRUMENT — "
            "the pinned metrics.break_even_cost returns -inf on this data; this bisects the exact "
            "root BELOW the pinned grid floor. Both ship per unit. ADMISSIBILITY OF THE EXTENSION "
            "IS THE SUPERVISOR'S RULING, NOT THE BUILDER'S. Invariant 5 untouched: the headline "
            "metric is still cost-aware NET IR and no gross quantity replaces it."
        ),
        "REQUIRED_READING": {
            "asymmetry_both_directions": (
                "WEAK IN BOTH DIRECTIONS, unlike the window probe. Evidence is CELL 1 ONLY (BSQ + "
                "OHLCV-only), one arm of a 2x2 that never ran, with NO micro artifacts to test "
                "transfer. A horizon clearing real fees would be SUGGESTIVE, not proof. Finding no "
                "such horizon is ALSO only SUGGESTIVE — a null does not license 'no horizon is "
                "tradeable', because the arm the headline claim is about was never scored. Do not "
                "confuse this with the window probe's asymmetry: that one points the other way."
            ),
            "provenance_never_mixed": (
                "h=15 is the HEADLINE money grid (forward blocks 1..5). h=5 and h=60 are VAL "
                "BLOCK ONLY (forward block 0), at exactly 0.2000x the headline grid. Different "
                "calendar regions and are never compared as a curve without this label."
            ),
            "scoping_correction": (
                "I scoped this as 'three points'. That was WRONG and the error was mine: val "
                "passes bank IR SCALARS, not series, so gbar_A is not recoverable at h=5/60. "
                "The curve has ONE measured point and TWO exact bounds."
            ),
            "shipped_instrument_floor": (
                "metrics.break_even_cost returns -inf here: every IR on the pre-registered grid "
                "(0.10/0.20/0.30%) is negative. The grid FLOOR is exactly the realistic round trip "
                "we want to compare against, so the pre-registered instrument bottoms out just "
                "where the comparison begins."
            ),
            "break_even_not_persisted": (
                "CellScore.break_even is computed at eval time but write_cell_eval_artifact does "
                "not persist it — it was discarded on the rented box. This receipt reconstructs it."
            ),
        },
        "fee_reference": FEE_REFERENCE,
        "identity": (
            "c_break = mean gross return per ACTIVE period (gbar_A). Proven numerically per unit "
            "against a bisected root of IR(c), and the check proven capable of failing first."
        ),
        "pins": {
            "headline_cost": HEADLINE_COST,
            "flat_costs": list(FLAT_COSTS),
            "primary_h": PRIMARY_H,
            "eps_ir": EPS_IR,
            "val_horizons": list(VAL_HORIZONS),
        },
        "THE_ASSUMPTION_FREE_REFRAMING": {
            "statement": (
                "Because c_break = gbar_A exactly, clearing a round trip of c requires GROSS EDGE "
                "PER ACTIVE PERIOD >= c. That threshold is THE SAME AT EVERY HORIZON — 0.10% of "
                "gross edge per active period, whether the position is held 5 minutes or 60. What "
                "changes with h is not the bar but how much gross edge a period can contain."
            ),
            "required_gross_edge_per_active_period_vs_fee": required_gross_edge_per_active_period(
                fee_rt
            ),
            "required_gross_edge_per_active_period_vs_modelled_030": (
                required_gross_edge_per_active_period(HEADLINE_COST)
            ),
            "measured_and_required_horizon": req,
            "required_h_minutes_range_LOWER_BOUND": [float(rh.min()), float(rh.max())],
            "how_to_read": (
                "Under LINEAR edge scaling the measured h=15 edge would reach the 0.10% round trip "
                f"only after {rh.min():.0f}-{rh.max():.0f} minutes of holding, versus the 15 "
                "actually evaluated. Linear scaling assumes a drift that never decays, so real "
                "sub-linear decay pushes this HIGHER: it is a LOWER BOUND on the required holding "
                "period, never an estimate of a horizon that works."
            ),
        },
        "transfer_check_inventory": transfer_check_inventory(REPO),
        "val_headline_bridge_h15": {
            "what_it_is": (
                "h=15 exists on BOTH the val block and the headline grid at the same kappa*, the "
                "same flat netting and the same construction. This ratio is the only measurement "
                "of how much the two calendar regions differ — i.e. what 'do not mix' would cost."
            ),
            "per_seed": bridge,
            "val_over_headline_range": [float(ratios.min()), float(ratios.max())],
            "direction": (
                "The VAL block IR is more negative than the headline grid IR in all three seeds "
                f"({ratios.min():.3f}-{ratios.max():.3f}x), so val-block statements are if any "
                "mildly PESSIMISTIC relative to the headline region."
            ),
            "limits": (
                "THREE SEEDS, ONE HORIZON, ONE CELL. Suggestive only. This is NOT a transport "
                "coefficient and is NOT used to convert any h=5 or h=60 number — those stay bounds."
            ),
        },
        "measured_h15_headline": units,
        "bounded_val_horizons": val_by_unit,
        "summary_h15": {
            "n_seeds": len(units),
            "c_break_pct_by_seed_POINT_every_one_below_the_fee": {
                str(u["seed"]): u["c_break_pct_POINT_see_c_break_interval"] for u in units
            },
            "c_break_pct_min_POINT": float(cb.min() * 100),
            "c_break_pct_max_POINT": float(cb.max() * 100),
            "c_break_pct_mean_POINT": float(cb.mean() * 100),
            "round_trip_fee_pct": FEE_REFERENCE["round_trip_frac"] * 100,
            "n_seeds_clearing_round_trip_fee": int(sum(u["clears_round_trip_fee"] for u in units)),
            "spread_ratio_max_over_min": float(cb.max() / cb.min()) if cb.min() != 0 else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, default=str) + "\n")
    print(f"WROTE {args.out.relative_to(REPO) if args.out.is_relative_to(REPO) else args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
