"""Figure 13 — three seeds of one configuration, and what their disagreement does to the money path.

THE FINDING THE FIGURE MUST CARRY WITHOUT ITS CAPTION
    Three units that differ only by seed do not merely score differently -- they take structurally
    DIFFERENT DIRECTIONAL POSITIONS, two leaning short and one leaning long, and that difference
    propagates all the way to a 24x spread in how often they trade and a 118-point spread in
    information ratio. Yet the gross edge of the two short-leaning units is earned AGAINST their
    own lean, in a sample that rose.

WHY THESE THREE PANELS AND NOT SEVEN FIGURES. The supervisor's candidate list had the seed
disagreement, the over-dispersion mechanism and the skill-versus-bias result as three separate
figures. They are three views of the SAME three units and the same 1,402,520 decisions each, and
splitting them would make a reader assemble the chain that is the actual point. One figure, three
panels, left to right in causal order: what the forecasts look like, how the book leans, what it
costs.

ARTIFACTS — loaded at render time; nothing transcribed.
    runs_cloud/results/{r0/cell1_seed0,r1/cell1_seed2,r2/cell1_seed4}_eval.json
        .mu_diag.{std, frac_negative, activity_decisions, n}   -> panels (a) and (b)
    runs_manifest/m6_skill_vs_bias.json
        .rows[*].{modal_sign, a_ir_gross_model, b_ir_gross_constant_direction, delta_ci95_lower}
        .market_drift.annualized_ir_of_always_long_market      -> the drift the lean faces
    runs_manifest/m6_skill_vs_bias_exact.json
        .a_ir_gross_model, .b_ir_gross_constant_direction_Ap   -> seed 4 on the SPECIFIED benchmark

THE 25-446x OVER-DISPERSION FIGURE IS NOT DRAWN. It appears in no receipt I could locate, so the
panel uses the two ratios that ARE sourced: std(mu-hat) against the return sd it forecasts, and
against the sd a CALIBRATED forecast at our own measured prior would have.

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig13_seeds.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"

fs.apply()

EVALS = {
    0: "runs_cloud/results/r0/cell1_seed0_eval.json",
    2: "runs_cloud/results/r1/cell1_seed2_eval.json",
    4: "runs_cloud/results/r2/cell1_seed4_eval.json",
}
MU = {s: json.loads((ROOT / p).read_text())["mu_diag"] for s, p in EVALS.items()}
IR = {s: json.loads((ROOT / p).read_text())["val_ir_by_kappa_by_h"] for s, p in EVALS.items()}
SVB = json.loads((RM / "m6_skill_vs_bias.json").read_text())
EXACT = json.loads((RM / "m6_skill_vs_bias_exact.json").read_text())
BE = json.loads((RM / "m6_horizon_break_even.json").read_text())

# net IR at the pinned headline cost, from the break-even receipt's own record of each unit
NET_IR = {r["seed"]: r["ir_net_at_headline_cost"] for r in BE["measured_h15_headline"]}

# the calibrated reference: a forecast with our measured prior would have std = IC * std(y)
PRIOR_RANKIC = 0.027  # M2 univariate screen, quoted in §7.7
STD_Y15 = 0.003504  # full BTCUSDT history, §7.7 / the activity-leg ruling
CALIB_STD = PRIOR_RANKIC * STD_Y15

SEEDS = (0, 2, 4)


def _row(seed: int) -> dict:
    for r in SVB["rows"]:
        if r["seed"] == seed:
            return r
    raise AssertionError(f"skill-vs-bias receipt has no seed {seed}")


def _verify() -> None:
    """The figure claims a divergence and an inversion. Fail if the receipts stop supporting it."""
    for s in SEEDS:
        if MU[s]["n"] != 1_402_520:
            raise AssertionError(f"seed {s}: decision count {MU[s]['n']} is not the money grid")
        if MU[s]["estimator"] != "expectation":
            raise AssertionError(f"seed {s}: estimator is {MU[s]['estimator']}, not the pinned one")
    leans = {s: ("SHORT" if MU[s]["frac_negative"] > 0.5 else "LONG") for s in SEEDS}
    if len(set(leans.values())) != 2:
        raise AssertionError(f"the panel claims the seeds lean differently; leans are {leans}")
    for s in SEEDS:
        if leans[s] != _row(s)["modal_sign"]:
            raise AssertionError(
                f"seed {s}: mu_diag lean {leans[s]} disagrees with the skill receipt's "
                f"{_row(s)['modal_sign']}"
            )
    for s in (0, 2):
        if _row(s)["b_ir_gross_constant_direction"] >= 0:
            raise AssertionError(
                f"seed {s}: the panel claims its lean alone LOSES; receipt says "
                f"{_row(s)['b_ir_gross_constant_direction']}"
            )
    if SVB["market_drift"]["mean_per_period"] <= 0:
        raise AssertionError("the panel claims the sample drifted UP; the receipt disagrees")


def main() -> None:
    _verify()
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(fs.SINGLE_COL, 2.30), gridspec_kw={"width_ratios": [1.0, 1.06, 1.22]}
    )
    x = np.arange(len(SEEDS))
    cols = {0: fs.BSQ, 2: fs.OHLCV, 4: fs.MICRO}
    labels = [f"seed {s}" for s in SEEDS]

    # ---- (a) the forecasts are far wider than what they forecast ------------------------------
    fs.panel_title(axA, "a", "forecast spread")
    for i, s in enumerate(SEEDS):
        axA.bar(i, MU[s]["std"] / STD_Y15, width=0.62, color=cols[s], zorder=3)
        axA.text(
            i,
            MU[s]["std"] / STD_Y15 + 0.06,
            f"{MU[s]['std'] / STD_Y15:.1f}×",
            ha="center",
            va="bottom",
            fontsize=fs.T_MICRO,
            color=cols[s],
            zorder=5,
        )
    axA.axhline(1.0, color=fs.RULE, ls=(0, (3, 2)), lw=0.8, zorder=4)
    axA.text(
        len(SEEDS) - 0.42,
        0.90,
        "sd of the returns\nbeing forecast",
        ha="right",
        va="top",
        fontsize=fs.T_MICRO,
        color=fs.RULE,
        linespacing=1.3,
        zorder=5,
    )
    axA.set_ylabel("sd$(\\hat{\\mu})$ ÷ sd$(y_{15})$", fontsize=fs.T_LABEL)
    axA.set_ylim(0, 3.9)

    # ---- (b) and the book leans in different DIRECTIONS ---------------------------------------
    fs.panel_title(axB, "b", "directional lean")
    for i, s in enumerate(SEEDS):
        fneg = MU[s]["frac_negative"]
        axB.bar(i, fneg, width=0.62, color=cols[s], zorder=3)
        axB.text(
            i,
            fneg + 0.028,
            f"{fneg:.3f}",
            ha="center",
            va="bottom",
            fontsize=fs.T_MICRO,
            color=cols[s],
            zorder=5,
        )
        axB.text(
            i,
            0.035,
            "SHORT" if fneg > 0.5 else "LONG",
            ha="center",
            va="bottom",
            fontsize=fs.T_MICRO,
            color="white",
            weight="bold",
            zorder=6,
        )
    axB.axhline(0.5, color=fs.RULE, ls=(0, (3, 2)), lw=0.8, zorder=4)
    axB.set_ylabel("fraction of forecasts negative", fontsize=fs.T_LABEL)
    axB.set_ylim(0, 0.98)

    # ---- (c) the consequence: gross edge earned AGAINST the lean ------------------------------
    fs.panel_title(axC, "c", "gross IR: model vs its lean")
    w = 0.34
    for i, s in enumerate(SEEDS):
        r = _row(s)
        model = r["a_ir_gross_model"]
        bias = r["b_ir_gross_constant_direction"]
        if s == 4:  # the specified benchmark exists for seed 4 only
            model = EXACT["a_ir_gross_model"]
            bias = EXACT["b_ir_gross_constant_direction_Ap"]
        axC.bar(i - w / 2, model, width=w, color=cols[s], zorder=3)
        axC.bar(
            i + w / 2,
            bias,
            width=w,
            color="white",
            edgecolor=cols[s],
            lw=0.8,
            hatch="////",
            zorder=3,
        )
        if bias < 0:
            axC.text(
                i + w / 2,
                bias - 0.10,
                "lean\nLOSES",
                ha="center",
                va="top",
                fontsize=fs.T_MICRO,
                color=fs.FAIL,
                weight="bold",
                linespacing=1.25,
                zorder=6,
            )
    axC.axhline(0.0, color=fs.INK, lw=0.7, zorder=4)
    axC.set_ylabel("gross IR, annualized", fontsize=fs.T_LABEL)
    axC.set_ylim(-2.35, 3.05)
    axC.text(
        -0.42,
        2.72,
        "solid = model   hatched = constant direction",
        ha="left",
        va="center",
        fontsize=fs.T_MICRO,
        color=fs.RULE,
        zorder=6,
    )

    for ax in (axA, axB, axC):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=fs.T_TICK)
        ax.tick_params(axis="y", labelsize=fs.T_TICK)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.set_xlim(-0.62, len(SEEDS) - 0.38)

    fs.finding(
        fig,
        "Same configuration, three seeds: two lean short and one leans long — and the short ones "
        "earn their edge against that lean",
    )
    act = [MU[s]["activity_decisions"] for s in SEEDS]
    fs.scope(
        fig,
        f"cell 1 only (BSQ, OHLCV-only), {MU[0]['n']:,} decisions per unit at the pinned h = 15. "
        f"The lean propagates: decision activity spans\n"
        f"{max(act) / min(act):.0f}× ({min(act):.4f}–{max(act):.4f}) and net IR at the pinned "
        f"cost spans {max(NET_IR.values()) - min(NET_IR.values()):.0f} points. In (c) the sample "
        "drifted "
        f"UP (always-long IR\n"
        f"{SVB['market_drift']['annualized_ir_of_always_long_market']:+.4f}), so a short lean "
        "loses "
        "by construction. Seed 4 uses the benchmark of the names it traded; 0 and 2 the evaluated "
        "cross-section.",
    )
    fig.subplots_adjust(top=0.80, bottom=0.345, left=0.075, right=0.995, wspace=0.44)
    fs.assert_text_legible(fig, (axA, axB, axC))
    fs.save(fig, OUT, "fig13_seeds")


if __name__ == "__main__":
    main()
