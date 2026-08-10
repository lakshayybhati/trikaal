"""Figure 2 — capacity eviction: reconstruction by variance class, three seeds.

VERDICT THE FIGURE MUST CARRY WITHOUT ITS CAPTION
    Two channels with identical marginal variance receive opposite treatment. The eleven that are
    mutually correlated are coded at 0.82-0.92; the one that is statistically independent is coded
    at ~0.005. The three seeds agree. Therefore the objective is allocating by covariance, not by
    chance and not by task relevance.

ARTIFACTS — every plotted number is loaded at render time; nothing here is transcribed.
    runs_manifest/m6_interface_respec_design_pass.json
      .tokenizers.new_pointwise_fine_3seed.seeds.{0,1,2}.nonlinear_diagnostics
          .point_decoder_per_dim_corrs        -> the 13 bar heights, per seed
      .tokenizers.new_pointwise_fine_3seed.seeds.0.recon.per_dim.<i>.mean_baseline_mae
          -> marginal sd, via E|X| = sigma * sqrt(2/pi); dim 0 -> 3.15, all others -> 1.00
      .filler_calibration.filler_rho          -> the AR(1) coupling, 0.7993
      .gates.gate2_legibility.threshold       -> the 0.90 rule
      .gates.channel_receipt_non_gating.dims_ge_090_by_seed -> dims clearing 0.90, per seed

    The fixture's construction (scripts/m6_canary.py:220-247) gives the return dimension a marginal
    sd of sqrt(1 + c^2) = sqrt(10) = 3.162 exactly. The figure and the caption both report 3.15,
    the finite-sample value the receipt supports, so that no number in the manuscript is larger
    than what an auditor opening the receipt can reproduce.

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig2_eviction.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"

fs.apply()

RETURN_DIM, SIGNAL_DIM = 0, 9
FILLER_EDGE = "#A08C33"  # darker rim for the sand fill, so the bars read on white


def main() -> None:
    with open(RM / "m6_interface_respec_design_pass.json") as fh:
        d = json.load(fh)
    seeds = d["tokenizers"]["new_pointwise_fine_3seed"]["seeds"]
    corr = np.array(
        [seeds[s]["nonlinear_diagnostics"]["point_decoder_per_dim_corrs"] for s in ("0", "1", "2")]
    )  # [3 seeds, 13 dims]
    recon = seeds["0"]["recon"]["per_dim"]
    sd = np.array([recon[str(i)]["mean_baseline_mae"] / np.sqrt(2 / np.pi) for i in range(13)])
    rho = d["filler_calibration"]["filler_rho"]
    thresh = d["gates"]["gate2_legibility"]["threshold"]
    n_clear = d["gates"]["channel_receipt_non_gating"]["dims_ge_090_by_seed"]
    # countable from the bars; reported in the caption, asserted here so a receipt
    # shape change cannot silently invalidate the caption text
    assert set(n_clear.values()) == {4}, n_clear

    fillers = [i for i in range(13) if i not in (RETURN_DIM, SIGNAL_DIM)]
    lo, hi = corr[:, fillers].min(), corr[:, fillers].max()
    mean = corr.mean(axis=0)

    # x layout: three variance classes, separated so the grouping is pre-attentive
    xpos = np.empty(13)
    xpos[RETURN_DIM] = 0.0
    for k, i in enumerate(fillers):
        xpos[i] = 1.75 + k
    xpos[SIGNAL_DIM] = 13.5
    W = 0.74

    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.62))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, zorder=0)

    # the evicted channel gets a faint field behind it, so the eye lands there first
    ax.add_patch(
        Rectangle(
            (xpos[SIGNAL_DIM] - 0.85, -0.045),
            1.7,
            1.12,
            fc=fs.FAIL,
            alpha=0.055,
            ec="none",
            zorder=0,
        )
    )

    for i in range(13):
        if i == RETURN_DIM:
            fc, ec = fs.OHLCV, fs.OHLCV
        elif i == SIGNAL_DIM:
            fc, ec = fs.FAIL, fs.FAIL
        else:
            fc, ec = fs.FILLER, FILLER_EDGE
        ax.bar(xpos[i], mean[i], width=W, fc=fc, ec=ec, lw=0.5, zorder=3)
        # the three seeds, drawn on the bar: agreement is the point, so it must be visible
        for k in range(3):
            ax.plot(
                [xpos[i] - W * 0.36, xpos[i] + W * 0.36],
                [corr[k, i], corr[k, i]],
                lw=0.7,
                color="white" if mean[i] > 0.1 else fs.INK,
                solid_capstyle="butt",
                zorder=4,
            )

    ax.axhline(thresh, color=fs.RULE, ls=(0, (4, 3)), lw=0.7, zorder=5)
    ax.text(
        14.45,
        thresh + 0.012,
        f"legibility target {thresh:.2f}",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=fs.RULE,
    )

    # ---- direct class labels, in the class colours -----------------------------------------
    ax.text(
        xpos[RETURN_DIM],
        mean[RETURN_DIM] + 0.035,
        f"return\nsd {sd[RETURN_DIM]:.2f}",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=fs.OHLCV,
        linespacing=1.3,
    )
    fx = xpos[fillers].mean()
    ax.text(
        fx,
        hi + 0.055,
        f"eleven correlated fillers  ·  sd 1.00  ·  AR(1) across dims, $\\rho$ = {rho}\n"
        f"coded at {lo:.2f} – {hi:.2f}",
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=FILLER_EDGE,
        linespacing=1.4,
    )
    ax.annotate(
        f"independent  ·  sd 1.00\n{corr[:, SIGNAL_DIM].min():.3f} – "
        f"{corr[:, SIGNAL_DIM].max():.3f}\nnot coded",
        xy=(xpos[SIGNAL_DIM], 0.028),
        xytext=(xpos[SIGNAL_DIM], 0.44),
        ha="center",
        va="bottom",
        fontsize=6.4,
        color=fs.FAIL,
        linespacing=1.4,
        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=fs.FAIL, shrinkA=2, shrinkB=1),
    )

    # ---- the brace that makes the verdict readable without the caption ---------------------
    y_br = -0.190
    x0, x1 = xpos[fillers[0]] - 0.45, xpos[SIGNAL_DIM] + 0.45
    for xa, xb in ((x0, x0), (x1, x1)):
        ax.plot([xa, xb], [y_br, y_br + 0.028], lw=0.6, color=fs.INK, clip_on=False, zorder=6)
    ax.plot([x0, x1], [y_br, y_br], lw=0.6, color=fs.INK, clip_on=False, zorder=6)
    ax.text(
        (x0 + x1) / 2,
        y_br - 0.032,
        "identical marginal variance \u2014 opposite outcome",
        ha="center",
        va="top",
        fontsize=6.7,
        color=fs.INK,
        weight="bold",
        clip_on=False,
    )

    # ---- seed key, drawn as the thing it depicts, in the free strip beside the brace -------
    y_key = y_br - 0.032
    ax.plot([-0.82, -0.50], [y_key, y_key], lw=0.7, color=fs.INK, clip_on=False, zorder=6)
    ax.text(
        -0.42,
        y_key,
        "one rule per seed",
        ha="left",
        va="center",
        fontsize=6.2,
        color=fs.RULE,
        clip_on=False,
    )

    ax.set_xticks(xpos)
    for i, lab in enumerate(ax.set_xticklabels([str(i) for i in range(13)])):
        lab.set_color(
            fs.OHLCV if i == RETURN_DIM else (fs.FAIL if i == SIGNAL_DIM else FILLER_EDGE)
        )
    ax.set_xlim(-0.85, 14.5)
    ax.set_ylim(0, 1.075)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("feature dimension", labelpad=2)
    ax.set_ylabel("point-decoder correlation")
    ax.spines["bottom"].set_position(("data", 0))

    fs.save(fig, OUT, "fig2_eviction_by_variance_class")


if __name__ == "__main__":
    main()
