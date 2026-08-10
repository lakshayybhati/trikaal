"""Figure 6 — the evaluation geometry: train once, evaluate forward, purge and embargo.

VERDICT THE FIGURE MUST CARRY WITHOUT ITS CAPTION
    Training stops at one boundary and never moves. The forward region is cut into six blocks; the
    first is spent on selecting the execution filter and is excluded from every headline number, so
    the headline is scored on five blocks the selection never saw. Between a training window and any
    label that could contaminate it sit a 60-bar label look-forward and a 120-bar embargo, and the
    premise behind that margin is measured rather than assumed -- with the caveat that it holds for
    the signed-return channel and not for absolute returns.

ARTIFACTS — loaded at render time.
    runs_manifest/m6_mde_inputs.json          .primary_region_ms, .n_blocks
    runs_manifest/m6_c20_embargo_premise.json .reading.{H_MAX_DEFAULT, L_CORR_DEFAULT,
                                                EMBARGO_DEFAULT, ALL200_signed_acf_at_60_mean,
                                                ALL200_signed_acf_at_60_worst,
                                                abs_acf_at_L_corr_60_mean,
                                                abs_acf_at_L_corr_60_worst_symbol,
                                                premise_supported_for_*}
                                              .control_arm.{series, recovers_known_answer}

The train/eval boundary (1697820480000 ms) is the 0.7 point of the four-year window, from the
committed fold plan; the six forward blocks are equal divisions of boundary -> primary_region end.

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig6_validation.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"

fs.apply()

MDE = json.loads((RM / "m6_mde_inputs.json").read_text())
EMB = json.loads((RM / "m6_c20_embargo_premise.json").read_text())["reading"]
CTRL = json.loads((RM / "m6_c20_embargo_premise.json").read_text())["control_arm"]

K = MDE["n_blocks"]
PRIM0, PRIM1 = MDE["primary_region_ms"]
BOUNDARY = 1_697_820_480_000
LAKE0 = 1_609_459_200_000  # 2021-01-01T00:00Z, the lake's first bar month
H_MAX, L_CORR, EMBARGO = EMB["H_MAX_DEFAULT"], EMB["L_CORR_DEFAULT"], EMB["EMBARGO_DEFAULT"]


def d(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.UTC).strftime("%Y-%m-%d")


def main() -> None:
    fig = plt.figure(figsize=(fs.SINGLE_COL, 3.25))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.0, 1.15, 0.85],
        hspace=0.62,
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.06,
    )

    # ---------------------------------------------------------------- (a) the timeline
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(LAKE0, PRIM1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        LAKE0,
        1.10,
        "(a)  train once, evaluate forward",
        ha="left",
        va="bottom",
        fontsize=7.2,
        weight="bold",
    )

    ax.add_patch(
        Rectangle((LAKE0, 0.42), BOUNDARY - LAKE0, 0.34, fc=fs.GRID, ec="#bfbfbf", lw=0.6, zorder=2)
    )
    ax.text(
        (LAKE0 + BOUNDARY) / 2,
        0.59,
        "training region — used once, never refit",
        ha="center",
        va="center",
        fontsize=6.5,
        color=fs.INK,
        zorder=3,
    )

    span = (PRIM1 - BOUNDARY) / K
    for b in range(K):
        x0 = BOUNDARY + b * span
        val = b == 0
        ax.add_patch(
            Rectangle(
                (x0, 0.42),
                span,
                0.34,
                fc=fs.PLACEBO if val else fs.OHLCV,
                alpha=0.30 if val else 0.22,
                ec=fs.RULE if val else fs.OHLCV,
                lw=0.7,
                zorder=2,
            )
        )
        ax.text(
            x0 + span / 2,
            0.59,
            f"b{b}",
            ha="center",
            va="center",
            fontsize=6.4,
            color=fs.RULE if val else fs.OHLCV,
            weight="bold",
            zorder=3,
        )
    ax.annotate(
        "VAL — $\\kappa$ selection,\nexcluded from every headline number",
        xy=(BOUNDARY + span / 2, 0.42),
        xytext=(BOUNDARY - span * 0.4, 0.10),
        ha="right",
        va="center",
        fontsize=5.8,
        color=fs.RULE,
        linespacing=1.35,
        arrowprops=dict(arrowstyle="-|>", lw=0.5, color=fs.RULE, shrinkA=1, shrinkB=2),
    )
    ax.annotate(
        "",
        xy=(PRIM1, 0.86),
        xytext=(BOUNDARY + span, 0.86),
        arrowprops=dict(arrowstyle="|-|,widthA=0.35,widthB=0.35", lw=0.7, color=fs.OHLCV),
    )
    ax.text(
        BOUNDARY + span + (PRIM1 - BOUNDARY - span) / 2,
        0.90,
        "headline: blocks 1–5",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=fs.OHLCV,
    )
    for i, (x, lab) in enumerate(
        ((LAKE0, d(LAKE0)), (BOUNDARY, d(BOUNDARY)), (PRIM0, d(PRIM0)), (PRIM1, d(PRIM1)))
    ):
        low = i == 2
        ax.plot([x, x], [0.40, 0.22 if low else 0.34], lw=0.6, color=fs.RULE)
        ax.text(x, 0.16 if low else 0.28, lab, ha="center", va="top", fontsize=5.8, color=fs.RULE)

    # ---------------------------------------------------------------- (b) purge / embargo
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(-30, 330)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(
        -30,
        1.06,
        "(b)  what separates a training window from the label it must not see",
        ha="left",
        va="bottom",
        fontsize=7.2,
        weight="bold",
    )

    ax2.add_patch(Rectangle((-20, 0.60), 120, 0.20, fc=fs.GRID, ec="#bfbfbf", lw=0.6))
    ax2.text(40, 0.70, "training window", ha="center", va="center", fontsize=6.3)
    ax2.add_patch(
        Rectangle(
            (100, 0.60), EMBARGO, 0.20, fc=fs.FAIL, alpha=0.16, ec=fs.FAIL, lw=0.8, hatch="////"
        )
    )
    ax2.text(
        100 + EMBARGO / 2,
        0.70,
        f"embargo  {EMBARGO} bars",
        ha="center",
        va="center",
        fontsize=6.3,
        color=fs.FAIL,
    )
    ax2.add_patch(
        Rectangle((100 + EMBARGO, 0.60), 90, 0.20, fc=fs.OHLCV, alpha=0.20, ec=fs.OHLCV, lw=0.8)
    )
    ax2.text(
        100 + EMBARGO + 45,
        0.70,
        "evaluation",
        ha="center",
        va="center",
        fontsize=6.3,
        color=fs.OHLCV,
    )

    for x0, w, y, dy, lab in (
        (100, H_MAX, 0.47, 0.085, f"label look-forward  $H_{{\\max}}$ = {H_MAX} bars"),
        (
            100 + H_MAX,
            L_CORR,
            0.47,
            -0.085,
            f"serial-correlation margin  $L_{{corr}}$ = {L_CORR} bars",
        ),
    ):
        ax2.add_patch(
            FancyArrowPatch(
                (x0, y), (x0 + w, y), arrowstyle="|-|,widthA=0.3,widthB=0.3", lw=0.7, color=fs.INK
            )
        )
        ax2.text(x0 + w / 2, y + dy, lab, ha="center", va="center", fontsize=5.9, color=fs.INK)
    ax2.text(
        100 + EMBARGO / 2,
        0.22,
        f"embargo = $H_{{\\max}}$ + $L_{{corr}}$ = {EMBARGO} bars",
        ha="center",
        va="center",
        fontsize=6.2,
        weight="bold",
        color=fs.FAIL,
    )

    # ---------------------------------------------------------------- (c) the measured premise
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis("off")
    ax3.text(
        0.0,
        1.02,
        f"(c)  is the {L_CORR}-bar margin earned?  measured on all 200 instruments",
        ha="left",
        va="bottom",
        fontsize=7.2,
        weight="bold",
    )

    rows = [
        (
            "signed return — the channel a label leak would travel on",
            EMB["ALL200_signed_acf_at_60_mean"],
            EMB["ALL200_signed_acf_at_60_worst"],
            bool(EMB["ALL200_premise_supported_for_SIGNED_returns"]),
        ),
        (
            "absolute return — volatility clustering, not a label channel",
            EMB["abs_acf_at_L_corr_60_mean"],
            EMB["abs_acf_at_L_corr_60_worst_symbol"],
            bool(EMB["premise_supported_for_ABS_returns"]),
        ),
    ]
    for i, (lab, mean, worst, ok) in enumerate(rows):
        y = 0.72 - i * 0.38
        col = fs.PASS if ok else fs.FAIL
        ax3.text(0.0, y + 0.11, lab, ha="left", va="center", fontsize=6.1, color=fs.INK)
        ax3.add_patch(
            Rectangle(
                (0.0, y - 0.055),
                0.52 * min(worst / 0.40, 1.0),
                0.075,
                fc=col,
                alpha=0.22,
                ec="none",
            )
        )
        ax3.add_patch(
            Rectangle(
                (0.0, y - 0.055), 0.52 * min(mean / 0.40, 1.0), 0.075, fc=col, alpha=0.65, ec="none"
            )
        )
        ax3.text(
            0.995,
            y + 0.11,
            "premise holds" if ok else "premise does NOT hold",
            ha="right",
            va="center",
            fontsize=6.2,
            weight="bold",
            color=col,
        )
        ax3.text(
            0.56,
            y - 0.018,
            f"ACF at lag {L_CORR}:   mean {mean:.4f}    worst {worst:.4f}",
            ha="left",
            va="center",
            fontsize=6.0,
            color=fs.RULE,
        )
    ax3.text(
        0.0,
        0.03,
        f"control arm recovers a known answer first ({CTRL['series']}), so a null "
        "reading is a measurement and not a silent failure.",
        ha="left",
        va="center",
        fontsize=5.9,
        color=fs.RULE,
    )

    fs.save(fig, OUT, "fig6_validation")


if __name__ == "__main__":
    main()
