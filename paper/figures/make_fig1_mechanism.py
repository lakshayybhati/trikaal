"""Figure 1 (page-1 hero) — the interface defect and its repair, before and after.

VERDICT THE FIGURE MUST CARRY WITHOUT ITS CAPTION
    Left: one encoder feeds both halves of the token, so bar t's own state is spread across the
    tokens of every later bar -- the window reconstructs and the per-bar symbol carries nothing.
    Right: the fine half is produced from bar t alone, so the state stays where the autoregressive
    stage can reach it. Same bits per bar, same vocabulary, same backbone. The readouts underneath
    are the same fixture at the same budget with the same planted signal: not recovered, recovered.

ARTIFACTS — every number is loaded at render time.
    src/trikaal/constants.py                    FSQ_LEVELS, FSQ_V_C, FSQ_V_F -> bit budgets
    src/trikaal/train/gates.py                  MICRO_LEGIBILITY_MIN -> the 0.90 gate
    runs_manifest/m6_token_control_step0.json   id_visibility_probe.logistic_sign_accuracy -> 0.5135
    runs_manifest/m6_lambda_search_receipt.json formal_cell_path_pin3_omp3.mean
                                                -> post-fix legibility
                                                pin.PINNED_MICRO_POINT_WEIGHT -> lambda
    runs_manifest/m6_canary_v6_stage1_manifest.json     decision_inputs.val_planted_minus_noise
    runs_manifest/m6_acceptance_stage1_manifest.json    decision_inputs.val_planted_minus_noise

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig1_mechanism.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT / "src"))
from trikaal.constants import FSQ_LEVELS, FSQ_V_C, FSQ_V_F, derive_coarse_fine
from trikaal.train.gates import MICRO_LEGIBILITY_MIN

fs.apply()

CI, FI = derive_coarse_fine(FSQ_LEVELS)
BITS_C = sum(math.log2(FSQ_LEVELS[i]) for i in CI)
BITS_F = sum(math.log2(FSQ_LEVELS[i]) for i in FI)


def load(name, *path):
    cur = json.loads((RM / name).read_text())
    for p in path:
        cur = cur[p]
    return cur


LEG_BEFORE = load("m6_token_control_step0.json", "id_visibility_probe", "logistic_sign_accuracy")
LEG_AFTER = load("m6_lambda_search_receipt.json", "formal_cell_path_pin3_omp3", "mean")
LAMBDA = load("m6_lambda_search_receipt.json", "pin", "PINNED_MICRO_POINT_WEIGHT")
DVAL_BEFORE = load(
    "m6_canary_v6_stage1_manifest.json", "decision_inputs", "val_planted_minus_noise"
)
DVAL_AFTER = load(
    "m6_acceptance_stage1_manifest.json", "decision_inputs", "val_planted_minus_noise"
)

N_BARS = 5
COARSE_FC, FINE_FC = "#e8eef5", "#dff0ec"


def token_stack(ax, x, y, w, h, *, fine_fc, fine_ec, coarse_ec):
    """One token: coarse block over fine block, the pair the backbone consumes."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y + h * 0.52),
            w,
            h * 0.48,
            boxstyle="round,pad=0,rounding_size=0.008",
            fc=COARSE_FC,
            ec=coarse_ec,
            lw=0.7,
            zorder=3,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h * 0.48,
            boxstyle="round,pad=0,rounding_size=0.008",
            fc=fine_fc,
            ec=fine_ec,
            lw=0.9,
            zorder=3,
        )
    )


def arrow(ax, p0, p1, *, color, lw=0.7, style="-|>", ls="-", alpha=1.0, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            lw=lw,
            color=color,
            ls=ls,
            alpha=alpha,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
            mutation_scale=6,
            zorder=2,
        )
    )


def panel(ax, *, after: bool):
    accent = fs.OHLCV if after else fs.FAIL
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title = (
        "re-specified — the fine half is per-bar"
        if after
        else "original — one contextual encoder feeds both halves"
    )
    ax.text(0.5, 0.985, title, ha="center", va="bottom", fontsize=7.2, weight="bold", color=accent)

    # ---- the bar stream ---------------------------------------------------------------
    bx = [0.12 + i * 0.19 for i in range(N_BARS)]
    for i, x in enumerate(bx):
        focal = i == 1
        ax.add_patch(
            Rectangle(
                (x - 0.030, 0.855),
                0.060,
                0.070,
                fc=fs.MICRO if focal else "#ececec",
                ec=fs.INK if focal else "#c4c4c4",
                lw=0.7,
                zorder=3,
            )
        )
        ax.plot([x, x], [0.838, 0.945], lw=0.6, color=fs.INK if focal else "#c4c4c4", zorder=2)
    ax.text(bx[1], 0.955, "bar $t$", ha="center", fontsize=6.3, color=fs.INK)

    ENC_Y, ENC_H = 0.645, 0.095
    TOK_Y, TOK_H, TOK_W = 0.380, 0.145, 0.090

    if after:
        boxes = [
            (0.030, 0.445, "contextual encoder\ncausal, bars $\\leq t$", fs.INK, 0.8),
            (0.525, 0.445, "pointwise encoder\nbar $t$ only", fs.OHLCV, 1.1),
        ]
    else:
        boxes = [(0.030, 0.940, "contextual encoder   (causal, bars $\\leq t$)", fs.INK, 0.8)]
    for bx0, bw, lab, ec, lw in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (bx0, ENC_Y),
                bw,
                ENC_H,
                boxstyle="round,pad=0,rounding_size=0.012",
                fc="white",
                ec=ec,
                lw=lw,
                zorder=3,
            )
        )
        ax.text(
            bx0 + bw / 2,
            ENC_Y + ENC_H / 2,
            lab,
            ha="center",
            va="center",
            fontsize=6.6,
            color=ec if ec != fs.INK else fs.INK,
            zorder=4,
            linespacing=1.35,
        )

    # bars -> encoder(s): one bundled arrow per encoder, so nothing crosses
    for bx0, bw, *_ in boxes:
        arrow(
            ax,
            (bx0 + bw / 2, 0.850),
            (bx0 + bw / 2, ENC_Y + ENC_H + 0.006),
            color="#9a9a9a",
            lw=1.1,
        )

    # ---- the token row -----------------------------------------------------------------
    for x in bx:
        token_stack(
            ax,
            x - TOK_W / 2,
            TOK_Y,
            TOK_W,
            TOK_H,
            fine_fc=FINE_FC if after else COARSE_FC,
            fine_ec=fs.OHLCV if after else fs.INK,
            coarse_ec=fs.INK,
        )
    ax.text(
        bx[0] - 0.072,
        TOK_Y + TOK_H * 0.76,
        "coarse",
        ha="right",
        va="center",
        fontsize=5.9,
        color=fs.RULE,
    )
    ax.text(
        bx[0] - 0.072,
        TOK_Y + TOK_H * 0.24,
        "fine",
        ha="right",
        va="center",
        fontsize=5.9,
        color=fs.OHLCV if after else fs.RULE,
    )

    y_c = TOK_Y + TOK_H * 0.76
    y_f = TOK_Y + TOK_H * 0.24
    if after:
        cx = boxes[0][0] + boxes[0][1] / 2
        px = boxes[1][0] + boxes[1][1] / 2
        arrow(
            ax,
            (cx, ENC_Y - 0.006),
            (bx[0] - TOK_W / 2 - 0.010, y_c),
            color="#9a9a9a",
            lw=1.1,
            rad=0.16,
        )
        arrow(
            ax,
            (px, ENC_Y - 0.006),
            (bx[-1] + TOK_W / 2 + 0.010, y_f),
            color=fs.OHLCV,
            lw=1.2,
            rad=-0.16,
        )
        ax.text(
            0.5,
            0.332,
            "each bar's fine half is a function of that bar alone",
            ha="center",
            fontsize=6.3,
            color=fs.OHLCV,
        )
    else:
        arrow(
            ax,
            (0.32, ENC_Y - 0.006),
            (bx[0] - TOK_W / 2 - 0.010, y_c),
            color="#9a9a9a",
            lw=1.1,
            rad=0.16,
        )
        arrow(
            ax,
            (0.68, ENC_Y - 0.006),
            (bx[-1] + TOK_W / 2 + 0.010, y_f),
            color="#9a9a9a",
            lw=1.1,
            rad=-0.16,
        )
        for j in range(1, N_BARS):
            arrow(
                ax,
                (bx[1], 0.845),
                (bx[j], TOK_Y + TOK_H + 0.008),
                color=fs.FAIL,
                lw=0.8,
                alpha=0.9,
                rad=-0.34,
            )
        ax.text(
            0.5,
            0.332,
            "bar $t$'s state is spread across every later bar's token",
            ha="center",
            fontsize=6.3,
            color=fs.FAIL,
        )

    # ---- readout -----------------------------------------------------------------------
    leg = LEG_AFTER if after else LEG_BEFORE
    ax.add_patch(Rectangle((0.01, 0.030), 0.98, 0.250, fc=accent, alpha=0.055, ec="none", zorder=0))
    ax.text(
        0.5,
        0.232,
        "$\\mathrm{sign}(s_t)$ recoverable from bar $t$'s own token",
        ha="center",
        fontsize=6.3,
        color=fs.INK,
        zorder=4,
    )
    ax.text(
        0.5, 0.088, f"{leg:.4f}", ha="center", fontsize=13, weight="bold", color=accent, zorder=4
    )
    ax.text(
        0.5,
        0.048,
        "chance 0.50" if not after else f"gate {MICRO_LEGIBILITY_MIN:.2f}",
        ha="center",
        fontsize=6.0,
        color=fs.RULE,
        zorder=4,
    )


def main() -> None:
    # §3 REVIEW, 2026-08-10: the Delta-val consequence strip is REMOVED. It carried the extraction
    # result, which Figure 3 owns; two figures asserting the same result is one place for them to
    # drift apart, and the strip made this figure argue a conclusion instead of showing a mechanism.
    # Figure 1 is now purely the interface diagram plus the two legibility numbers -- the single
    # quantity the interface change is about. DVAL_BEFORE / DVAL_AFTER stay loaded from their
    # receipt so the removal is a layout decision, not a loss of provenance.
    fig = plt.figure(figsize=(fs.SINGLE_COL, 2.55))
    gs = fig.add_gridspec(
        1,
        2,
        wspace=0.10,
        left=0.055,
        right=0.985,
        top=0.890,
        bottom=0.025,
    )
    panel(fig.add_subplot(gs[0, 0]), after=False)
    panel(fig.add_subplot(gs[0, 1]), after=True)

    # bits parity note, since "unchanged capacity" is the claim that makes the pair fair
    fig.text(
        0.5,
        0.970,
        f"coarse {FSQ_V_C:,} codes / {BITS_C:.2f} bits  +  fine {FSQ_V_F:,} codes / "
        f"{BITS_F:.2f} bits  =  {BITS_C + BITS_F:.3f} bits per bar, both panels"
        f"      ($\\lambda$ = {LAMBDA:g} on the micro block, right panel only)",
        ha="center",
        fontsize=6.1,
        color=fs.RULE,
    )

    fs.save(fig, OUT, "fig1_mechanism")


if __name__ == "__main__":
    main()
