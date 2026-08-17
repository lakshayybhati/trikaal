"""Figure 1 — the instrument: per-bar features -> dual-path tokenizer -> fused-token AR backbone.

Structural values (vocabulary sizes, bit budgets, gate threshold, class weight) are read from
the repository's own constants and receipts rather than typed in, so the diagram cannot drift
from the code it depicts. Everything the caption can carry is left to the caption; the diagram
holds only what must be seen spatially.

    python3 paper/figures/make_fig1_architecture.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT / "src"))
from trikaal.constants import (
    FSQ_LEVELS,
    FSQ_V_C,
    FSQ_V_F,
    MICRO_DIMS_IDX,
    N_FEATURES,
    OHLCV_ONLY_IDX,
    derive_coarse_fine,
)
from trikaal.train.gates import MICRO_LEGIBILITY_MIN

fs.apply()

COARSE_IDX, FINE_IDX = derive_coarse_fine(FSQ_LEVELS)
BITS_C = sum(math.log2(FSQ_LEVELS[i]) for i in COARSE_IDX)
BITS_F = sum(math.log2(FSQ_LEVELS[i]) for i in FINE_IDX)
BPT = BITS_C + BITS_F

with open(ROOT / "runs_manifest" / "m6_lambda_search_receipt.json") as fh:
    LAMBDA = json.load(fh)["pin"]["PINNED_MICRO_POINT_WEIGHT"]

# §7 v1.6.34: the HEADLINE figure is the REALIZED PREDICTOR TOTAL. The backbone-excluding-MTP
# count is the named secondary. This generator carried 21.3M through the whole correction pass
# because that sweep covered the .tex files and one other generator, not this one.
N_PARAMS = 31_795_200  # realized predictor total at the FSQ vocabulary (checkpoint-verified)
N_BACKBONE = 21_301_248  # backbone EXCLUDING the 10,493,952 MTP heads

# ---- layout ------------------------------------------------------------------------------
FIG_W, FIG_H = fs.SINGLE_COL, 1.95
Y_HI, Y_LO, BH = 0.760, 0.480, 0.235
Y_MID = (Y_HI + Y_LO) / 2
CAP = 0.318  # every hanging caption starts here
COL = {  # (x_left, width);  "gate" is (x_centre, half-diagonal)
    "in": (0.000, 0.135),
    "enc": (0.169, 0.180),
    "tok": (0.383, 0.178),
    "fuse": (0.595, 0.078),
    "gate": (0.759, 0.052),
    "bb": (0.845, 0.155),
}


def box(
    ax,
    x,
    y,
    w,
    h,
    label,
    *,
    fc="white",
    ec=None,
    lw=0.7,
    size=7.0,
    weight="normal",
    tc=None,
    ls="-",
):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.012",
            fc=fc,
            ec=ec or fs.INK,
            lw=lw,
            ls=ls,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=size,
        weight=weight,
        color=tc or fs.INK,
        zorder=3,
        linespacing=1.45,
    )


def cbox(ax, col, ycen, h, label, **kw):
    x, w = COL[col]
    box(ax, x, ycen - h / 2, w, h, label, **kw)


def arrow(ax, x0, y0, x1, y1, *, color=None, lw=0.7, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle=style,
            lw=lw,
            color=color or fs.INK,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            mutation_scale=7,
            zorder=2,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- headers ----------------------------------------------------------------------
    ax.text(0.0675, 0.975, "per-bar features", ha="center", fontsize=7.4, weight="bold")
    ax.text(
        0.0675,
        0.908,
        f"{N_FEATURES}-wide, {N_FEATURES - 3} live",
        ha="center",
        fontsize=6.6,
        color=fs.GREY,
    )
    ax.text(
        0.472,
        0.955,
        "FSQ    (BSQ arm: 10 + 10 bits)",
        ha="center",
        fontsize=6.9,
        color=fs.GREY,
        style="italic",
    )

    # ---- input stack ------------------------------------------------------------------
    x0, w0 = COL["in"]
    for ycen, hh, lab, fc, ec, tc, ls in [
        (
            Y_HI,
            0.190,
            f"OHLC + volume\ndims 0–{max(OHLCV_ONLY_IDX)}",
            "white",
            fs.INK,
            fs.INK,
            "-",
        ),
        (
            Y_LO,
            0.235,
            f"microstructure\ndims {min(MICRO_DIMS_IDX)}–{max(MICRO_DIMS_IDX)}\nTFI, trade flow",
            "#f4f4e8",
            fs.OLIVE,
            fs.INK,
            "-",
        ),
        (0.235, 0.155, "funding, OI\nmasked, 100%", "#f5f5f5", "#aaaaaa", "#8a8a8a", (0, (2, 1.6))),
    ]:
        box(ax, x0, ycen - hh / 2, w0, hh, lab, fc=fc, ec=ec, tc=tc, ls=ls, size=5.9)

    # ---- encoders ---------------------------------------------------------------------
    arrow(ax, x0 + w0, Y_HI, COL["enc"][0], Y_HI, color=fs.GREY)
    arrow(ax, x0 + w0, Y_LO, COL["enc"][0], Y_LO, color=fs.GREY)
    cbox(ax, "enc", Y_HI, BH, "contextual\nencoder\ncausal, bars $\\leq t$")
    cbox(ax, "enc", Y_LO, BH, "pointwise\nencoder\nbar $t$ only", ec=fs.BLUE, lw=1.0)

    xe = COL["enc"][0] + COL["enc"][1]
    arrow(ax, xe, Y_HI, COL["tok"][0], Y_HI)
    arrow(ax, xe, Y_LO, COL["tok"][0], Y_LO, color=fs.BLUE)

    # ---- subtokens --------------------------------------------------------------------
    cbox(ax, "tok", Y_HI, BH, f"coarse subtoken\n{FSQ_V_C:,} codes\n{BITS_C:.2f} bits")
    cbox(
        ax,
        "tok",
        Y_LO,
        BH,
        f"fine subtoken\n{FSQ_V_F:,} codes\n{BITS_F:.2f} bits",
        ec=fs.BLUE,
        lw=1.0,
    )

    xt, wt = COL["tok"]
    box(
        ax,
        xt,
        0.075,
        wt,
        0.215,
        f"per-bar bottleneck leg\nmicro dims $\\times\\,\\lambda\\!=\\!{LAMBDA:g}$\n"
        "train-time only",
        ec=fs.OLIVE,
        ls=(0, (2.5, 1.5)),
        fc="#fbfbf0",
        size=6.4,
    )
    arrow(
        ax,
        xt + wt / 2,
        Y_LO - BH / 2 - 0.004,
        xt + wt / 2,
        0.294,
        color=fs.OLIVE,
        ls=(0, (2.2, 1.4)),
        style="-|>",
        lw=0.6,
    )

    # ---- fusion -----------------------------------------------------------------------
    xf, wf = COL["fuse"]
    arrow(ax, xt + wt, Y_HI, xf, Y_MID + 0.055)
    arrow(ax, xt + wt, Y_LO, xf, Y_MID - 0.055, color=fs.BLUE)
    box(
        ax,
        xf,
        Y_MID - 0.145,
        wf,
        0.290,
        f"fused\ntoken\n{BPT:.2f}\nbits/bar",
        weight="bold",
        size=6.6,
    )

    # ---- gate -------------------------------------------------------------------------
    gx, gh = COL["gate"]
    arrow(ax, xf + wf, Y_MID, gx - gh, Y_MID)
    ax.add_patch(
        Polygon(
            [(gx, Y_MID + 0.150), (gx + gh, Y_MID), (gx, Y_MID - 0.150), (gx - gh, Y_MID)],
            closed=True,
            fc="#fbeeeb",
            ec=fs.RED,
            lw=1.0,
            zorder=2,
        )
    )
    ax.text(
        gx,
        Y_MID,
        "legibility\ngate",
        ha="center",
        va="center",
        fontsize=6.5,
        color=fs.RED,
        weight="bold",
        zorder=3,
        linespacing=1.3,
    )
    ax.plot([gx, gx], [Y_MID - 0.155, CAP + 0.014], lw=0.5, color=fs.RED, zorder=1)
    ax.text(
        gx,
        CAP,
        f"six dims $\\geq$ {MICRO_LEGIBILITY_MIN:.2f}\nhard stop",
        ha="center",
        va="top",
        fontsize=6.4,
        color=fs.RED,
        linespacing=1.45,
    )

    # ---- backbone ---------------------------------------------------------------------
    xb, wb = COL["bb"]
    arrow(ax, gx + gh, Y_MID, xb, Y_MID)
    box(
        ax,
        xb,
        Y_MID - 0.145,
        wb,
        0.290,
        f"AR backbone + MTP\n{N_PARAMS / 1e6:.1f}M params\n"
        f"({N_BACKBONE / 1e6:.1f}M excl. MTP)\n8 layers · $d$512",
        size=6.6,
    )
    ax.text(
        xb + wb / 2,
        CAP,
        "held fixed across\nall five cells",
        ha="center",
        va="top",
        fontsize=6.4,
        color=fs.GREY,
        linespacing=1.45,
    )

    fs.save(fig, OUT, "fig1_architecture")


if __name__ == "__main__":
    main()
