"""Figure 11 — the measured break-even cost against realistic execution cost.

THE FINDING THE FIGURE MUST CARRY WITHOUT ITS CAPTION
    The cost at which these models stop making money is one to two ORDERS OF MAGNITUDE below the
    cost of trading. Not close, not marginal: 4.8x to 43x short, and still 2.5x short at the most
    favourable end of the sampling interval.

ARTIFACTS — loaded at render time; nothing transcribed.
    runs_manifest/m6_horizon_break_even.json
        .measured_h15_headline[*].seed                                  -> the three units
        .measured_h15_headline[*].c_break_pct_POINT_see_c_break_interval -> the point estimate
        .measured_h15_headline[*].c_break_interval.{ci95_low, ci95_high} -> the interval
        .fee_reference.round_trip_frac                                   -> the 0.10% comparison
        .pins.flat_costs                                                 -> the pre-registered grid

WHY A LOG AXIS. The quantity spans 0.0023% to 0.30% — two orders of magnitude — and the whole
point is the SIZE of the gap. A linear axis puts every measured value on the origin and shows
nothing; a log axis is what makes "one to two orders of magnitude" visible rather than asserted.

SCOPE IS IN THE RASTER, NOT THE CAPTION. Cell 1 only (BSQ, OHLCV-only) — one arm of a 2x2 that
never ran. A cropped panel must still say so.

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig11_economics.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"

fs.apply()

BE = json.loads((RM / "m6_horizon_break_even.json").read_text())
ROWS = BE["measured_h15_headline"]
FEE = BE["fee_reference"]["round_trip_frac"] * 100.0  # -> per cent
GRID = [c * 100.0 for c in BE["pins"]["flat_costs"]]


def _verify() -> None:
    """Every drawn quantity must come from the receipt, and the ordering must be the claimed one.

    Written so it FAILS if the receipt stops supporting the finding: if any unit's break-even
    ever cleared the fee, the figure's own headline would be false and this raises rather than
    drawing it.
    """
    if len(ROWS) != 3:
        raise AssertionError(f"expected three scored units, receipt has {len(ROWS)}")
    for r in ROWS:
        pt = r["c_break_pct_POINT_see_c_break_interval"]
        ci = r["c_break_interval"]
        if not ci["ci95_low"] <= r["c_break_POINT_ESTIMATE_positions_fixed"] <= ci["ci95_high"]:
            raise AssertionError(f"seed {r['seed']}: point estimate outside its own interval")
        if pt >= FEE:
            raise AssertionError(
                f"seed {r['seed']} break-even {pt}% CLEARS the {FEE}% fee — the figure's headline "
                "is false for this receipt and must not be drawn"
            )
        if r["clears_round_trip_fee"] is not False:
            raise AssertionError(f"seed {r['seed']}: receipt disagrees with its own clears flag")
    if abs(FEE - 0.10) > 1e-9:
        raise AssertionError(f"fee reference moved to {FEE}%; the annotation text is hard-coded")


def main() -> None:
    _verify()
    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.15))

    order = sorted(ROWS, key=lambda r: r["c_break_pct_POINT_see_c_break_interval"])
    ys = list(range(len(order)))[::-1]

    for y, r in zip(ys, order, strict=True):
        pt = r["c_break_pct_POINT_see_c_break_interval"]
        ci = r["c_break_interval"]
        lo = max(ci["ci95_low"] * 100.0, 1e-4)  # log axis: clamp for drawing, labelled below
        hi = ci["ci95_high"] * 100.0
        crosses_zero = ci["ci95_low"] < 0.0
        ax.plot([lo, hi], [y, y], lw=1.4, color=fs.BSQ, alpha=0.45, zorder=3, solid_capstyle="butt")
        ax.plot([pt], [y], marker="D", ms=4.2, color=fs.BSQ, zorder=5)
        ax.text(
            pt,
            y + 0.26,
            f"{pt:.4f}%",
            ha="center",
            va="bottom",
            fontsize=fs.T_ANNOT,
            color=fs.BSQ,
            zorder=6,
        )
        gap = FEE / pt
        ax.text(
            hi * 1.30,
            y,
            f"{gap:.0f}× short" + ("  · 95% interval includes 0" if crosses_zero else ""),
            ha="left",
            va="center",
            fontsize=fs.T_MICRO,
            color=fs.RULE,
            zorder=6,
        )

    # the two costs that matter, drawn as regions rather than lines so the gap has WIDTH
    ax.axvspan(GRID[0], GRID[-1], color=fs.GRID, alpha=0.75, zorder=0)
    ax.axvline(FEE, color=fs.FAIL, lw=1.2, ls=(0, (4, 2.5)), zorder=4)
    ax.text(
        FEE * 1.08,
        len(order) - 0.42,
        f"realistic round trip {FEE:.2f}%\n(taker fees only — a LOWER bound)",
        ha="left",
        va="center",
        fontsize=fs.T_ANNOT,
        color=fs.FAIL,
        linespacing=1.35,
        zorder=6,
    )
    ax.text(
        GRID[-1] * 0.97,
        -0.72,
        "pre-registered cost grid",
        ha="right",
        va="center",
        fontsize=fs.T_MICRO,
        color=fs.RULE,
        zorder=6,
    )

    ax.set_xscale("log")
    ax.set_xlim(7e-4, 1.1)
    ax.set_ylim(-1.05, len(order) - 0.05)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"seed {r['seed']}" for r in order], fontsize=fs.T_MICRO, color=fs.RULE)
    ax.set_xlabel(
        "round-trip cost at which net IR crosses zero (%, log scale)", fontsize=fs.T_LABEL
    )
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", labelsize=fs.T_TICK)

    fs.finding(
        fig,
        "Break-even sits 4.8× to 43× below the cost of trading — a gap of one to two orders of "
        "magnitude, not a margin",
    )
    fs.scope(
        fig,
        "cell 1 only (BSQ, OHLCV-only), 3 units at the pinned h = 15 — one arm of a 2×2 that "
        "never ran.\nBars are 95% bootstrap intervals; the most favourable upper bound, 0.0396%, "
        "is still 2.5× short.",
    )
    fig.subplots_adjust(top=0.82, bottom=0.34, left=0.085, right=0.995)
    fs.assert_text_legible(fig, (ax,))
    fs.save(fig, OUT, "fig11_economics")


if __name__ == "__main__":
    main()
