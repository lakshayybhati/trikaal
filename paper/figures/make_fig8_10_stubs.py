"""Figures 8, 9 and 10 — the three post-run panels, built now against the artifact schema.

WHY ONE SCRIPT. The three share their placeholder machinery, their DRAFT watermark and their
schema binding; splitting them would triplicate that and let the three drift apart. Each still
emits its own PDF and its own caption slot.

EVERY NUMBER IN ALL THREE IS PLACEHOLDER, AND EVERY PLACEHOLDER IS IMPLAUSIBLE BY CONSTRUCTION.
The pre-registered run has not been executed. These panels exist so that the axes, the orderings,
the thresholds, the captions and the LaTeX floats are final before any result is seen.

An earlier draft of Figures 9 and 10 drew plausible positive information ratios with the
microstructure arm above the OHLCV arm above the placebo -- that is, it manufactured a
realistic-looking FAVOURABLE result for a run nobody has performed. That defect is the origin of
the project's CONTEXT-STRIPPING RULE (CLAUDE.md standing rules; BUILD_RECORD.md section 5): an
artifact must remain safe when its context is stripped, because a reader who crops or screenshots a
panel loses the watermark and keeps the numbers.

Applying that rule here: placeholder data must be impossible to mistake for a result even with the
watermark removed. Concretely --
  Fig 8   no values at all -- empty outcome cells
  Fig 9   ONE curve, with all three cells coincident on it, at magnitudes near +-45 annualized IR,
          which is roughly two orders of magnitude above anything this design could produce. No
          cell is ordered above another and no crossing point can be read off.
  Fig 10  all five cells centred on zero with identical dispersion, at the same absurd magnitude.
          No cell is favoured, and the claimed-effect bracket is drawn symmetric about zero.

NO --real FLAG EXISTS, and none will: under the pre-registration's stopping rule no verdict
manifest is ever emitted, so a swap-in-the-artifact path has no input that can exist. This
docstring previously described one, and Appendix F.5 repeated the description.

THRESHOLDS AND STRUCTURE BELOW ARE REAL — they are pins, not results, and are loaded at render
time from the committed decision surface:
    src/trikaal/eval/verdict.py       SURVIVES / NULL / HALT_ADJUDICATE, the five clause keys
                                      1_paired_ci, 2_mde_paired, 3_placebo_validity,
                                      4_economic_floor, 5_dsr; ECON_FLOOR_IR
    src/trikaal/eval/conformance.py   PINNED_HEADLINE_COST, PINNED_SEEDS, the kappa grid
    runs_manifest/m6_mde_inputs.json  the tabled MDE that clause 2 is compared against

SCHEMA THE REAL SWAP READS (the verdict manifest emitted by scripts/m6_verdict.py):
    .verdict.emitted                          -> the outcome word           (Fig 8)
    .clauses.<key>.{rule, passed, value}      -> the five clause rows       (Fig 8)
    .guards.{degeneracy, power}.{armed, halted} -> the two guard lamps      (Fig 8)
    .cost_stress.<cost_bps>.<cell>.net_ir     -> the curves                 (Fig 9)
    .per_cell.<cell>.per_seed.<seed>.net_ir   -> the seed scatter           (Fig 10)
    .guards.power.{claimed_delta, worst_range} -> the power-guard comparison (Fig 10)

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig8_10_stubs.py
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
sys.path.insert(0, str(ROOT / "src"))
from trikaal.eval.verdict import ECON_FLOOR_IR

fs.apply()

MDE_H15 = json.loads((RM / "m6_mde_inputs.json").read_text())["h15_pooled"]["MDE_annualized_IR"]
HEADLINE_COST = 0.0030
KAPPAS = (1.0, 1.5, 2.0, 3.0)
SEEDS = (0, 1, 2, 3, 4)

CLAUSES = [
    ("1_paired_ci", "one-sided 95% CI lower bound of $\\Delta$IR(4$-$5) $>$ 0"),
    ("2_mde_paired", f"$\\Delta$IR(4$-$5) $\\geq$ paired MDE   (tabled {MDE_H15:.3f} at $h$=15)"),
    ("3_placebo_validity", "CI lower bound of IR(4)$-$IR(2) $>$ 0   (placebo-independent)"),
    ("4_economic_floor", f"$\\Delta$IR(4$-$5) $\\geq$ {ECON_FLOOR_IR} annualized IR"),
    ("5_dsr", "deflated Sharpe ratio $\\geq$ 0.95 over 60 enumerated trials"),
]

RNG = np.random.default_rng(20260809)


def watermark(ax, *, rot=14, size=15, y=0.5, x=0.5):
    ax.text(
        x,
        y,
        "PLACEHOLDER — awaiting the run",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=size,
        weight="bold",
        color=fs.FAIL,
        alpha=0.30,
        rotation=rot,
        zorder=20,
        gid="overlay",  # exempt from fs.assert_text_legible: this is MEANT to lie across the panel
    )


# =============================================================================================
def fig8_verdict() -> None:
    """The decision surface: five clauses, two guards, one emitted word."""
    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.30))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.0,
        1.02,
        "the pre-registered decision surface",
        ha="left",
        va="bottom",
        fontsize=7.4,
        weight="bold",
    )
    ax.text(
        1.0,
        1.02,
        "conjunctive — SURVIVES only if none fails",
        ha="right",
        va="bottom",
        fontsize=6.2,
        color=fs.RULE,
    )

    for i, (key, rule) in enumerate(CLAUSES):
        y = 0.855 - i * 0.132
        ax.add_patch(
            Rectangle(
                (0.0, y - 0.048),
                1.0,
                0.104,
                fc=fs.GRID,
                alpha=0.35 if i % 2 == 0 else 0.15,
                ec="none",
            )
        )
        ax.text(
            0.012,
            y,
            key.replace("_", " "),
            ha="left",
            va="center",
            fontsize=6.3,
            weight="bold",
            color=fs.INK,
        )
        ax.text(0.225, y, rule, ha="left", va="center", fontsize=6.0, color=fs.RULE)
        ax.add_patch(
            Rectangle(
                (0.83, y - 0.035), 0.155, 0.070, fc="white", ec="#c8c8c8", lw=0.6, ls=(0, (2, 1.5))
            )
        )
        ax.text(0.9075, y, "—", ha="center", va="center", fontsize=7.0, color="#b0b0b0")

    ax.text(0.9075, 0.945, "outcome", ha="center", va="center", fontsize=5.9, color=fs.RULE)
    ax.add_patch(Rectangle((0.0, 0.012), 0.47, 0.125, fc=fs.GRID, alpha=0.30, ec="none"))
    ax.text(
        0.235,
        0.100,
        "HALT-only guards — never flip SURVIVES$\\leftrightarrow$NULL",
        ha="center",
        va="center",
        fontsize=5.8,
        color=fs.RULE,
    )
    ax.text(
        0.235,
        0.045,
        "degeneracy  —          power  —",
        ha="center",
        va="center",
        fontsize=6.3,
        color="#b0b0b0",
    )
    ax.add_patch(
        Rectangle((0.53, 0.008), 0.47, 0.150, fc="white", ec="#c8c8c8", lw=0.7, ls=(0, (2, 1.5)))
    )
    # Two lines, not one: at 5.5 pt the single-line form overran the dashed box's right edge
    # (caught by looking at the render, not by any check). The box is 0.47 wide in axes units.
    ax.text(
        0.765,
        0.112,
        "SURVIVES / NULL /\nINCONCLUSIVE / HALT_ADJUDICATE",
        ha="center",
        va="center",
        fontsize=5.2,
        linespacing=1.35,
        color=fs.RULE,
    )
    ax.text(
        0.765,
        0.038,
        "emitted  —",
        ha="center",
        va="center",
        fontsize=7.0,
        weight="bold",
        color="#b0b0b0",
    )
    watermark(ax, y=0.55, size=13)
    fs.save(fig, OUT, "fig8_verdict")


# =============================================================================================
BREAK_EVEN_LO, BREAK_EVEN_HI = 0.0023236455995917574, 0.02082459364067311


def fig9_cost_stress() -> None:
    """The specified cost-stress apparatus, with NO curve — the ablation it was for never ran."""
    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.35))
    # NO CURVE IS DRAWN. This panel previously carried one placeholder line whose zero crossing
    # landed EXACTLY on the pinned 0.30% rule -- the single most quotable point on the axis --
    # so a crop read "break-even is about 0.30%" against a MEASURED break-even of 0.0023-0.0208%,
    # wrong by 14-130x. Its defence in the caption was also false: it claimed a magnitude "two
    # orders above anything this design could produce", and the design produced net IRs of -28.5
    # to -146.5, so +-45 sat INSIDE the measured range. Figure 8's solution is used instead:
    # draw the specified apparatus and no data at all. There is then no crossing to quote.
    #
    # What IS drawn is the one break-even we have actually measured, so that the quantity a
    # reader tries to read off this panel is true rather than absent -- with its cell-1-only
    # scope written into the raster, because a crop removes the caption.
    ax.text(
        0.31,
        -0.62,
        "no curve is drawn — the ablation this panel was specified for never ran",
        ha="center",
        va="center",
        fontsize=6.3,
        color=fs.RULE,
        style="italic",
        zorder=6,
    )
    ax.axvspan(BREAK_EVEN_LO, BREAK_EVEN_HI, color=fs.FAIL, alpha=0.9, zorder=5)
    ax.annotate(
        f"MEASURED break-even\n{BREAK_EVEN_LO:.4f}–{BREAK_EVEN_HI:.4f}%\n"
        "cell 1 only, 3 units\nNOT an ablation result",
        xy=(BREAK_EVEN_HI, 0.72),
        xytext=(0.088, 0.72),
        textcoords="data",
        fontsize=5.9,
        color=fs.FAIL,
        linespacing=1.4,
        va="center",
        arrowprops={"arrowstyle": "-", "lw": 0.7, "color": fs.FAIL},
        zorder=6,
    )

    ax.axhline(0.0, color=fs.INK, lw=0.7, zorder=2)
    ax.axhline(ECON_FLOOR_IR, color=fs.RULE, ls=(0, (4, 3)), lw=0.8, zorder=2)
    ax.text(
        0.605,
        ECON_FLOOR_IR - 0.11,
        f"economic floor {ECON_FLOOR_IR}",
        ha="right",
        va="center",
        fontsize=6.1,
        color=fs.RULE,
        zorder=6,
    )
    ax.axvline(HEADLINE_COST * 100, color=fs.FAIL, ls=(0, (4, 2.5)), lw=1.0, zorder=4)
    ax.text(
        HEADLINE_COST * 100 + 0.012,
        -0.20,
        f"pinned headline\n{HEADLINE_COST * 100:.2f}% round trip",
        ha="left",
        va="center",
        fontsize=6.1,
        color=fs.FAIL,
        linespacing=1.35,
        zorder=6,
    )
    ax.axvspan(0.10, 0.30, color=fs.GRID, alpha=0.55, zorder=0)
    ax.text(
        0.20,
        -0.20,
        "pre-registered\ncost band",
        ha="center",
        va="center",
        fontsize=6.0,
        color=fs.RULE,
        linespacing=1.3,
        zorder=6,
    )

    ax.set_xlabel("round-trip transaction cost (%)")
    ax.set_ylabel("cost-aware net IR, annualized")
    ax.set_title(
        "how far the result survives a worse cost assumption", loc="left", fontsize=7.0, pad=5
    )
    ax.set_xlim(0, 0.62)
    ax.set_ylim(-0.85, 1.15)
    ax.set_yticks([ECON_FLOOR_IR, 0.0])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    watermark(ax, y=0.42)
    fs.assert_text_legible(fig, (ax,))
    fs.save(fig, OUT, "fig9_cost_stress")


# =============================================================================================
def fig10_seed_stability() -> None:
    """Per-seed dispersion against the effect being claimed — the power guard's own comparison."""
    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.35))
    cells = [1, 2, 3, 4, 5]
    colours = {1: fs.BSQ, 2: fs.OHLCV, 3: fs.BSQ, 4: fs.MICRO, 5: fs.PLACEBO}

    # Every cell centred on ZERO with identical dispersion, at a magnitude this design could not
    # produce. No cell is favoured; no ordering exists to be misread as a result.
    ABSURD = 40.0
    for i, c in enumerate(cells):
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]) * ABSURD
        ax.plot([i] * len(vals), vals, "o", ms=3.6, mfc="white", mec=colours[c], mew=0.9, zorder=3)
        ax.plot([i], [0.0], "D", ms=4.2, mfc=colours[c], mec="white", mew=0.5, zorder=4)
        ax.plot([i - 0.16, i + 0.16], [vals.min()] * 2, lw=0.8, color=colours[c], zorder=3)
        ax.plot([i - 0.16, i + 0.16], [vals.max()] * 2, lw=0.8, color=colours[c], zorder=3)
        ax.plot([i, i], [vals.min(), vals.max()], lw=0.6, color=colours[c], alpha=0.5, zorder=2)
    ax.axhline(0.0, color=fs.INK, lw=0.7, zorder=1)

    ax.annotate(
        "",
        xy=(4.95, ABSURD * 0.55),
        xytext=(4.95, -ABSURD * 0.55),
        arrowprops=dict(arrowstyle="|-|,widthA=0.35,widthB=0.35", lw=0.8, color=fs.FAIL),
    )
    ax.text(
        5.05,
        ABSURD * 0.20,
        "claimed $\\Delta$IR(4$-$5)",
        ha="left",
        va="center",
        fontsize=6.2,
        color=fs.FAIL,
    )
    ax.text(
        5.05,
        -ABSURD * 0.12,
        "the guard halts if any\ncell's across-seed\nrange reaches this",
        ha="left",
        va="top",
        fontsize=5.9,
        color=fs.RULE,
        linespacing=1.35,
    )
    ax.text(
        -0.42,
        ABSURD * 1.02,
        "placeholder: every cell centred on zero, magnitudes deliberately absurd",
        ha="left",
        va="bottom",
        fontsize=5.8,
        color=fs.FAIL,
    )

    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(
        [f"cell {c}" + ("\n(placebo)" if c == 5 else "") for c in cells], fontsize=6.3
    )
    ax.set_ylabel("cost-aware net IR, annualized")
    ax.set_title(
        f"per-seed dispersion against the effect claimed  ({len(SEEDS)} seeds, "
        "seeds are replicates)",
        loc="left",
        fontsize=7.0,
        pad=5,
    )
    ax.set_xlim(-0.5, 6.6)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    watermark(ax, y=0.26, x=0.34)
    fs.save(fig, OUT, "fig10_seed_stability")


if __name__ == "__main__":
    fig8_verdict()
    fig9_cost_stress()
    fig10_seed_stability()
