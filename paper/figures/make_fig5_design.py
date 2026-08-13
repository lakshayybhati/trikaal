"""Figure 5 — the five-cell design at matched capacity, and the placebo's reach.

VERDICT THE FIGURE MUST CARRY WITHOUT ITS CAPTION
    The design is a 2x2 over quantizer and input arm, plus a fifth cell that is cell 4 with the
    microstructure block shuffled. The two quantizer arms are matched to within 0.06 bits per bar
    and 0.33% of parameters, so capacity is controlled rather than assumed. The placebo destroys
    what it was built to destroy -- microstructure's link to price and its own autocorrelation --
    and leaves the within-block correlation standing, which is why the headline difference carries
    a capacity handicap that the design discloses rather than removes.

CONTEXT-STRIPPING RULE (CLAUDE.md standing rules; BUILD_RECORD.md section 5). Until 2026-08-10 the
    red annotation block read "Cell 6 ... is conditional on a pre-committed probe" while the LaTeX
    caption had already been corrected to "future work rather than a committed contingency". Cell 6
    appears nowhere in docs/m6_prereg.md -- verified by exhaustive case-insensitive search -- so the
    in-figure text asserted a commitment the pre-registration explicitly refuses. Cropped out of its
    caption, which is the state this figure is most likely to be read in, the panel made a claim the
    project has no basis for. The rule is that the ARTEFACT must be safe with its context stripped,
    so the graphic and the caption must land on the same statement, not merely be reconcilable.
    Standing constraint on this figure: the string "pre-committed" must not appear in the pixels.

ARTIFACTS — loaded at render time.
    src/trikaal/train/cells.py            the five CellSpec entries (the authoritative registry)
    src/trikaal/constants.py              FSQ_LEVELS, FSQ_V_C, FSQ_V_F, MICRO_DIMS_IDX
    runs_manifest/m6_interface_respec_design_pass.json .g_parity_new_layout.{bpt_a,bpt_b}
    docs/m6_c12_placebo_mechanism.md      the measured before/after correlation table
    src/trikaal/eval/verdict.py           ECON_FLOOR_IR (clause 4's materiality floor)

The parameter counts are the REALIZED PREDICTOR totals summed from the shipped checkpoints
    runs_cloud/ckpt_cell4_seed0/predictor.pt                  31,795,200  (FSQ)
    runs_cloud/rescue/r0/cell1_bsq_ohlcv_seed0/predictor.pt   31,725,568  (BSQ)
and are VERIFIED against those files at render time when they are present (see _verify_params).
The figures carried here previously — 21,301,248 / 21,231,616 — are the backbone EXCLUDING the
10,493,952 MTP parameters, identical in both arms; they are quoted as the named secondary. The
gap percentage is printed WITH ITS DENOMINATOR: a cropped panel must not carry a bare "0.327%".

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig5_design.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT / "src"))
from trikaal.constants import MICRO_DIMS_IDX

fs.apply()

PAR = json.loads((RM / "m6_interface_respec_design_pass.json").read_text())["g_parity_new_layout"]
BPT_FSQ, BPT_BSQ = PAR["bpt_a"], PAR["bpt_b"]
N_FSQ, N_BSQ = 31_795_200, 31_725_568  # realized predictor totals, summed from the checkpoints
N_MTP = 10_493_952  # identical in BOTH arms, so the absolute gap is basis-independent
PARAM_GAP = 100.0 * (N_FSQ - N_BSQ) / N_FSQ  # 0.219% OF THE MODEL (0.327% of the backbone alone)


def _verify_params() -> None:
    """Assert the hard-coded totals against the checkpoints when they are on disk.

    Skipped, not faked, when the weights are absent — a renderer without them still works. When
    they ARE present this must fail on a wrong constant, which is the only reason it exists.
    """
    ckpts = {
        "FSQ": (pathlib.Path("runs_cloud/ckpt_cell4_seed0/predictor.pt"), N_FSQ),
        "BSQ": (pathlib.Path("runs_cloud/rescue/r0/cell1_bsq_ohlcv_seed0/predictor.pt"), N_BSQ),
    }
    if not all(f.exists() for f, _ in ckpts.values()):
        return
    import torch

    for arm, (f, expect) in ckpts.items():
        blob = torch.load(f, map_location="cpu", weights_only=False)
        sd = blob["state_dict"] if "state_dict" in blob else blob  # checkpoints wrap the tensors
        total = sum(v.numel() for v in sd.values())
        mtp = sum(v.numel() for k, v in sd.items() if k.split(".")[0] == "mtp")
        assert total == expect, f"{arm} realized total {total:,} != {expect:,} ({f})"
        assert mtp == N_MTP, f"{arm} mtp {mtp:,} != {N_MTP:,} ({f})"


# the placebo's measured effect, read out of the mechanism note rather than retyped
_doc = (ROOT / "docs" / "m6_c12_placebo_mechanism.md").read_text()
PLACEBO_ROWS = []
for line in _doc.splitlines():
    m = re.match(
        r"\|\s*(mean.*?)\s*\|\s*\*?\*?([\d.]+)\*?\*?\s*\|\s*\*?\*?([\d.]+)\*?\*?\s*\|\s*\*?\*?(\w+)",
        line,
    )
    if m:
        PLACEBO_ROWS.append((m.group(1), float(m.group(2)), float(m.group(3)), m.group(4)))

CELLS = {(0, 0): (1, "BSQ"), (0, 1): (2, "FSQ"), (1, 0): (3, "BSQ"), (1, 1): (4, "FSQ")}


def main() -> None:
    _verify_params()  # must run BEFORE anything is drawn; a wrong constant is a wrong figure
    fig = plt.figure(figsize=(fs.SINGLE_COL, 3.05))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.02, 1.0], wspace=0.16, left=0.055, right=0.985, top=0.90, bottom=0.045
    )

    # ---------------------------------------------------------------- the grid
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 1.02, "the five cells", ha="center", va="bottom", fontsize=7.4, weight="bold")

    x0, w, gap = 0.305, 0.295, 0.040
    y0, h, vgap = 0.620, 0.170, 0.032
    for (r, c), (cid, quant) in CELLS.items():
        xx = x0 + c * (w + gap)
        yy = y0 - r * (h + vgap)
        col = fs.FSQ if quant == "FSQ" else fs.BSQ
        ax.add_patch(
            FancyBboxPatch(
                (xx, yy),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.014",
                fc=col,
                alpha=0.13,
                ec=col,
                lw=1.0,
                zorder=2,
            )
        )
        ax.text(
            xx + w / 2,
            yy + h * 0.63,
            f"cell {cid}",
            ha="center",
            va="center",
            fontsize=8.2,
            weight="bold",
            color=col,
            zorder=3,
        )
        bpt = BPT_FSQ if quant == "FSQ" else BPT_BSQ
        ax.text(
            xx + w / 2,
            yy + h * 0.24,
            f"{bpt:.3f} bits/bar",
            ha="center",
            va="center",
            fontsize=6.2,
            color=col,
            zorder=3,
        )

    # placebo cell, hanging off cell 4
    px, py = x0 + (w + gap), 0.215
    ax.add_patch(
        FancyBboxPatch(
            (px, py),
            w,
            h * 0.88,
            boxstyle="round,pad=0,rounding_size=0.014",
            fc=fs.PLACEBO,
            alpha=0.16,
            ec=fs.PLACEBO,
            lw=1.0,
            ls=(0, (3, 2)),
            zorder=2,
        )
    )
    ax.text(
        px + w / 2,
        py + h * 0.57,
        "cell 5",
        ha="center",
        va="center",
        fontsize=8.2,
        weight="bold",
        color="#5f5f5f",
        zorder=3,
    )
    ax.text(
        px + w / 2,
        py + h * 0.21,
        f"{BPT_FSQ:.3f} bits/bar, shuffled",
        ha="center",
        va="center",
        fontsize=6.0,
        color="#5f5f5f",
        zorder=3,
    )
    ax.annotate(
        "",
        xy=(px + w / 2, py + h * 0.90),
        xytext=(px + w / 2, y0 - vgap - 0.004),
        arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#8a8a8a"),
    )
    ax.text(
        px - 0.018,
        py + h * 0.44,
        "the placebo",
        ha="right",
        va="center",
        fontsize=6.2,
        color="#5f5f5f",
    )

    # axes labels
    ax.text(
        x0 + w / 2, y0 + h + 0.030, "BSQ", ha="center", fontsize=7.0, color=fs.BSQ, weight="bold"
    )
    ax.text(
        x0 + w + gap + w / 2,
        y0 + h + 0.030,
        "FSQ",
        ha="center",
        fontsize=7.0,
        color=fs.FSQ,
        weight="bold",
    )
    ax.text(x0 + w + gap / 2, y0 + h + 0.090, "quantizer", ha="center", fontsize=6.4, color=fs.RULE)
    ax.text(
        x0 - 0.022,
        y0 + h / 2,
        "OHLCV only\n7 dims",
        ha="right",
        va="center",
        fontsize=6.4,
        color=fs.OHLCV,
        linespacing=1.35,
    )
    ax.text(
        x0 - 0.022,
        y0 - vgap - h / 2,
        f"+ microstructure\ndims {min(MICRO_DIMS_IDX)}–{max(MICRO_DIMS_IDX)}",
        ha="right",
        va="center",
        fontsize=6.4,
        color=fs.MICRO,
        linespacing=1.35,
    )

    # matched-capacity strip
    ax.add_patch(Rectangle((0.0, 0.010), 1.0, 0.150, fc=fs.GRID, alpha=0.45, ec="none"))
    ax.text(
        0.5,
        0.126,
        "capacity is controlled, not assumed",
        ha="center",
        fontsize=6.5,
        weight="bold",
        color=fs.INK,
    )
    ax.text(
        0.5,
        0.078,
        f"{BPT_FSQ:.3f} vs {BPT_BSQ:.3f} bits per bar ({BPT_FSQ - BPT_BSQ:.3f} apart)",
        ha="center",
        fontsize=6.1,
        color=fs.RULE,
    )
    ax.text(
        0.5,
        0.036,
        f"{N_FSQ:,} vs {N_BSQ:,} realized params "
        f"({PARAM_GAP:.3f}% of the model apart; {N_MTP:,} of each are MTP heads)",
        ha="center",
        fontsize=6.1,
        color=fs.RULE,
    )

    # ---------------------------------------------------------------- what the placebo does
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.text(
        0.5,
        1.02,
        "what shuffling the block destroys — and what it leaves",
        ha="center",
        va="bottom",
        fontsize=7.4,
        weight="bold",
    )

    ys = [0.905, 0.762, 0.619, 0.476]
    for y, (name, before, after, verdict) in zip(ys, PLACEBO_ROWS[:4], strict=False):
        destroyed = verdict.lower().startswith("destroy")
        col = fs.FAIL if destroyed else fs.PASS
        label = name.replace("\\|corr\\|", "|corr|").replace("\\|", "|")
        ax2.text(0.0, y + 0.048, label, ha="left", va="center", fontsize=6.0, color=fs.INK)
        for x, v, tag in ((0.02, before, "before"), (0.40, after, "after")):
            ax2.add_patch(
                Rectangle(
                    (x, y - 0.028),
                    0.30 * min(v / 0.45, 1.0),
                    0.040,
                    fc=col if x > 0.2 else "#b9b9b9",
                    alpha=0.55,
                    ec="none",
                )
            )
            ax2.text(
                x, y - 0.056, f"{tag} {v:.4f}", ha="left", va="center", fontsize=5.8, color=fs.RULE
            )
        ax2.text(
            0.995,
            y - 0.006,
            "destroyed" if destroyed else "preserved",
            ha="right",
            va="center",
            fontsize=6.2,
            weight="bold",
            color=col,
        )

    ax2.add_patch(Rectangle((0.0, 0.010), 1.0, 0.310, fc=fs.FAIL, alpha=0.055, ec="none"))
    ax2.text(
        0.5,
        0.276,
        "the handicap this creates, disclosed not removed",
        ha="center",
        fontsize=6.5,
        weight="bold",
        color=fs.FAIL,
    )
    ax2.text(
        0.5,
        0.138,
        "cell 4's micro dims are partly free: they share structure the code already carries.\n"
        "Shuffling removes that sharing too, so $\\Delta$IR(4$-$5) is "
        "(information) + (capacity handicap),\nnot information alone. The handicap is NOT "
        "quantified in IR units. A sixth cell holding the\nblock constant would bracket it; "
        "no such cell is in the pre-registration, so it is future work.",
        ha="center",
        va="center",
        fontsize=5.9,
        color=fs.INK,
        linespacing=1.5,
    )

    fs.save(fig, OUT, "fig5_design")


if __name__ == "__main__":
    main()
