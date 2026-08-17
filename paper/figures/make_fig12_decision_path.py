"""Figure 12 — what actually happened: the pre-registered path from design to primary result.

THE FINDING THE FIGURE MUST CARRY WITHOUT ITS CAPTION
    The study did not fail and then get reinterpreted. A gate written before the data stopped it,
    and the disposition that followed -- mechanism finding becomes primary, ablation reports
    nothing -- was itself pre-registered, months of prose reduced to one path a reader can follow.

WHY IT EXISTS. A reader currently assembles this from prose spread across §4, §6 and §7: the
five-cell design, the gate that stands in front of second-stage training, the firing, item E, and
the resulting swap of primary and secondary. That reconstruction is work, and a reader who does
not do it comes away thinking the ablation ran and returned nothing.

ARTIFACTS — every quantity on the panel is loaded, not transcribed. The BOXES are schematic; the
NUMBERS in them are not.
    src/trikaal/eval/conformance.py   PINNED_MICRO_LEGIBILITY_MIN, PINNED_SEEDS -> the gate rule
    src/trikaal/train/gates.py        MICRO_DIMS                                -> six dimensions
    runs_manifest/m6_micro_legibility_stop.json
        .stage2_entered, .artifacts_produced                    -> what the firing did
        .legibility_receipt.<arm>.per_dim.<d>.sign_acc          -> worst dim per gated arm

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig12_decision_path.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT / "src"))
from trikaal.eval.conformance import PINNED_MICRO_LEGIBILITY_MIN, PINNED_SEEDS
from trikaal.train.gates import MICRO_DIMS

fs.apply()

STOP = json.loads((RM / "m6_micro_legibility_stop.json").read_text())
LR = STOP["legibility_receipt"]
GATED = ("cell4_fsq_micro_seed0", "cell5_fsq_micro_shuffled_seed0")
WORST = {a: min(v["sign_acc"] for v in LR[a]["per_dim"].values()) for a in GATED}


def _verify() -> None:
    """The panel asserts a STOP. If the receipt ever stops recording one, do not draw it."""
    if STOP.get("stage2_entered") is not False:
        raise AssertionError("receipt does not record stage2_entered = false")
    if STOP.get("artifacts_produced") != 0:
        raise AssertionError(
            f"receipt records {STOP.get('artifacts_produced')} artifacts; the panel says none"
        )
    for arm in GATED:
        if WORST[arm] >= PINNED_MICRO_LEGIBILITY_MIN:
            raise AssertionError(
                f"{arm} worst dim {WORST[arm]} CLEARS the {PINNED_MICRO_LEGIBILITY_MIN} gate — "
                "this figure claims both arms were refused and must not be drawn"
            )
    if len(MICRO_DIMS) != 6:
        raise AssertionError(f"gate covers {len(MICRO_DIMS)} dims; the panel says six")


def _box(ax, x, y, w, h, text, *, fc, ec, lw=0.8, size=None, weight="normal", tc=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.012",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size or fs.T_ANNOT,
        color=tc or fs.INK,
        weight=weight,
        linespacing=1.45,
        zorder=5,
    )


def _arrow(ax, p0, p1, *, color=None, lw=0.9, style="-|>"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle=style,
            mutation_scale=7,
            lw=lw,
            color=color or fs.RULE,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=4,
        )
    )


def main() -> None:
    _verify()
    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.62))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    pale_ok = "#EAF2EA"
    pale_no = "#FBECE9"

    # ---- row 1: what was pre-registered ------------------------------------------------------
    ax.text(
        0.0,
        0.965,
        "PRE-REGISTERED, BEFORE ANY DATA",
        fontsize=fs.T_MICRO,
        color=fs.RULE,
        weight="bold",
        ha="left",
        va="center",
    )
    _box(
        ax,
        0.00,
        0.74,
        0.30,
        0.175,
        "five-cell ablation\n{BSQ, FSQ} × {OHLCV, +micro}\n"
        f"+ shuffled placebo · {len(PINNED_SEEDS)} seeds",
        fc="white",
        ec=fs.RULE,
    )
    _box(
        ax,
        0.355,
        0.74,
        0.28,
        0.175,
        f"legibility gate\n{PINNED_MICRO_LEGIBILITY_MIN:.2f} on EVERY one of\n"
        f"{len(MICRO_DIMS)} live micro dims",
        fc="white",
        ec=fs.FAIL,
        lw=1.1,
    )
    _box(
        ax,
        0.69,
        0.74,
        0.31,
        0.175,
        "item E, written in advance\nif the gate fires: STOP,\nmechanism becomes primary",
        fc="white",
        ec=fs.RULE,
    )
    _arrow(ax, (0.30, 0.828), (0.355, 0.828))
    _arrow(ax, (0.635, 0.828), (0.69, 0.828))

    # ---- row 2: what the gate did ------------------------------------------------------------
    ax.text(
        0.0,
        0.615,
        "ON REAL DATA, 2026-08-12",
        fontsize=fs.T_MICRO,
        color=fs.RULE,
        weight="bold",
        ha="left",
        va="center",
    )
    _box(
        ax,
        0.00,
        0.40,
        0.30,
        0.165,
        "stage 1 trained\nboth microstructure arms",
        fc="white",
        ec=fs.RULE,
    )
    _box(
        ax,
        0.355,
        0.375,
        0.28,
        0.19,
        "THE GATE FIRED\nboth arms refused\n"
        f"worst dim {min(WORST.values()):.4f} / {max(WORST.values()):.4f}",
        fc=pale_no,
        ec=fs.FAIL,
        lw=1.2,
        weight="bold",
        tc=fs.FAIL,
    )
    _box(
        ax,
        0.69,
        0.40,
        0.31,
        0.165,
        f"stage 2 never entered\n{STOP['artifacts_produced']} evaluation artifacts",
        fc=pale_no,
        ec=fs.FAIL,
    )
    _arrow(ax, (0.30, 0.4825), (0.355, 0.4725))
    _arrow(ax, (0.635, 0.4725), (0.69, 0.4825), color=fs.FAIL)
    _arrow(ax, (0.495, 0.74), (0.495, 0.565), color=fs.FAIL, lw=1.1)

    # ---- row 3: the disposition that followed ------------------------------------------------
    ax.text(
        0.0,
        0.292,
        "WHAT THIS PAPER THEREFORE REPORTS",
        fontsize=fs.T_MICRO,
        color=fs.RULE,
        weight="bold",
        ha="left",
        va="center",
    )
    _box(
        ax,
        0.00,
        0.055,
        0.485,
        0.195,
        "PRIMARY — the mechanism finding\nthe tokenizer keeps the microstructure that\n"
        "duplicates price and loses the rest",
        fc=pale_ok,
        ec=fs.PASS,
        lw=1.1,
        weight="bold",
    )
    _box(
        ax,
        0.545,
        0.055,
        0.455,
        0.195,
        "NOT REPORTED — the ablation\nno verdict, no null, no cost-aware net IR;\n"
        "an experiment its entry condition stopped",
        fc="white",
        ec=fs.RULE,
    )
    # the primary comes from the GATE FIRING, not from stage-1 training — the first draft drew it
    # from the wrong box, which is a claim about causation, not a layout choice
    _arrow(ax, (0.845, 0.40), (0.80, 0.25))
    _arrow(ax, (0.40, 0.375), (0.30, 0.25), color=fs.PASS, lw=1.1)

    fs.finding(
        fig,
        "A gate written before the data stopped the study — and the disposition that followed was "
        "pre-registered too",
    )
    fs.scope(
        fig,
        "Boxes are schematic; every number on them is loaded from the pinned constants and the "
        "gate's own stop receipt. The two refused arms are\nthe microstructure cells; three "
        "ungated "
        "BSQ units were scored, and what they support is confined to the limitations.",
    )
    fig.subplots_adjust(top=0.90, bottom=0.115, left=0.005, right=0.995)
    fs.assert_text_legible(fig, (ax,))
    fs.save(fig, OUT, "fig12_decision_path")


if __name__ == "__main__":
    main()
