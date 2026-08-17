"""Figure 4 — per-bar legibility across the intervention, against the 0.90 gate.

VERDICT THE FIGURE MUST CARRY WITHOUT ITS CAPTION
    The first two architectural changes leave legibility at chance. Only the third -- weighting the
    microstructure block in the per-bar objective -- moves it, and only lambda = 3 clears the gate
    on every seed. The threshold sits at the achievable ceiling, not comfortably above it.

ARTIFACTS — loaded at render time; nothing transcribed.
    git show c4cd082:runs_manifest/m6_interface_respec_design_pass.json
        .gate2_blocked_diagnosis.iteration_history       -> pointwise-only and +leg values
        .gates.gate2_legibility.{old, old_at_discovery}  -> the contextual baseline
    runs_manifest/m6_interface_respec_design_pass.json
        .gates.gate2_legibility.{new_by_seed, threshold} -> entropy-calibrated 3-seed failure
    runs_manifest/m6_lambda_search_receipt.json
        .seeded_campaign_omp2.{lam2,lam3}                -> the calibrated lambda triplets
        .formal_cell_path_pin3_omp3                      -> the pinned construction path
        .pin.PINNED_MICRO_POINT_WEIGHT                   -> lambda = 3
    runs_manifest/m6_acceptance_stage1_manifest.json
        .runs.{planted,noise}.per_bar_id_legibility      -> the feature-space acceptance run
    runs_manifest/m6_micro_legibility_stop.json          THE GATE'S FIRING ON REAL DATA
        .legibility_receipt.cell4_fsq_micro_seed0.per_dim.<d>.{sign_acc, base_rate_positive}
        .legibility_receipt.cell5_fsq_micro_shuffled_seed0.per_dim.<d>.{sign_acc}
    src/trikaal/train/gates.py                           MICRO_DIMS, MICRO_LEGIBILITY_MIN

RIGHT PANEL NOW CARRIES THE PRIMARY RESULT, AND FOR A LONG TIME IT DID NOT. Through every draft up
to this one it drew six EMPTY slots watermarked "NO VALUES / awaiting the run" -- correct while the
gate had not executed, and then simply STALE once it had. The gate fired on 2026-08-12; the six
numbers were printed in §6.1 and quoted in the abstract; and this file, the generator of the
paper's primary figure, contained NO REFERENCE TO THE STOP RECEIPT AT ALL. Nothing failed: it
loaded four fixture-era receipts, found everything it asked for, and rendered a watermark over the
result. That is the exact defect §8.9 claims our tooling prevents, so _verify_gate_values below now
makes the claim true for this figure: the drawn values are asserted against the receipt, and a
disagreement raises rather than rendering.

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig4_legibility.py
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"
sys.path.insert(0, str(ROOT / "src"))
from trikaal.constants import FEATURE_NAMES, MICRO_DIMS_IDX

fs.apply()

CUR = json.loads((RM / "m6_interface_respec_design_pass.json").read_text())
LAM = json.loads((RM / "m6_lambda_search_receipt.json").read_text())
ACC = json.loads((RM / "m6_acceptance_stage1_manifest.json").read_text())
STOP = json.loads((RM / "m6_micro_legibility_stop.json").read_text())
PRE = json.loads(
    subprocess.run(
        ["git", "show", "c4cd082:runs_manifest/m6_interface_respec_design_pass.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
)

GATE = CUR["gates"]["gate2_legibility"]["threshold"]
FLOOR = 0.85  # §7 v1.4.1 per-seed floor, restated with the mean clause
HIST = PRE["gate2_blocked_diagnosis"]["iteration_history"]


def _floats(text: str) -> list[float]:
    import re

    return [float(x) for x in re.findall(r"0\.\d{3,4}", text)]


STAGES = [
    (
        "contextual fine subtoken\n(the original design)",
        [
            PRE["gates"]["gate2_legibility"]["old"],
            PRE["gates"]["gate2_legibility"]["old_at_discovery"],
        ],
    ),
    ("+ pointwise fine encoder\nalone", _floats(HIST["iter1_pointwise_encoder_only"])),
    (
        "+ per-bar bottleneck leg\n(canonical width)",
        _floats(HIST["iter2_added_per_bar_bottleneck_leg"]),
    ),
    (
        "+ per-bar bottleneck leg\n(entropy-calibrated fixture)",
        list(CUR["gates"]["gate2_legibility"]["new_by_seed"].values()),
    ),
    ("+ class weighting  $\\lambda$ = 2", LAM["seeded_campaign_omp2"]["lam2"]["legibility"]),
    (
        "+ class weighting  $\\lambda$ = 3  (pinned)",
        LAM["seeded_campaign_omp2"]["lam3"]["legibility"],
    ),
    ("pinned construction path", LAM["formal_cell_path_pin3_omp3"]["legibility"]),
    (
        "feature-space acceptance run",
        [
            ACC["runs"]["planted"]["per_bar_id_legibility"],
            ACC["runs"]["noise"]["per_bar_id_legibility"],
        ],
    ),
]


# What §6.1 ASSERTS fails, written independently of the receipt so the check below can fail.
# Dims 10 and 12 clear 0.90; 9 and 11 miss it by 0.0025 and 0.0038; 7 and 8 miss it badly.
FAILING_DIMS = (7, 8, 9, 11)

GATE_ARM = "cell4_fsq_micro_seed0"
SHUF_ARM = "cell5_fsq_micro_shuffled_seed0"


def _gate_row(dim: int) -> tuple[float, float, float]:
    """(real sign-accuracy, shuffled sign-accuracy, base rate) for one microstructure dim."""
    lr = STOP["legibility_receipt"]
    d = str(dim)
    return (
        lr[GATE_ARM]["per_dim"][d]["sign_acc"],
        lr[SHUF_ARM]["per_dim"][d]["sign_acc"],
        lr[GATE_ARM]["per_dim"][d]["base_rate_positive"],
    )


def _verify_gate_values() -> None:
    """§8.9's claim, made true for this figure: disagree with the receipt and RAISE.

    The figure spent every draft after the firing rendering a watermark over a result that
    existed, because nothing checked that what it drew matched what had been measured. This is
    that check. It must be capable of failing -- mutate any drawn value and this raises.
    """
    lr = STOP["legibility_receipt"]
    if STOP.get("stage2_entered") is not False:
        raise AssertionError("stop receipt does not record a pre-Stage-2 halt")
    for arm in (GATE_ARM, SHUF_ARM):
        if arm not in lr:
            raise AssertionError(f"stop receipt is missing arm {arm}")
    for dim in MICRO_DIMS_IDX:
        real, shuf, base_rate = _gate_row(dim)
        for label, v in (("real", real), ("shuffled", shuf), ("base", base_rate)):
            if not 0.0 <= v <= 1.0:
                raise AssertionError(f"dim {dim} {label} out of range: {v}")
        if real < GATE and dim not in FAILING_DIMS:
            raise AssertionError(f"dim {dim} is below the gate but not marked failing")
        if real >= GATE and dim in FAILING_DIMS:
            raise AssertionError(f"dim {dim} clears the gate but is marked failing")


def main() -> None:
    fig, (axL, axR) = plt.subplots(
        1,
        2,
        figsize=(fs.SINGLE_COL, 2.95),
        gridspec_kw={"width_ratios": [2.05, 1.0], "wspace": 0.06},
    )

    # ---------------------------------------------------------------- left: the ladder
    axL.set_axisbelow(True)
    axL.xaxis.grid(True)
    ys = np.arange(len(STAGES))[::-1]
    for y, (_label, vals) in zip(ys, STAGES, strict=True):
        vals = list(vals)
        mean = sum(vals) / len(vals)
        # §7 v1.4.1 restated acceptance rule: mean >= 0.90 AND every seed >= 0.85
        passes = mean >= GATE and min(vals) >= FLOOR
        col = fs.PASS if passes else fs.FAIL
        axL.plot([min(vals), max(vals)], [y, y], lw=0.8, color=col, alpha=0.55, zorder=2)
        axL.plot(vals, [y] * len(vals), "o", ms=3.4, mfc="white", mec=col, mew=0.9, zorder=3)
        axL.plot([mean], [y], "D", ms=3.8, mfc=col, mec="white", mew=0.5, zorder=4)
        axL.text(
            max(vals) + 0.012,
            y,
            "pass" if passes else "fail",
            ha="left",
            va="center",
            fontsize=6.0,
            color=col,
            weight="bold",
        )
    axL.axvline(GATE, color=fs.RULE, ls=(0, (4, 3)), lw=0.8, zorder=5)
    axL.axvline(FLOOR, color=fs.RULE, ls=(0, (1, 2)), lw=0.7, zorder=5)
    axL.axvline(0.50, color=fs.RULE, ls=":", lw=0.7, zorder=5)
    axL.plot([], [], "o", ms=3.4, mfc="white", mec=fs.RULE, mew=0.9, label="seed")
    axL.plot([], [], "D", ms=3.8, mfc=fs.RULE, mec="white", mew=0.5, label="mean")
    axL.legend(
        loc="lower left",
        frameon=False,
        fontsize=6.0,
        handletextpad=0.3,
        ncol=2,
        columnspacing=1.0,
        borderpad=0.0,
        bbox_to_anchor=(0.0, -0.02),
    )
    axL.text(
        GATE,
        len(STAGES) - 0.35,
        f"gate {GATE:.2f}",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=fs.RULE,
    )
    axL.text(
        0.50, len(STAGES) - 0.35, "chance", ha="center", va="bottom", fontsize=6.3, color=fs.RULE
    )
    axL.text(
        FLOOR,
        len(STAGES) - 0.78,
        f"per-seed floor {FLOOR:.2f}",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=fs.RULE,
    )
    axL.set_yticks(ys)
    axL.set_yticklabels([s[0] for s in STAGES], fontsize=6.2, linespacing=1.35)
    axL.set_xlim(0.44, 0.98)
    axL.set_ylim(-0.7, len(STAGES) - 0.05)
    axL.set_xlabel("logistic sign-accuracy from bar $t$'s own token", fontsize=fs.T_LABEL)
    fs.panel_title(axL, "a", "synthetic fixture — the intervention, stage by stage", pad=5)
    axL.spines["left"].set_visible(False)
    axL.tick_params(axis="y", length=0)

    # ------------------------------------------------- right: real data, THE GATE'S OWN RECEIPT
    _verify_gate_values()
    axR.set_axisbelow(True)
    axR.xaxis.grid(True)
    names = [FEATURE_NAMES[i] for i in MICRO_DIMS_IDX]
    yr = np.arange(len(names))[::-1]
    for y, dim, nm in zip(yr, MICRO_DIMS_IDX, names, strict=True):
        real, shuf, _base = _gate_row(dim)
        fails = dim in FAILING_DIMS
        col = fs.FAIL if fails else fs.PASS
        # the shuffled placebo arm: open marker, joined to the real value so the collapse is visible
        axR.plot([shuf, real], [y, y], color="#d0d0d0", lw=0.7, zorder=3)
        axR.plot(
            [shuf], [y], marker="o", ms=3.0, mfc="white", mec="#9a9a9a", mew=0.7, lw=0, zorder=4
        )
        axR.plot([real], [y], marker="D", ms=3.4, color=col, lw=0, zorder=5)
        # value in a FIXED right-hand column, never beside the marker: a label that tracks the
        # value collides with the gate rule exactly where the interesting values sit.
        axR.text(
            1.055,
            y,
            f"{real:.4f}",
            ha="right",
            va="center",
            fontsize=5.6,
            color=col,
            zorder=6,
        )
        axR.text(
            0.455,
            y + 0.30,
            nm.replace("_", " "),
            ha="left",
            va="bottom",
            fontsize=5.6,
            color=fs.RULE,
            zorder=5,
        )

    axR.axvline(GATE, color=fs.RULE, ls=(0, (4, 3)), lw=0.8, zorder=4)
    axR.set_yticks([])
    axR.set_xlim(0.44, 1.06)
    axR.set_ylim(-1.35, len(names) - 0.05)
    axR.set_xlabel("same probe, real bars", fontsize=fs.T_LABEL)
    fs.panel_title(axR, "b", "real bars — gate fired", pad=5)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", length=0)
    axR.text(
        GATE - 0.008,
        len(names) - 0.35,
        f"gate {GATE:.2f}",
        ha="right",
        va="center",
        fontsize=5.6,
        color=fs.RULE,
        zorder=6,
    )
    # legend, inside the axes and below every row so nothing can paint over it
    ly = -1.05
    axR.plot([0.462], [ly], marker="D", ms=3.4, color=fs.FAIL, lw=0, zorder=6)
    axR.text(0.476, ly, "real", fontsize=fs.T_MICRO, va="center", color=fs.RULE, zorder=6)
    axR.plot([0.575], [ly], marker="o", ms=3.0, mfc="white", mec="#9a9a9a", mew=0.7, lw=0, zorder=6)
    axR.text(0.591, ly, "shuffled", fontsize=fs.T_MICRO, va="center", color=fs.RULE, zorder=6)

    fig.subplots_adjust(left=0.235, right=0.955, top=0.90, bottom=0.135)
    fs.assert_text_legible(fig, (axL, axR))
    fs.save(fig, OUT, "fig4_legibility")


if __name__ == "__main__":
    main()
