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
    src/trikaal/train/gates.py                           MICRO_DIMS, MICRO_LEGIBILITY_MIN

RIGHT PANEL CARRIES NO VALUES AT ALL, DELIBERATELY. The standing gate's first execution on real
microstructure has not happened. An earlier draft drew six plausible bars near the gate, which
implied a passing result that nobody has measured -- exactly the defect class this project exists
to prevent. The panel now draws six EMPTY slots: the axis, the ordering, the threshold and the
caption are final, and no value is implied in either direction. Six numbers from the run's gate
receipt fill the slots.

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
from matplotlib.patches import Rectangle

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
    axL.set_xlabel("logistic sign-accuracy from bar $t$'s own token")
    axL.set_title(
        "synthetic fixture — the intervention, stage by stage", loc="left", fontsize=7.0, pad=5
    )
    axL.spines["left"].set_visible(False)
    axL.tick_params(axis="y", length=0)

    # ---------------------------------------------------------------- right: real data, MOCK
    axR.set_axisbelow(True)
    axR.xaxis.grid(True)
    names = [FEATURE_NAMES[i] for i in MICRO_DIMS_IDX]
    yr = np.arange(len(names))[::-1]
    for y, nm in zip(yr, names, strict=True):
        # an EMPTY slot, matching the empty outcome cells of Figure 8. No value is implied.
        axR.add_patch(
            Rectangle(
                (0.455, y - 0.29),
                0.50,
                0.58,
                fc="white",
                ec="#c8c8c8",
                lw=0.6,
                ls=(0, (2, 1.5)),
                zorder=2,
            )
        )
        axR.text(
            0.468,
            y,
            nm.replace("_", " "),
            ha="left",
            va="center",
            fontsize=6.0,
            color=fs.RULE,
            zorder=5,
        )
        axR.text(0.93, y, "—", ha="right", va="center", fontsize=7.0, color="#b0b0b0", zorder=5)

    axR.axvline(GATE, color=fs.RULE, ls=(0, (4, 3)), lw=0.8, zorder=4)
    axR.set_yticks([])
    axR.set_xlim(0.44, 0.98)
    axR.set_ylim(-0.7, len(names) - 0.05)
    axR.set_xlabel("same probe, real bars")
    axR.set_title("real microstructure — the standing gate", loc="left", fontsize=7.0, pad=5)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", length=0)

    axR.text(
        0.71,
        (len(names) - 1) / 2,
        "NO VALUES\nawaiting the run",
        ha="center",
        va="center",
        fontsize=8.5,
        weight="bold",
        color=fs.FAIL,
        rotation=16,
        alpha=0.85,
        zorder=7,
        linespacing=1.4,
    )

    fs.save(fig, OUT, "fig4_legibility")


if __name__ == "__main__":
    main()
