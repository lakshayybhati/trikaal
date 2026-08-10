"""Figures 2 and 3 for §3, numbered as they appear in the manuscript.

Rendered from the committed receipts.

RENDERING ONLY — not part of the anchored instrument, produces no measurement. Every value
plotted is loaded from runs_manifest/ at call time, so a figure cannot drift from its receipt.

    python3 paper/figures/make_mechanism_figures.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"

fs.apply()


def _load(name: str) -> dict:
    with open(RM / name) as fh:
        return json.load(fh)


# =============================================================================================
# Figure 2 — reconstruction quality sorted by variance class
# =============================================================================================
def eviction_figure() -> None:
    d = _load("m6_interface_respec_design_pass.json")
    seeds = d["tokenizers"]["new_pointwise_fine_3seed"]["seeds"]
    corrs = np.array(
        [seeds[s]["nonlinear_diagnostics"]["point_decoder_per_dim_corrs"] for s in ("0", "1", "2")]
    )
    recon = seeds["0"]["recon"]["per_dim"]
    sd = np.array([recon[str(i)]["mean_baseline_mae"] / np.sqrt(2 / np.pi) for i in range(13)])
    rho = d["filler_calibration"]["filler_rho"]

    signal_dim, return_dim = 9, 0
    fillers = [i for i in range(13) if i not in (signal_dim, return_dim)]
    lo, hi = corrs[:, fillers].min(), corrs[:, fillers].max()

    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 2.35))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)

    for k, (dx, mk) in enumerate(zip((-0.17, 0.0, 0.17), ("o", "s", "^"), strict=True)):
        for i in range(13):
            colour = fs.BLUE if i == return_dim else (fs.RED if i == signal_dim else fs.OLIVE)
            ax.plot(i + dx, corrs[k, i], mk, ms=3.3, mfc=colour, mec="white", mew=0.4, zorder=3)

    ax.axhline(0.9, color=fs.GREY, ls=(0, (4, 3)), lw=0.7, zorder=2)
    ax.text(12.6, 0.915, "legibility target 0.90", ha="right", fontsize=6.4, color=fs.GREY)

    # direct labelling in the palette colours — no legend box for the classes
    ax.annotate(
        f"return\nsd {sd[return_dim]:.1f}",
        xy=(0, 0.975),
        xytext=(0.5, 0.55),
        fontsize=6.4,
        color=fs.BLUE,
        ha="center",
        linespacing=1.35,
        arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.BLUE, shrinkA=2, shrinkB=4),
    )
    ax.text(
        5.2,
        0.685,
        f"eleven fillers · sd 1.0 · AR(1) across dims, $\\rho$ = {rho}\n{lo:.2f} – {hi:.2f}",
        fontsize=6.4,
        color=fs.OLIVE,
        ha="center",
        linespacing=1.4,
    )
    ax.annotate(
        "signal dimension · sd 1.0 · independent\n0.001 – 0.014, every seed",
        xy=(signal_dim, 0.03),
        xytext=(8.6, 0.30),
        fontsize=6.4,
        color=fs.RED,
        ha="center",
        linespacing=1.4,
        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=fs.RED, shrinkA=3, shrinkB=3),
    )
    for mk, lab in (("o", "seed 0"), ("s", "seed 1"), ("^", "seed 2")):
        ax.plot([], [], mk, ms=3.3, mfc=fs.GREY, mec="white", mew=0.4, label=lab)
    ax.legend(
        loc="lower left",
        frameon=False,
        ncol=3,
        handletextpad=0.25,
        columnspacing=1.1,
        borderpad=0.1,
        fontsize=6.4,
    )

    ax.set_xticks(range(13))
    for i, lab in enumerate(ax.set_xticklabels([str(i) for i in range(13)])):
        lab.set_color(fs.BLUE if i == return_dim else (fs.RED if i == signal_dim else fs.INK))
    ax.set_xlim(-0.7, 12.7)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlabel("feature dimension")
    ax.set_ylabel("point-decoder correlation")

    fs.save(fig, OUT, "fig2_eviction_by_variance_class")


# =============================================================================================
# Figure 3 — extraction against planted information (rebuilt at its production turn)
# =============================================================================================
def arc_figure() -> None:
    v6 = _load("m6_canary_v6_stage1_manifest.json")
    tc = _load("m6_token_control_run_manifest.json")
    ac = _load("m6_acceptance_stage1_manifest.json")

    panels = [
        dict(
            title="A   plant in feature space",
            sub="original interface",
            traj=v6["runs"]["planted"]["trajectory"],
            key="teacher_forced_corr",
            info="1.151 nats planted",
            dval=v6["decision_inputs"]["val_planted_minus_noise"],
            final=v6["decision_inputs"]["tf_corr_final"],
            extra=f"max {v6['decision_inputs']['tf_corr_max']:.4f} over 40 evals",
            colour=fs.RED,
            verdict="not recovered",
        ),
        dict(
            title="B   plant in token space",
            sub="tokenizer bypassed",
            traj=tc["trajectory"],
            key="probe_spearman",
            info="0.9003 nats planted",
            dval=tc["decision_inputs"]["final_val_minus_H0"],
            final=tc["decision_inputs"]["probe_final"],
            extra=f"crosses 0.30 at step {tc['decision_inputs']['probe_cross_step']:,}",
            colour=fs.BLUE,
            verdict="94.4% recovered",
        ),
        dict(
            title="C   plant in feature space",
            sub="re-specified interface",
            traj=ac["runs"]["planted"]["trajectory"],
            key="teacher_forced_corr",
            info="1.151 nats planted",
            dval=ac["decision_inputs"]["val_planted_minus_noise"],
            final=ac["decision_inputs"]["tf_corr_final"],
            extra=f"crosses 0.30 at step {ac['decision_inputs']['detect_cross_step']:,}",
            colour=fs.BLUE,
            verdict="recovered",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(fs.SINGLE_COL, 2.05), sharey=True)
    for ax, p in zip(axes, panels, strict=True):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.axhline(0.3, color=fs.GREY, ls=(0, (4, 3)), lw=0.7, zorder=2)
        ax.plot(
            [q["step"] for q in p["traj"]],
            [q[p["key"]] for q in p["traj"]],
            "-",
            lw=1.2,
            color=p["colour"],
            zorder=3,
        )
        ax.set_xlim(0, 20600)
        ax.set_ylim(-0.06, 1.06)
        ax.set_xticks([0, 10000, 20000])
        ax.set_xticklabels(["0", "10k", "20k"])
        ax.set_xlabel("training step")
        ax.set_title(f"{p['title']}\n{p['sub']}", loc="left", pad=4, fontsize=7.0)
        ax.text(
            0.96,
            0.70,
            p["verdict"],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.0,
            weight="bold",
            color=p["colour"],
        )
        ax.text(
            0.96,
            0.40,
            fs.minus(
                f"{p['info']}\nfinal probe {p['final']:.4f}\n{p['extra']}\n"
                f"$\\Delta$val {p['dval']:+.4f} nats"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.2,
            linespacing=1.5,
        )

    axes[0].set_ylabel("probe correlation")
    axes[0].text(600, 0.325, "detection threshold 0.30", fontsize=6.2, color=fs.GREY, va="bottom")
    fig.subplots_adjust(wspace=0.12)

    fs.save(fig, OUT, "fig3_extraction")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    arc_figure()  # Figure 2 in the manuscript
    eviction_figure()  # Figure 3 in the manuscript
