"""Figure 7 — symbol-month coverage of the 200-instrument lake, with survivorship visible.

VERDICT THE FIGURE MUST CARRY WITHOUT ITS CAPTION
    The universe is ragged on both edges by construction: instruments list at different times, and
    those that stop trading are kept, so their absence afterwards is the ragged right edge. A
    survivorship-filtered universe would be a solid rectangle -- exactly the artefact this panel
    exists to show we did not create.

ARTIFACTS — enumerated from the lake itself at render time, not transcribed.
    processed/universe_bars/symbol=*/month=*      -> the coverage matrix, 200 x 48
    runs_manifest/m6_c10_micro_density.json       -> total_bars, n_symbols (cross-check)
    runs_manifest/m6_mde_inputs.json              -> primary_region_ms (the headline eval region)
    docs/milestone4b_universe_ingest.md           -> the "41 delisted" selection figure

RECONCILIATION, measured 2026-08-09. The ingest note records "200 symbols selected, 41 delisted".
This panel reports 17. The two count different things and both are right: 41 is the number known to
be delisted as of the ingest scan, at any date; 17 is the number whose data ENDS INSIDE the lake's
2021-01 to 2024-12 window, which is all a coverage raster over that window can show. The remaining
24 delist after 2024-12 and are complete here. Spot-check against the note's two worked examples:
AGIXUSDT 2023-02 to 2024-06 and AUDIOUSDT 2021-08 to 2024-05 in the lake, against 2024-07 and
2024-06 in the note -- the note quotes the first ABSENT month, the lake stores the last PRESENT
one. The figure therefore says "stop trading inside the window", never "delisted", and §5 must
carry the same distinction. Two further symbols (ICPUSDT, TLMUSDT) have interior gaps rather than
a truncated tail; they are present in the raster as internal white streaks.

The train/eval boundary and the primary evaluation region are drawn from the committed fold plan,
so a reader can see which part of the raggedness the headline metric is actually scored on.

RENDERING ONLY — not part of the anchored instrument; produces no measurement.

    python3 paper/figures/make_fig7_coverage.py
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import figstyle as fs

ROOT = pathlib.Path(__file__).resolve().parents[2]
LAKE = ROOT / "processed" / "universe_bars"
RM = ROOT / "runs_manifest"
OUT = ROOT / "paper" / "figures"

fs.apply()

TRAIN_EVAL_MS = 1_697_820_480_000  # 0.7 point of the 4-year window; committed fold plan


def month_index(months: list[str], ms: int) -> float:
    d = dt.datetime.fromtimestamp(ms / 1000, dt.UTC)
    key = f"{d.year:04d}-{d.month:02d}"
    frac = (d.day - 1) / 31.0
    return (months.index(key) if key in months else len(months)) + frac


def coverage() -> tuple[np.ndarray, list[str], list[str]]:
    cov: dict[str, set[str]] = {}
    for sd in sorted(LAKE.glob("symbol=*")):
        sym = sd.name.split("=", 1)[1]
        ms = {p.name.split("=", 1)[1] for p in sd.rglob("month=*")}
        if ms:
            cov[sym] = ms
    months = sorted({m for v in cov.values() for m in v})
    # sort by listing month, then by delisting month, so both edges read as staircases
    order = sorted(cov, key=lambda s: (min(cov[s]), max(cov[s]), s))
    mat = np.zeros((len(order), len(months)), dtype=float)
    for i, sym in enumerate(order):
        for m in cov[sym]:
            mat[i, months.index(m)] = 1.0
    return mat, order, months


def main() -> None:
    mat, _symbols, months = coverage()
    n_sym, n_mon = mat.shape
    last_month = months[-1]
    delisted = [i for i in range(n_sym) if mat[i, -1] == 0]

    prim = json.loads((RM / "m6_mde_inputs.json").read_text())["primary_region_ms"]
    x_train = month_index(months, TRAIN_EVAL_MS)
    x_prim0 = month_index(months, prim[0])

    fig, ax = plt.subplots(figsize=(fs.SINGLE_COL, 3.15))
    ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=ListedColormap(["#ffffff", fs.OHLCV]),
        vmin=0,
        vmax=1,
        extent=(0, n_mon, 0, n_sym),
        zorder=2,
    )

    # the two committed boundaries the headline metric depends on
    for x, col in ((x_train, fs.RULE), (x_prim0, fs.FAIL)):
        ax.axvline(x, color=col, lw=1.0, ls=(0, (4, 2.5)), zorder=4)
    ax.annotate(
        "train / eval boundary\n2023-10-20",
        xy=(x_train, n_sym * 0.30),
        xytext=(x_train - 12.5, n_sym * 0.30),
        ha="right",
        va="center",
        fontsize=6.0,
        color=fs.RULE,
        linespacing=1.35,
        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=fs.RULE, shrinkA=1, shrinkB=2),
    )
    ax.annotate(
        "headline eval region\nblocks 1–5",
        xy=(x_prim0, n_sym * 0.10),
        xytext=(x_prim0 - 9.0, n_sym * 0.10),
        ha="right",
        va="center",
        fontsize=6.0,
        color=fs.FAIL,
        linespacing=1.35,
        arrowprops=dict(arrowstyle="-|>", lw=0.6, color=fs.FAIL, shrinkA=1, shrinkB=2),
    )

    # mark the delisted rows on the right margin
    for i in delisted:
        ax.plot(
            [n_mon + 0.6],
            [i + 0.5],
            marker="_",
            ms=2.6,
            color=fs.FAIL,
            mew=0.9,
            clip_on=False,
            zorder=5,
        )
    ax.text(
        n_mon + 1.6,
        n_sym * 0.5,
        f"{len(delisted)} instruments\nstop trading\ninside the window",
        ha="left",
        va="center",
        fontsize=6.1,
        color=fs.FAIL,
        linespacing=1.4,
    )

    tick_at = [i for i, m in enumerate(months) if m.endswith("-01")]
    ax.set_xticks([t + 0.5 for t in tick_at])
    ax.set_xticklabels([months[t][:4] for t in tick_at])
    ax.set_yticks([0, 50, 100, 150, n_sym])
    ax.set_xlim(0, n_mon)
    ax.set_ylim(0, n_sym)
    ax.set_xlabel(f"month  ({months[0]} to {last_month}, {n_mon} months)")
    ax.set_ylabel("instrument, ordered by listing month")
    ax.set_title(
        f"{n_sym} instruments · {mat.sum():,.0f} of {n_sym * n_mon:,} symbol-months present"
        f"  ({100 * mat.mean():.0f}%)",
        loc="left",
        fontsize=7.0,
        pad=5,
    )
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fs.save(fig, OUT, "fig7_coverage")


if __name__ == "__main__":
    main()
