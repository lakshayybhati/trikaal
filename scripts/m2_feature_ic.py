"""M2 report add-on — per-feature causal IC screen + first-cut MDE / effective-N.

READ-ONLY on the existing BTCUSDT-2023 Parquet/DuckDB lake; CPU/local; independent of any
training run. It does NOT re-download, modify the feature pipeline, retrain, or start Stage-2.
It answers the half M2 left open: M2 proved the tokenizer *reconstructs* real microstructure;
this asks whether each feature *predicts* forward returns (spec §8.B / §8.E.8 first cut).

    python3 scripts/m2_feature_ic.py [N_BOOT] [SEED]      # defaults: 1000, 0

What it does:
  * loads the STORED causal features x_t straight from the lake (never recomputes them — and
    asserts the reload reproduces the frozen M2 dataset_hash, so there is no chance of a
    recompute that peeks);
  * builds the forward-return label y_{t,h} = log(C_{t+h}/C_{t+1}) (entry next-bar, §8.B) by
    cumulating the lake's stored raw_ret_close within each segment — the label is the *only*
    forward-reading object;
  * reports Spearman RankIC (primary) + Pearson IC with a moving-block-bootstrap 95% CI for the
    13 active features at h ∈ {1,5,15,60}, foregrounding the 6 aggTrades microstructure channels;
  * reports a first-cut effective-N / MDE so the micro RankIC can be read against the noise floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np

from trikaal.constants import FEATURE_NAMES, MASK_BITS_AGGTRADES, MASK_BITS_PERP, N_FEATURES
from trikaal.eval.ic_screen import (
    effective_sample_size,
    mde_from_neff,
    moving_block_bootstrap_ci,
    pearson,
    rankdata_average,
    spearman_rankic,
)
from trikaal.utils.hashing import content_hash

SYMBOL = "BTCUSDT"
LAKE_ROOT = Path("processed/bars")
HORIZONS = (1, 5, 15, 60)  # minutes / bars
ACTIVE_IDX = tuple(range(13))  # dims 0..12 are active in the M2 build; 13..15 (funding/OI) masked
MIN_VALID = 1000  # need a usable sample before quoting an IC
DAY_BARS = 1440  # bootstrap block-length floor (1 trading day)
# Frozen M2 dataset hash (scripts/build_btcusdt_stage1.py) — reused here purely as an integrity
# assert that we are reading the same stored features, not a fresh (possibly peeking) recompute.
M2_DATASET_HASH_PREFIX = "sha256:1d3ec4f8db6686e2"


def load_lake(root: Path) -> dict[str, np.ndarray]:
    """Read the stored per-bar features + raw_ret_close hook for one symbol, bar-open order."""
    con = duckdb.connect()
    glob = f"{root}/**/*.parquet"
    con.execute(f"CREATE VIEW bars AS SELECT * FROM read_parquet('{glob}', hive_partitioning=true)")
    xcols = ", ".join(f"x_{i}" for i in range(N_FEATURES))
    mcols = ", ".join(f"m_{i}" for i in range(N_FEATURES))
    tbl = con.execute(
        f"SELECT bar_open_ms, segment_id, raw_ret_close, {xcols}, {mcols} "
        "FROM bars WHERE symbol = ? AND frequency = '1m' ORDER BY bar_open_ms",
        [SYMBOL],
    ).to_arrow_table()
    n = tbl.num_rows
    x = np.empty((n, N_FEATURES), dtype=np.float32)
    m = np.empty((n, N_FEATURES), dtype=np.uint8)
    for i in range(N_FEATURES):
        x[:, i] = tbl.column(f"x_{i}").to_numpy()
        m[:, i] = tbl.column(f"m_{i}").to_numpy()
    return {
        "x": np.ascontiguousarray(x),
        "m": np.ascontiguousarray(m),
        "ts": tbl.column("bar_open_ms").to_numpy().astype(np.int64),
        "segment_id": tbl.column("segment_id").to_numpy().astype(np.int64),
        "raw_ret_close": tbl.column("raw_ret_close").to_numpy().astype(np.float64),
    }


def segment_bounds(segment_id: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [s0, s1) ranges of equal segment_id (the lake is bar-open ordered)."""
    cuts = np.flatnonzero(np.diff(segment_id) != 0) + 1
    edges = [0, *cuts.tolist(), segment_id.shape[0]]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def forward_returns(
    raw_ret_close: np.ndarray, bounds: list[tuple[int, int]], h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Forward return y_t at horizon ``h`` and its validity mask, per the §8.B entry-next-bar rule.

    * h == 1: the §8.B return log(C_{t+1}/C_{t+1}) is identically 0 (entry == exit), so the 1-bar
      column reports the decision-relative next-bar return log(C_{t+1}/C_t) = raw_ret_close[t+1] —
      exactly the codebase's canonical 1-step label and what the model's 1-step head predicts.
    * h >= 2: y_t = log(C_{t+h}/C_{t+1}) = clp[t+h] − clp[t+1], cumulated within-segment.
    All variants read only bars > t; validity requires the whole window to lie in one segment.
    """
    n = raw_ret_close.shape[0]
    y = np.full(n, np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for s0, s1 in bounds:
        # within-segment cumulative log-price clp[i] = log(C_i / C_{s0}); raw_ret_close[s0] == 0.
        clp = np.cumsum(raw_ret_close[s0:s1])
        seg_n = s1 - s0
        if h == 1:
            # y_local[p] = raw_ret_close[s0 + p + 1] for p in [0, seg_n - 2]
            last = seg_n - 1  # last decision bar p with p+1 in segment
            if last > 0:
                p = np.arange(0, last)
                y[s0 + p] = raw_ret_close[s0 + p + 1]
                valid[s0 + p] = True
        else:
            last = seg_n - h  # last decision bar p with p+h in segment
            if last > 0:
                p = np.arange(0, last)
                y[s0 + p] = clp[p + h] - clp[p + 1]
                valid[s0 + p] = True
    return y, valid


def frac_constant(v: np.ndarray) -> float:
    """Fraction of observations equal to the single most frequent value (liveness probe)."""
    if v.shape[0] == 0:
        return 1.0
    _, counts = np.unique(v, return_counts=True)
    return float(counts.max()) / float(v.shape[0])


def _fmt_signed(x: float, nd: int = 4) -> str:
    return "nan" if not np.isfinite(x) else f"{x:+.{nd}f}"


def main(n_boot: int = 1000, seed: int = 0) -> int:
    if not LAKE_ROOT.exists():
        print(f"lake not found at {LAKE_ROOT} — run scripts/build_btcusdt_stage1.py first")
        return 1
    data = load_lake(LAKE_ROOT)
    x, m, ts = data["x"], data["m"], data["ts"]
    rrc, seg = data["raw_ret_close"], data["segment_id"]
    n = x.shape[0]

    # --- integrity: assert we read the STORED frozen features, not a peeking recompute -------
    dataset_hash = content_hash(x, m, ts)
    assert dataset_hash.startswith(M2_DATASET_HASH_PREFIX), (
        f"stored-feature integrity FAILED: {dataset_hash[:23]}… != M2 {M2_DATASET_HASH_PREFIX}…"
    )

    # --- funding/OI are deferred in this build: assert fully masked, then report N/A ----------
    perp_masked = {i: float(m[:, i].mean()) for i in MASK_BITS_PERP}
    bounds = segment_bounds(seg)

    rng = np.random.default_rng(seed)
    fwd = {h: forward_returns(rrc, bounds, h) for h in HORIZONS}

    # --- per-feature × horizon RankIC / Pearson IC + bootstrap CI ----------------------------
    rows: list[dict[str, object]] = []
    # arrays for a stable, reproducible results hash (feature-major, horizon-minor)
    h_rankic, h_pear, h_lo, h_hi, h_nval = [], [], [], [], []
    for f in ACTIVE_IDX:
        feat_ok = m[:, f] == 0
        per_h: dict[int, dict[str, float]] = {}
        carries = False
        std_any = 0.0
        fc_any = 1.0
        for h in HORIZONS:
            y, vmask = fwd[h]
            sel = feat_ok & vmask & np.isfinite(x[:, f])
            nval = int(sel.sum())
            xf = x[sel, f].astype(np.float64)
            yf = y[sel]
            std_any = max(std_any, float(xf.std()) if nval else 0.0)
            fc = frac_constant(xf) if nval else 1.0
            fc_any = min(fc_any, fc)
            if nval < MIN_VALID or xf.std() == 0.0:
                ric = pic = lo = hi = float("nan")
            else:
                ric = spearman_rankic(xf, yf)
                pic = pearson(xf, yf)
                block = max(h, DAY_BARS)
                lo, hi = moving_block_bootstrap_ci(
                    rankdata_average(xf),
                    rankdata_average(yf),
                    block_len=block,
                    n_boot=n_boot,
                    rng=rng,
                )
            sig = np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0)
            carries = carries or sig
            per_h[h] = {"ric": ric, "pic": pic, "lo": lo, "hi": hi, "n": nval, "sig": sig}
            h_rankic.append(ric)
            h_pear.append(pic)
            h_lo.append(lo)
            h_hi.append(hi)
            h_nval.append(nval)
        if std_any < 1e-8 or fc_any > 0.99:
            verdict = "dead"
        elif carries:
            verdict = "carries-signal"
        else:
            verdict = "live-no-signal"
        rows.append(
            {
                "f": f,
                "name": FEATURE_NAMES[f],
                "per_h": per_h,
                "verdict": verdict,
                "std": std_any,
                "fc": fc_any,
            }
        )

    # --- effective-N / MDE on the per-bar (1-bar) forward return -----------------------------
    y1, v1 = fwd[1]
    eff = effective_sample_size(y1[v1], max_lag=DAY_BARS)
    mde = mde_from_neff(eff.n_eff)

    # --- results content-hash (reproducible from lake + seed + n_boot + block rule) ----------
    results_hash = content_hash(
        np.asarray(h_rankic),
        np.asarray(h_pear),
        np.asarray(h_lo),
        np.asarray(h_hi),
        np.asarray(h_nval, dtype=np.int64),
        {
            "seed": seed,
            "n_boot": n_boot,
            "horizons": list(HORIZONS),
            "block_floor": DAY_BARS,
            "neff": round(eff.n_eff, 3),
        },
    )

    _print_report(rows, eff, mde, perp_masked, n, n_boot, seed, dataset_hash, results_hash)
    return 0


def _print_report(rows, eff, mde, perp_masked, n, n_boot, seed, dataset_hash, results_hash) -> None:
    micro = set(MASK_BITS_AGGTRADES)
    p = print
    p("\n## M2 add-on — per-feature causal IC screen + first-cut statistical power\n")
    p(
        f"Read-only on the BTCUSDT-2023 lake ({n:,} bars, 1 segment). Forward-return label "
        "`y_{t,h} = log(C_{t+h}/C_{t+1})` (entry next-bar, §8.B), reconstructed within-segment "
        "from stored `raw_ret_close`. RankIC = Spearman (primary); 95% CI = moving-block bootstrap "
        f"(block = max(h, 1440 bars = 1 day), {n_boot} resamples, seed {seed}). A feature "
        "**carries standalone signal** only if its RankIC CI excludes 0.\n"
    )
    p(
        "> **h=1 convention.** The §8.B entry-next-bar return is identically 0 at h=1 "
        "(entry==exit), so the 1-bar column reports the decision-relative next-bar return "
        "`log(C_{t+1}/C_t)` — the model's 1-step target. h∈{5,15,60} use the §8.B return.\n"
    )

    # primary table: RankIC ± CI per (feature, horizon) + verdict
    p("### RankIC ± 95% CI  (★ = CI excludes 0)\n")
    head = "| feature (dim) | " + " | ".join(f"h={h}" for h in HORIZONS) + " | verdict |"
    p(head)
    p("|" + "---|" * (len(HORIZONS) + 2))
    for r in rows:
        tag = " ·micro" if r["f"] in micro else ""
        cells = []
        for h in HORIZONS:
            c = r["per_h"][h]
            star = "★" if c["sig"] else ""
            cells.append(
                f"{_fmt_signed(c['ric'])} [{_fmt_signed(c['lo'])},{_fmt_signed(c['hi'])}]{star}"
            )
        p(f"| `{r['name']}` ({r['f']}){tag} | " + " | ".join(cells) + f" | {r['verdict']} |")
    p("")

    # secondary: Pearson IC point estimates
    p("### Pearson IC (secondary)\n")
    p("| feature (dim) | " + " | ".join(f"h={h}" for h in HORIZONS) + " |")
    p("|" + "---|" * (len(HORIZONS) + 1))
    for r in rows:
        cells = [_fmt_signed(r["per_h"][h]["pic"]) for h in HORIZONS]
        p(f"| `{r['name']}` ({r['f']}) | " + " | ".join(cells) + " |")
    p("")

    # funding/OI deferred note
    perp_names = ", ".join(f"`{FEATURE_NAMES[i]}` ({i})" for i in MASK_BITS_PERP)
    allmask = all(v > 0.999 for v in perp_masked.values())
    p(
        f"**Funding/OI ({perp_names}): N/A in M2 build** — all bars masked "
        f"(mask rate {'=1.000 ✓' if allmask else str(perp_masked)}); IC is not computed on zero "
        "valid bars. Screen these when the futures-API ingest lands.\n"
    )

    # interpretation of the 6 aggTrades channels
    p("### Which of the 6 aggTrades channels predict vs are inert\n")
    for r in rows:
        if r["f"] in micro:
            best = max(
                HORIZONS,
                key=lambda h: (
                    abs(r["per_h"][h]["ric"]) if np.isfinite(r["per_h"][h]["ric"]) else -1.0
                ),
            )
            c = r["per_h"][best]
            p(
                f"- `{r['name']}` (dim {r['f']}): **{r['verdict']}** — peak |RankIC| "
                f"{_fmt_signed(c['ric'])} @ h={best} "
                f"(CI [{_fmt_signed(c['lo'])},{_fmt_signed(c['hi'])}])."
            )
    p("")

    # power
    p("### First-cut MDE / effective-N (single-symbol, temporal only)\n")
    p(
        f"- Per-bar forward-return autocorrelation deflates N: **N = {eff.n:,}**, "
        f"**N_eff = {eff.n_eff:,.0f}** (ratio {eff.ratio:.3f}; {eff.lags_used} significant lags"
        f"{', capped at N' if eff.capped_at_n else ''})."
    )
    p(
        f"- **SE(RankIC) ≈ 1/√N_eff = {mde.se:.4f}** → single-feature MDE@95% = "
        f"**{mde.mde_single:.4f}**; two-cell IC-gap MDE = **{mde.mde_two_cell:.4f}**."
    )
    micro_rics = [
        abs(r["per_h"][h]["ric"])
        for r in rows
        if r["f"] in micro
        for h in HORIZONS
        if np.isfinite(r["per_h"][h]["ric"])
    ]
    peak_micro = max(micro_rics) if micro_rics else float("nan")
    underpowered = np.isfinite(peak_micro) and peak_micro < mde.mde_two_cell
    p(
        f"- Peak live-micro |RankIC| = **{peak_micro:.4f}** vs two-cell MDE "
        f"**{mde.mde_two_cell:.4f}** → "
        + (
            "**FLAG: microstructure leg may be underpowered at full eval — flag before committing "
            "GPU-days.**"
            if underpowered
            else "above the two-cell MDE on this single coin-year (necessary, not sufficient)."
        )
    )
    p(
        "- *First cut on one coin-year, TEMPORAL effective-N only.* The full portfolio MDE "
        "(cross-sectional breadth across ~one-factor symbols and K=6 forward blocks, §8.E.8) is "
        "computed at eval time and will be larger/harder.\n"
    )

    p(f"`dataset_hash = {dataset_hash[:30]}…` (stored x,m,ts — matches M2 ✓)  ")
    p(f"`ic_results_hash = {results_hash[:30]}…`\n")


if __name__ == "__main__":
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    raise SystemExit(main(nb, sd))
