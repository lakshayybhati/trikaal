"""M4b data-quality report over the finished universe lake (M4 exit-gate close-out).

Single-pass DuckDB aggregation over ``processed/universe_bars`` → per-symbol:
  * bar count + date-coverage span (first→last);
  * post-warm-up USABLE bars for BOTH 2x2 ablation arms (mask-bit groups stated below so the
    definition is auditable), plus the **micro-availability ratio** = micro-usable / OHLCV-usable;
  * is_stale% and zero-vol% — the QA mask-drop fractions (``is_stale`` column; ``dq_flags`` bit 4).

Mask semantics (features.py): ``m_i = 1`` means feature i is MASKED (warm-up incomplete, or
missing / zero-vol / vol-recon-err); ``m_i = 0`` means usable.

Ablation arms (a bar is "usable" for an arm iff EVERY bit in the arm's group is 0):
  * **OHLCV arm**  → bits 0-6  = ret_close, range, body, upper_wick_frac, lower_wick_frac,
                                 log_volume, log_amount.
  * **+micro arm** → bits 0-12 = OHLCV (0-6) PLUS the aggTrades microstructure 7-12 = TFI,
                                 signed_count_imbalance, trade_count, mean_trade_size,
                                 trade_size_dispersion, large_trade_share.
  * (Perp bits 13-15 = funding, log_oi, d_oi are EXCLUDED — masked historically per the §1.2 OI
    retention trap, a separate mostly-unavailable channel; including them would zero out the count.)

micro-usable <= OHLCV-usable because zero-vol / vol-recon-err bars mask the micro bits (7-12) while
OHLCV may still be valid. The **micro-availability ratio** is the M6 headline: a low ratio means
the coin contributes ~nothing to the +micro cells and dilutes the shuffled-micro placebo, so it is
a candidate to down-weight / drop from the +micro TRAINING draw (it stays in the eval lake).

Usage:  PYTHONPATH=src python3 scripts/m4b_data_quality.py [--lake processed/universe_bars]
        [--thin-usable-frac 0.5] [--thin-stale-pct 10] [--low-micro-ratio 0.7]
        [--md docs/_m4b_dq_table.md]
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from trikaal.data.lake import connect as lake_connect

LAKE_ROOT = Path("processed/universe_bars")

# arm mask-bit groups (a bar is usable for an arm iff every listed bit is 0)
_OHLCV_BITS = range(7)  # 0..6
_MICRO_BITS = range(13)  # 0..12  (OHLCV + aggTrades microstructure 7-12)
_USABLE_OHLCV = " AND ".join(f"m_{i} = 0" for i in _OHLCV_BITS)
_USABLE_MICRO = " AND ".join(f"m_{i} = 0" for i in _MICRO_BITS)

_QUERY = f"""
SELECT
  symbol,
  count(*)                                                AS n_bars,
  CAST(min(date) AS VARCHAR)                              AS first_date,
  CAST(max(date) AS VARCHAR)                              AS last_date,
  count(DISTINCT segment_id)                              AS n_segments,
  sum(CASE WHEN {_USABLE_OHLCV} THEN 1 ELSE 0 END)        AS usable_ohlcv,
  sum(CASE WHEN {_USABLE_MICRO} THEN 1 ELSE 0 END)        AS usable_micro,
  100.0 * avg(CAST(is_stale AS DOUBLE))                   AS stale_pct,
  100.0 * avg(CAST((dq_flags // 16) % 2 AS DOUBLE))       AS zerovol_pct
FROM bars
GROUP BY symbol
ORDER BY usable_ohlcv ASC
"""


def _micro_ratio(r: dict) -> float:
    return r["usable_micro"] / r["usable_ohlcv"] if r["usable_ohlcv"] else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", default=str(LAKE_ROOT))
    ap.add_argument(
        "--thin-usable-frac",
        type=float,
        default=0.5,
        help="flag symbols whose OHLCV-usable count is below this fraction of the median",
    )
    ap.add_argument("--thin-stale-pct", type=float, default=10.0)
    ap.add_argument(
        "--low-micro-ratio",
        type=float,
        default=0.7,
        help="flag symbols whose micro-avail ratio (micro-usable/OHLCV-usable) is below this",
    )
    ap.add_argument("--md", default=None, help="optional path to write a markdown table")
    args = ap.parse_args()

    con = lake_connect(Path(args.lake))
    rows = con.execute(_QUERY).fetchall()
    cols = [d[0] for d in con.description]
    recs = [dict(zip(cols, r, strict=True)) for r in rows]
    if not recs:
        print("EMPTY LAKE — nothing to report")
        return 1

    n_sym = len(recs)
    tot_bars = sum(r["n_bars"] for r in recs)
    tot_ohlcv = sum(r["usable_ohlcv"] for r in recs)
    tot_micro = sum(r["usable_micro"] for r in recs)
    ohlcv_counts = sorted(r["usable_ohlcv"] for r in recs)
    med_usable = statistics.median(ohlcv_counts)
    thin_floor = args.thin_usable_frac * med_usable

    def is_thin(r: dict) -> bool:
        return r["usable_ohlcv"] < thin_floor or r["stale_pct"] > args.thin_stale_pct

    def is_low_micro(r: dict) -> bool:
        return _micro_ratio(r) < args.low_micro_ratio

    thin = [r for r in recs if is_thin(r)]
    low_micro = [r for r in recs if is_low_micro(r)]

    print(f"\n=== M4b universe data-quality ({n_sym} symbols, lake={args.lake}) ===")
    print("  arms: OHLCV=bits 0-6 ; +micro=bits 0-12 (OHLCV + aggTrades 7-12); perp 13-15 excluded")
    ohlcv_pct = 100 * tot_ohlcv / tot_bars
    micro_pct = 100 * tot_micro / tot_bars
    micro_avail = 100 * tot_micro / tot_ohlcv if tot_ohlcv else 0.0
    print(f"  total bars               : {tot_bars:,}")
    print(f"  OHLCV-usable (post warm) : {tot_ohlcv:,}  ({ohlcv_pct:.2f}% of bars)")
    print(
        f"  +micro-usable            : {tot_micro:,}  ({micro_pct:.2f}% of bars; "
        f"micro-avail {micro_avail:.1f}% of OHLCV)"
    )
    print(
        f"  OHLCV-usable/symbol      : min {ohlcv_counts[0]:,} | med "
        f"{int(med_usable):,} | max {ohlcv_counts[-1]:,}"
    )
    print(f"  stale% (universe avg)    : {statistics.mean(r['stale_pct'] for r in recs):.3f}%")
    print(f"  zero-vol% (univ avg)     : {statistics.mean(r['zerovol_pct'] for r in recs):.3f}%")
    print(
        f"  thin-coin candidates     : {len(thin)} "
        f"(usable < {thin_floor:,.0f} OR stale% > {args.thin_stale_pct})"
    )
    print(f"  low-micro candidates     : {len(low_micro)} (ratio < {args.low_micro_ratio})")

    print("\n  thinnest / lowest-micro 14 (OHLCV-usable ASC):")
    print(
        f"  {'symbol':<14}{'n_bars':>12}{'OHLCV_use':>12}{'micro_use':>12}"
        f"{'m/o':>7}{'stale%':>8}{'zerovol%':>10}  span"
    )
    for r in recs[:14]:
        ratio = _micro_ratio(r)
        flags = "".join(["T" if is_thin(r) else "", "m" if is_low_micro(r) else ""])
        flag = f"  <- {flags}" if flags else ""
        print(
            f"  {r['symbol']:<14}{r['n_bars']:>12,}{r['usable_ohlcv']:>12,}{r['usable_micro']:>12,}"
            f"{ratio:>7.2f}{r['stale_pct']:>7.2f}%{r['zerovol_pct']:>9.3f}%"
            f"  {r['first_date']}→{r['last_date']}{flag}"
        )

    # spot-check: confirm warm-up + zero-vol/vre masking fired per-symbol on the 3 thinnest
    print("\n  spot-check (3 thinnest) — per-symbol warm-up + zero-vol/vre masking fired:")
    for r in recs[:3]:
        warm_dropped = r["n_bars"] - r["usable_ohlcv"]
        micro_dropped = r["usable_ohlcv"] - r["usable_micro"]
        print(
            f"  {r['symbol']:<14} OHLCV warm-up/QA dropped {warm_dropped:,}/{r['n_bars']:,} "
            f"({100 * warm_dropped / r['n_bars']:.1f}%); of OHLCV-usable, micro masked on "
            f"{micro_dropped:,} more ({100 * (1 - _micro_ratio(r)):.1f}%) via zero-vol/vre; "
            f"zero-vol {r['zerovol_pct']:.3f}%"
        )

    if args.md:
        _write_md(Path(args.md), recs, is_thin, is_low_micro)
        print(f"\n  markdown table → {args.md}")
    return 0


def _write_md(path: Path, recs: list[dict], is_thin, is_low_micro) -> None:
    lines = [
        "| symbol | bars | OHLCV-use | +micro-use | m-avail | stale% | zerovol% | span | flags |",
        "|---|--:|--:|--:|--:|--:|--:|---|:-:|",
    ]
    for r in recs:
        flags = ("⚠thin " if is_thin(r) else "") + ("micro↓" if is_low_micro(r) else "")
        lines.append(
            f"| {r['symbol']} | {r['n_bars']:,} | {r['usable_ohlcv']:,} | {r['usable_micro']:,} | "
            f"{_micro_ratio(r):.2f} | {r['stale_pct']:.2f}% | {r['zerovol_pct']:.3f}% | "
            f"{r['first_date']}→{r['last_date']} | {flags.strip()} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
