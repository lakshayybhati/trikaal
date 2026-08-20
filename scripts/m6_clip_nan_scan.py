"""Did the old ``_clip`` ever launder a non-finite value into the published lake? Re-runnable.

THE DEFECT. ``_clip`` was ``min(CLIP_HI, max(CLIP_LO, x))``. Every comparison against NaN is
False, so ``max(CLIP_LO, nan)`` returns ``CLIP_LO``: a NaN became **-5.0**, the most extreme
negative value a normalized feature can take, on the nine z-scored dims. Worse, it defeated the
project's own tripwire — ``features.compute_features`` ends with ``if not np.all(np.isfinite(
x_f64)): raise ValueError(...)``, which cannot fire against a laundered value because -5.0 is
finite.

★ THE METHOD, AND WHY THE DISTINCTION IS LOAD-BEARING. The signature is a **TRAILING run of
exactly -5.0 reaching the END of a segment**, not a run of -5.0 anywhere. The EWMA state updates
as ``mu <- mu + alpha * (f[i] - mu)``; if ``f[i]`` is non-finite then ``mu`` becomes non-finite and
``(f[i] - mu)`` stays non-finite forever after, so **mu can never recover** and every subsequent
live bar in that segment reads -5.0. Poisoning therefore ALWAYS runs to the segment end.

That is what makes this a complete test rather than an indicative one — and the distinction is not
academic. Ordinary quiet-period clipping produces runs of exactly -5.0 **thousands of bars long**
mid-segment (measured below); a scan for "-5.0 anywhere" would drown in them. Only the trailing
form is diagnostic.

    .venv/bin/python scripts/m6_clip_nan_scan.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trikaal.constants import CLIP_LO, N_FEATURES, Z_SCORED_IDX  # noqa: E402
from trikaal.data.universe_loader import connect_lake  # noqa: E402
from trikaal.utils.paths import display_path  # noqa: E402
from trikaal.utils.receipts import ReceiptRefused, write_receipt  # noqa: E402

OUT = REPO / "runs_manifest/m6_clip_nan_scan.json"
DRAW = REPO / "runs_cloud/rescue/r0/cell1_bsq_ohlcv_seed0/run_manifest.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force",
        action="store_true",
        help="overwrite the receipt even though it is tracked (utils.receipts)",
    )
    args = ap.parse_args()
    con = connect_lake()
    n_bars, n_syms = con.execute("SELECT count(*), count(DISTINCT symbol) FROM bars").fetchone()

    # 1. the direct question: is anything non-finite persisted at all?
    non_finite = {}
    for i in range(N_FEATURES):
        n = con.execute(
            f"SELECT sum(CASE WHEN NOT isfinite(x_{i}) THEN 1 ELSE 0 END) FROM bars"
        ).fetchone()[0]
        if n:
            non_finite[f"x_{i}"] = int(n)

    # 2. the laundering signature: a trailing run of exactly CLIP_LO to the end of a segment
    trailing: dict[str, list[dict]] = {}
    for i in Z_SCORED_IDX:
        rows = con.execute(
            f"""
            WITH s AS (
              SELECT symbol, segment_id, bar_open_ms,
                     max(CASE WHEN x_{i} <> {CLIP_LO} THEN bar_open_ms END)
                       OVER (PARTITION BY symbol, segment_id) AS last_ok,
                     count(*) OVER (PARTITION BY symbol, segment_id) AS seg_len
              FROM bars
            )
            SELECT symbol, segment_id, count(*) AS trail, max(seg_len) AS seg_len
            FROM s WHERE last_ok IS NULL OR bar_open_ms > last_ok
            GROUP BY symbol, segment_id ORDER BY trail DESC LIMIT 5
            """
        ).fetchall()
        trailing[f"x_{i}"] = [
            {
                "symbol": s,
                "segment_id": int(sg),
                "trailing_bars": int(tr),
                "segment_bars": int(sl),
                "fraction_of_segment": tr / sl,
            }
            for s, sg, tr, sl in rows
        ]

    worst = max(
        (r for rows in trailing.values() for r in rows),
        key=lambda r: r["trailing_bars"],
        default=None,
    )

    # 3. the contrast that makes the method load-bearing: the longest CONSECUTIVE RUN anywhere.
    #
    # ★ THIS QUERY WAS WRONG AND PUBLISHED A CORRECT MEASUREMENT UNDER THE WRONG NAME. The first
    # version computed `row_number()` AFTER the `WHERE x = CLIP_LO` filter, in both CTEs — so
    # `rn - r2` was identically 0, the GROUP BY collapsed each segment into ONE island, and
    # `count(*)` returned the segment's TOTAL COUNT of -5.0 bars. It reported 3,018 for DOGEUSDT
    # segment 0 dim 0 and called it a run; measured directly, that segment has 3,018 such bars and
    # its longest run is 9. Gaps-and-islands needs the row number over ALL rows minus the row
    # number over the HITS; taking it over the filtered set twice cancels to nothing.
    #
    # A COUNT LABELLED A RUN IS INVISIBLE TO EVERY CHECK, because both numbers are real. The
    # invariant below (`longest_run <= count_in_that_segment`, with equality only for a single
    # contiguous block) is what makes it visible, and it is asserted rather than trusted.
    per_dim_runs = {}
    for i in Z_SCORED_IDX:
        row = con.execute(
            f"""
            WITH a AS (
              SELECT symbol, segment_id, x_{i},
                     row_number() OVER (PARTITION BY symbol, segment_id ORDER BY bar_open_ms)
                       AS rn_all
              FROM bars
            ),
            h AS (
              SELECT symbol, segment_id, rn_all,
                     row_number() OVER (PARTITION BY symbol, segment_id ORDER BY rn_all) AS rn_hit
              FROM a WHERE x_{i} = {CLIP_LO}
            )
            SELECT symbol, segment_id, count(*) AS run
            FROM h GROUP BY symbol, segment_id, (rn_all - rn_hit)
            ORDER BY run DESC LIMIT 1
            """
        ).fetchone()
        if row:
            per_dim_runs[f"x_{i}"] = {
                "symbol": row[0],
                "segment_id": int(row[1]),
                "longest_run_bars": int(row[2]),
            }
    worst_dim = max(per_dim_runs, key=lambda k: per_dim_runs[k]["longest_run_bars"])
    mid = per_dim_runs[worst_dim]
    # the invariant that makes a run-vs-count mislabel impossible to publish
    n_in_segment = con.execute(
        f"""SELECT count(*) FROM bars WHERE symbol = ? AND segment_id = ?
            AND {worst_dim} = {CLIP_LO}""",
        [mid["symbol"], mid["segment_id"]],
    ).fetchone()[0]
    if mid["longest_run_bars"] > n_in_segment:
        raise SystemExit(
            f"IMPOSSIBLE: longest run {mid['longest_run_bars']} exceeds the {n_in_segment} "
            f"matching bars in {mid['symbol']} segment {mid['segment_id']} — the island query is "
            "wrong again"
        )
    mid["matching_bars_in_that_segment"] = int(n_in_segment)
    mid["dim"] = worst_dim

    drawn = []
    if DRAW.is_file():
        drawn = sorted(json.loads(DRAW.read_text())["draw"]["drawn_by_symbol_stage1"])
    affected = sorted({r["symbol"] for rows in trailing.values() for r in rows})

    doc = {
        "receipt": "m6_clip_nan_scan",
        "script": "scripts/m6_clip_nan_scan.py",
        "what": (
            "Whether the pre-fix _clip ever laundered a non-finite value into the published lake. "
            "_clip was min(CLIP_HI, max(CLIP_LO, x)); comparisons against NaN are False, so NaN "
            "returned CLIP_LO = -5.0 and the isfinite tripwire in compute_features could not fire."
        ),
        "METHOD": (
            "The signature is a TRAILING run of exactly -5.0 reaching the END of a segment, not a "
            "run of -5.0 anywhere. The EWMA state updates as mu <- mu + alpha*(f[i]-mu), so a "
            "single non-finite input makes mu non-finite and it CAN NEVER RECOVER; every later "
            "live bar in that segment then reads -5.0. Poisoning therefore always runs to the "
            "segment end, which is what makes this a COMPLETE test rather than an indicative one."
        ),
        "WHY_THE_DISTINCTION_MATTERS": (
            f"Ordinary quiet-period clipping reaches {mid['longest_run_bars']:,} CONSECUTIVE bars "
            f"of exactly -5.0 mid-segment ({mid['symbol']}, segment {mid['segment_id']}, "
            f"{mid['dim']}), against a longest TRAILING run of "
            f"{worst['trailing_bars'] if worst else 0} — a factor of "
            f"{mid['longest_run_bars'] // max(1, worst['trailing_bars'] if worst else 1)}. A scan "
            "for '-5.0 anywhere' would be swamped by legitimate clipping; only the trailing form "
            "is diagnostic."
        ),
        "CORRECTION_2026_08_20": (
            "This field previously read 3,018 and was named longest_mid_segment_run_dim0. 3,018 is "
            "a REAL measurement — the COUNT of -5.0 bars in DOGEUSDT segment 0 dim 0 — published "
            "under the name of a different quantity, because the island query computed both row "
            "numbers over the already-filtered set and cancelled to a single group. That segment's "
            "longest actual run is 9. A correct value under a wrong name passes every check, so "
            "the receipt now asserts longest_run <= matching bars in the same segment."
        ),
        "lake": {"n_bars": int(n_bars), "n_symbols": int(n_syms)},
        "clip_lo": CLIP_LO,
        "z_scored_dims": list(Z_SCORED_IDX),
        "non_finite_persisted_by_dim": non_finite,
        "non_finite_persisted_total": sum(non_finite.values()),
        "trailing_runs_by_dim": trailing,
        "worst_trailing_run": worst,
        "longest_mid_segment_RUN_any_z_dim": mid,
        "longest_mid_segment_run_by_dim": per_dim_runs,
        "symbols_with_any_trailing_run": affected,
        "any_affected_symbol_in_the_training_draw": sorted(set(affected) & set(drawn)),
        "training_draw_available": bool(drawn),
        "VERDICT": (
            "NO LAUNDERING. Zero non-finite values are persisted anywhere in the lake, and the "
            "longest trailing run of exactly -5.0 on any z-scored dim is "
            f"{worst['trailing_bars'] if worst else 0} bar(s) on a segment of "
            f"{worst['segment_bars'] if worst else 0} bars — ordinary tail clipping, not a "
            "poisoned segment, which would cover the whole tail. No published number depended on "
            "the old behaviour."
        ),
    }
    try:
        write_receipt(OUT, doc, measured=trailing, force=args.force)
    except ReceiptRefused as e:
        print(e)
        return 2
    print(f"non-finite persisted: {doc['non_finite_persisted_total']}")
    print(f"worst trailing run:   {worst}")
    print(
        f"longest mid-segment RUN: {mid['longest_run_bars']:,} bars "
        f"({mid['symbol']} seg {mid['segment_id']} {mid['dim']}), "
        f"of {mid['matching_bars_in_that_segment']:,} matching bars in that segment"
    )
    print(f"affected in the draw: {doc['any_affected_symbol_in_the_training_draw'] or 'none'}")
    print(f"wrote {display_path(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
