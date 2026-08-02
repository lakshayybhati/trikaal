"""C-10 MATERIALITY — can the vacuous-pass actually fire on the real lake?

Existence and materiality are different questions. `m6_tier4_vacuous_gates.json` proved the defect
EXISTS (the gate returns pass=True having measured nothing). This answers whether the input that
triggers it — a micro dim with fewer than ``THIN_FLOOR`` unmasked bars — occurs in production.

It also answers the second leg: `id_legibility_sign_acc` reads the FIRST 150,000 rows of a
per-symbol concatenation and splits 80/20 CONTIGUOUSLY, so on the real lake the "sample" is the
head of the symbol list, not a sample of it. This reports what that head actually covers.

MEMORY: aggregated in DuckDB with a hard memory_limit, reading only ``m_7..m_12``. A lazy
whole-lake read here is the ~28 GiB-vs-16 GiB trap that cost a rental once already.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

LAKE = Path("processed/universe_bars")
OUT = Path("runs_manifest/m6_c10_micro_density.json")
MICRO_DIMS = (7, 8, 9, 10, 11, 12)
THIN_FLOOR = 10_000  # gates.py: `if v.size < 10_000` -> skipped
HEAD_N = 150_000  # id_legibility_sign_acc default n


def main() -> int:
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")
    cols = ", ".join(
        f"SUM(CASE WHEN m_{d} = 0 THEN 1 ELSE 0 END) AS unmasked_{d}" for d in MICRO_DIMS
    )
    q = f"""
        SELECT symbol, COUNT(*) AS bars, {cols}
        FROM read_parquet('{LAKE}/**/*.parquet', hive_partitioning=1)
        GROUP BY symbol ORDER BY symbol
    """
    rows = con.execute(q).fetchall()
    names = [d[0] for d in con.description]
    recs = [dict(zip(names, r, strict=True)) for r in rows]

    total_bars = sum(int(r["bars"]) for r in recs)
    per_dim_total = {d: sum(int(r[f"unmasked_{d}"]) for r in recs) for d in MICRO_DIMS}
    per_dim_min = {d: min(int(r[f"unmasked_{d}"]) for r in recs) for d in MICRO_DIMS}
    thin_symbols = {
        d: sorted(r["symbol"] for r in recs if int(r[f"unmasked_{d}"]) < THIN_FLOOR)
        for d in MICRO_DIMS
    }
    zero_symbols = {
        d: sorted(r["symbol"] for r in recs if int(r[f"unmasked_{d}"]) == 0) for d in MICRO_DIMS
    }

    # what the head-150k sample actually covers, in symbol order (the gate's concatenation order)
    cum, head_symbols = 0, []
    for r in sorted(recs, key=lambda r: r["symbol"]):
        if cum >= HEAD_N:
            break
        head_symbols.append(r["symbol"])
        cum += int(r["unmasked_9"])  # dim 9 = the TFI-class dim the whole finding is about

    rep = {
        "what": "C-10 materiality — per-dim unmasked bar counts for micro dims 7-12, real lake",
        "lake": str(LAKE),
        "n_symbols": len(recs),
        "total_bars": total_bars,
        "thin_floor_in_gate": THIN_FLOOR,
        "per_dim_total_unmasked": {str(d): per_dim_total[d] for d in MICRO_DIMS},
        "per_dim_min_across_symbols": {str(d): per_dim_min[d] for d in MICRO_DIMS},
        "symbols_below_thin_floor_per_dim": {str(d): thin_symbols[d] for d in MICRO_DIMS},
        "symbols_with_zero_unmasked_per_dim": {str(d): zero_symbols[d] for d in MICRO_DIMS},
        "n_symbols_below_floor_per_dim": {str(d): len(thin_symbols[d]) for d in MICRO_DIMS},
        "head_sample_coverage": {
            "gate_reads_first_n_rows": HEAD_N,
            "symbols_covered_at_dim9": len(head_symbols),
            "of_total_symbols": len(recs),
            "first_symbols": head_symbols[:6],
            "reading": (
                f"the 150k window spans {len(head_symbols)} of {len(recs)} symbols in sorted "
                "order; the remaining symbols contribute NOTHING to the legibility measurement, "
                "and the 80/20 split is contiguous within that head"
            ),
        },
        "per_symbol": recs,
    }
    rep["verdict_can_c10_fire"] = (
        "NO on the CONCATENATED counts the gate actually applies the floor to — every dim clears "
        f"{THIN_FLOOR} by orders of magnitude, so the all-thin and partial-thin cases do not arise "
        "from the current 200-symbol lake"
        if all(v >= THIN_FLOOR for v in per_dim_total.values())
        else "YES — at least one dim falls below the floor on the concatenated counts"
    )
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    printable = {k: v for k, v in rep.items() if k != "per_symbol"}
    print(json.dumps(printable, indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
