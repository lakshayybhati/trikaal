"""C-15 (b) — IS THE CAUSAL SAFETY OF THE LAKE WE ALREADY HAVE DEMONSTRATED, OR MERELY SWEPT?

The lake is built, Merkle-anchored and pushed. The guard that admitted it
(``m4b_universe_ingest.py``, ``coverage == 1.0 and n_checks > 0``) proves the sweep EXECUTED, not
that each output surface it covers was capable of failing. This measures the second thing,
per-surface, against the shards already on disk.

NO SAMPLING. The density read did 304,625,181 bars in 2.2 s, so a sample would have been a choice
to know less for no saving. Every symbol, every shard, every bar. "Enough" is "all of it", and the
justification is that sampling was unnecessary rather than that some fraction was sufficient.

TWO QUESTIONS, KEPT SEPARATE — conflating them is how "the data looks fine" becomes "the data is
proven causal", which it is not:
  (1) ARE THE PERSISTED SURFACES NON-DEGENERATE? A constant, all-NaN or fully-masked surface
      cannot carry a leak and cannot show one either. Measured directly from parquet.
  (2) WAS THE SURFACE MEASURABLE IN THE SWEEP THAT ADMITTED THE LAKE? The ingest swept an
      800-bar HEAD SLICE (``--sweep-sample-bars`` default 800) under production
      ``FeatureConfig()``, whose ``effective_n_warm_vol()`` is 1440. ``target_valid[t]`` is True
      only past that warm-up, so on an 800-bar slice it is False everywhere and ``target`` is NaN
      everywhere — both compare equal under every truncation and perturbation, i.e. two of the
      sweep's eight surfaces could not fail. Reconstructed here from the persisted
      ``segment_id``/``bar_open_ms`` and reported for the FULL symbol and for the sweep's own
      800-bar slice.

``target_valid`` is NOT persisted; it is RECONSTRUCTED from its definition in
``targets.build_vol_scaled_targets`` — valid iff ``t+1`` is in the same segment and ``t`` is past
the volatility warm-up. Stated as a reconstruction, never as a read value.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from trikaal.data.config import FeatureConfig

LAKE = Path("processed/universe_bars")
OUT = Path("runs_manifest/m6_c15b_lake_surface_check.json")
SWEEP_SLICE_BARS = 800  # m4b_universe_ingest.py --sweep-sample-bars default
FEATURE_DIMS = range(16)


def main() -> int:
    cfg = FeatureConfig()
    warm = int(cfg.effective_n_warm_vol())
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")

    symbols = [
        r[0]
        for r in con.execute(
            f"SELECT DISTINCT symbol FROM read_parquet('{LAKE}/**/*.parquet', hive_partitioning=1)"
            " ORDER BY symbol"
        ).fetchall()
    ]

    # ---------- (1) per-surface non-degeneracy over the WHOLE lake -------------------------
    aggs = []
    for d in FEATURE_DIMS:
        aggs += [
            f"STDDEV_POP(x_{d}) AS x{d}_sd",
            f"SUM(CASE WHEN NOT isfinite(x_{d}) THEN 1 ELSE 0 END) AS x{d}_nonfinite",
            f"AVG(CAST(m_{d} AS DOUBLE)) AS m{d}_frac_masked",
        ]
    aggs += [
        "STDDEV_POP(sigma) AS sigma_sd",
        "MIN(sigma) AS sigma_min",
        "SUM(CASE WHEN NOT isfinite(sigma) OR sigma <= 0 THEN 1 ELSE 0 END) AS sigma_bad",
        "STDDEV_POP(raw_ret_close) AS raw_ret_sd",
        "STDDEV_POP(raw_range) AS raw_range_sd",
        "COUNT(DISTINCT segment_id) AS n_segments",
        "COUNT(DISTINCT bar_open_ms) AS n_distinct_ts",
        "COUNT(*) AS bars",
    ]
    row = con.execute(
        f"SELECT {', '.join(aggs)} FROM read_parquet('{LAKE}/**/*.parquet', hive_partitioning=1)"
    ).fetchone()
    lake = dict(zip([c[0] for c in con.description], row, strict=True))

    surfaces: dict[str, dict] = {}
    for d in FEATURE_DIMS:
        sd, nf, mf = lake[f"x{d}_sd"], int(lake[f"x{d}_nonfinite"]), lake[f"m{d}_frac_masked"]
        surfaces[f"x_{d}"] = {
            "stddev": sd,
            "n_nonfinite": nf,
            "degenerate": bool(sd is None or sd == 0.0),
        }
        surfaces[f"m_{d}"] = {
            "frac_masked": mf,
            # a mask that is NEVER set carries no information; one ALWAYS set means a dead dim
            "degenerate": bool(mf == 0.0 or mf == 1.0),
        }
    surfaces["sigma"] = {
        "stddev": lake["sigma_sd"],
        "min": lake["sigma_min"],
        "n_nonpositive_or_nonfinite": int(lake["sigma_bad"]),
        "degenerate": bool(lake["sigma_sd"] in (None, 0.0)),
    }
    surfaces["raw"] = {
        "raw_ret_close_sd": lake["raw_ret_sd"],
        "raw_range_sd": lake["raw_range_sd"],
        "degenerate": bool(lake["raw_ret_sd"] in (None, 0.0)),
    }
    surfaces["segment_id"] = {
        "n_segments": int(lake["n_segments"]),
        "degenerate": bool(int(lake["n_segments"]) < 2),
    }
    surfaces["ts"] = {
        "n_distinct_bar_open_ms": int(lake["n_distinct_ts"]),
        "degenerate": bool(int(lake["n_distinct_ts"]) < 2),
    }

    # ---------- (2) target_valid, reconstructed, full symbol vs the sweep's 800-bar slice ----
    full_true = slice_true = 0
    per_symbol_slice_true: dict[str, int] = {}
    for sym in symbols:
        q = f"""
        WITH s AS (
          SELECT segment_id, bar_open_ms,
                 ROW_NUMBER() OVER (PARTITION BY segment_id ORDER BY bar_open_ms) AS k_in_seg,
                 LEAD(segment_id) OVER (ORDER BY bar_open_ms) AS next_seg,
                 ROW_NUMBER() OVER (ORDER BY bar_open_ms) AS k_global
          FROM read_parquet('{LAKE}/symbol={sym}/**/*.parquet')
        )
        SELECT
          SUM(CASE WHEN next_seg = segment_id AND k_in_seg > {warm} THEN 1 ELSE 0 END) AS tv_full,
          SUM(CASE WHEN next_seg = segment_id AND k_in_seg > {warm}
                    AND k_global <= {SWEEP_SLICE_BARS} THEN 1 ELSE 0 END) AS tv_slice
        FROM s
        """
        tv_full, tv_slice = con.execute(q).fetchone()
        full_true += int(tv_full or 0)
        slice_true += int(tv_slice or 0)
        per_symbol_slice_true[sym] = int(tv_slice or 0)

    surfaces["target_valid (RECONSTRUCTED)"] = {
        "formula": "next bar in the same segment AND bar index within segment > "
        f"effective_n_warm_vol()={warm}",
        "n_true_over_the_whole_lake": full_true,
        "frac_true_over_the_whole_lake": full_true / int(lake["bars"]),
        "degenerate_on_the_full_lake": bool(full_true == 0 or full_true == int(lake["bars"])),
        "n_true_within_the_sweeps_own_800_bar_slice": slice_true,
        "symbols_with_any_valid_target_in_the_slice": sum(
            1 for v in per_symbol_slice_true.values() if v > 0
        ),
        "degenerate_in_the_sweep_that_admitted_the_lake": bool(slice_true == 0),
    }

    degenerate = sorted(k for k, v in surfaces.items() if v.get("degenerate"))
    rep = {
        "what": "C-15 (b) — retrospective per-surface non-degeneracy of the lake already on disk",
        "sampling": {
            "shards_sampled": "ALL",
            "symbols": len(symbols),
            "bars": int(lake["bars"]),
            "fraction_of_lake": 1.0,
            "why": (
                "the density read did 304,625,181 bars in 2.2 s, so sampling would have been a "
                "choice to know less for no saving. 'Enough' is 'all of it'."
            ),
        },
        "question_1_are_the_persisted_surfaces_non_degenerate": {
            "surfaces": surfaces,
            "degenerate_surfaces": degenerate,
            "verdict": "CLEAN — every persisted surface varies" if not degenerate else "DEGENERATE",
        },
        "question_2_was_each_surface_MEASURABLE_in_the_sweep_that_admitted_this_lake": {
            "sweep_slice_bars": SWEEP_SLICE_BARS,
            "effective_n_warm_vol": warm,
            "target_valid_true_in_slice": slice_true,
            "verdict": (
                "NO — target/target_valid could not fail in that sweep"
                if slice_true == 0
                else "YES — the slice contained valid targets"
            ),
        },
        "boundary_stated_explicitly": (
            "A non-degenerate surface is a PRECONDITION for a leak check to be meaningful, not "
            "evidence that the surface is leak-free. Question 1 coming back clean means the lake "
            "CAN be checked; it does not mean it HAS been."
        ),
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in rep.items()
                if k != "question_1_are_the_persisted_surfaces_non_degenerate"
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    print("\ndegenerate persisted surfaces:", degenerate or "NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
