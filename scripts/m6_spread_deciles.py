"""Per-symbol spread deciles from TRAIN-REGION liquidity → runs_manifest/m6_spread_deciles.json.

    PYTHONPATH=src python3 scripts/m6_spread_deciles.py

Audit item 4 (2026-07-06): the eval driver's modeled cost used the flat "major" (1bp) half-spread
for EVERY symbol — under-costing the long tail by ~9bp of half-spread in θ and the modeled-cost
secondary. This script assigns each lake symbol a SPREAD_DECILE_FRAC tier:

* **majors** = BTC/ETH — the §8.B.2 spec constant (pre-data, by name);
* the rest rank by the **causal train-region liquidity proxy**: the active-minute share over the
  train region's FINAL 180 days on a calendar-common denominator (present ∧ non-stale ∧
  non-zero-volume bars / window calendar minutes). The calendar-common window makes the rank
  era-fair (the naive full-train usable share ranked 2023 listings above 2021 majors — a
  data-era artifact); counting no-trade minutes breaks the top-cluster tie (155 symbols tie on
  stale-free coverage alone; 8 tie once zero-volume minutes count, and the order is the
  recognizable liquidity order, not the alphabet). Raw ADV is NOT reconstructible in-lake (the
  features are causally z-scored per symbol); this is the strongest in-lake liquidity ordering.
  Ties → train bar count → symbol name;
* ranks 1..18 → "top20" (3bp); the rest → "tail" (10bp); symbols with ZERO train-region bars
  (post-boundary listings) → "tail" by the conservative (0.0, 0) default.

The 0.30 % flat HEADLINE netting is untouched — deciles enter only θ = κ·c_modeled and the
modeled-cost secondary. The artifact is deterministic (sorted keys) and content-hashed; the money
run loads it via ``costs.load_spread_deciles`` and the conformance gate asserts it is in use.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from trikaal.data.universe_loader import (
    calendar_boundary_ms,
    connect_lake,
    train_region_liquidity_stats,
)
from trikaal.eval.costs import SPREAD_DECILE_FRAC, assign_spread_deciles

LAKE = Path("processed/universe_bars")
WINDOW = ("2021-01-01", "2025-01-01")
TRAIN_FRAC = 0.7
OUT = Path("runs_manifest/m6_spread_deciles.json")


def main() -> int:
    if not LAKE.exists():
        print("missing lake — run the M4b ingest first")
        return 1
    boundary = calendar_boundary_ms(*WINDOW, TRAIN_FRAC)
    con = connect_lake(LAKE)
    stats = train_region_liquidity_stats(con, boundary)  # causal: asserts max ts < boundary
    all_symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM bars").fetchall()]
    for s in all_symbols:  # post-boundary listings: no train evidence → conservative tail
        stats.setdefault(s, (0.0, 0))
    deciles = assign_spread_deciles(stats)
    tiers = {t: sum(1 for d in deciles.values() if d == t) for t in SPREAD_DECILE_FRAC}
    doc = {
        "basis": (
            "TRAIN-REGION ONLY (bar_open_ms < boundary; causality asserted in "
            "train_region_liquidity_stats). Proxy = active-minute share over the train "
            "region's final 180 days on a calendar-common denominator (present AND non-stale "
            "AND non-zero-volume bars / window calendar minutes) — era-fair by construction, "
            "no-trade minutes discriminate the top cluster. Majors pinned by name per spec "
            "§8.B.2 (BTC/ETH); next 18 by (proxy, n_bars_train, symbol) rank -> top20; rest "
            "-> tail; zero-train-bar symbols (post-boundary listings) -> tail. Raw ADV is "
            "not reconstructible from the causally z-scored lake features."
        ),
        "window": list(WINDOW),
        "train_frac": TRAIN_FRAC,
        "boundary_ms": boundary,
        "boundary_utc": datetime.fromtimestamp(boundary / 1000, tz=UTC).isoformat(),
        "spread_decile_frac": SPREAD_DECILE_FRAC,
        "n_symbols": len(deciles),
        "tier_counts": tiers,
        "deciles": {
            s: {
                "decile": deciles[s],
                "spread_frac": SPREAD_DECILE_FRAC[deciles[s]],
                "active_minute_share_train": round(stats[s][0], 6),
                "n_bars_train": stats[s][1],
            }
            for s in sorted(deciles)
        },
    }
    OUT.parent.mkdir(exist_ok=True)
    payload = json.dumps(doc, indent=2, sort_keys=True)
    OUT.write_text(payload)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    print(f"[deciles] {len(deciles)} symbols → {OUT} (sha256:{digest[:16]}…)")
    print(f"[deciles] tiers: {tiers}")
    top = sorted(deciles, key=lambda s: (deciles[s] != "major", deciles[s] != "top20", s))[:22]
    print(f"[deciles] major+top20: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
