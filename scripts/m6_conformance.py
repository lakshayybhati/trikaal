"""The money-run conformance gate, standalone — pre-flight Item 8's committed passing run.

    PYTHONPATH=src python3 scripts/m6_conformance.py

Builds THE money-run analysis config exactly as the real eval driver will (money mode, blocks
1–5 continuous headline, no cap, per-symbol deciles from the committed artifact, pinned κ/seeds)
and asserts it against the pre-registered §3a surface via ``trikaal.eval.conformance``. ANY
divergence → the full failure list on stderr and a NON-ZERO exit. A passing run writes the
deterministic receipt ``runs_manifest/m6_conformance_run.json`` (committed as Item-8 evidence).

The real eval driver's contract: construct its per-seed configs the same way and call
``assert_conformance(cfg, symbols=…, seeds=…)`` FIRST — before loading any checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from trikaal.eval.conformance import (
    MDE_INPUTS,
    PINNED_KAPPAS,
    PINNED_SEEDS,
    PINNED_SYMBOLS_SHA256,
    conformance_failures,
    money_config,
    symbols_sha256,
)
from trikaal.eval.xsection import primary_region_grid_ms

OUT = Path("runs_manifest/m6_conformance_run.json")


def main() -> int:
    mde = json.loads(MDE_INPUTS.read_text())
    symbols = list(mde["symbols_sampled"])
    cfg = money_config()
    fails = conformance_failures(cfg, symbols=symbols, seeds=PINNED_SEEDS)
    if fails:
        print("CONFORMANCE FAIL — the config diverges from the §3a surface:", file=sys.stderr)
        for f in fails:
            print(f"  - {f}", file=sys.stderr)
        return 1
    grid = primary_region_grid_ms(cfg)
    receipt = {
        "conformance": "PASS",
        "surface": {
            "window": mde["window"],
            "train_frac": mde["train_frac"],
            "primary_region_ms": mde["primary_region_ms"],
            "headline_grid": {
                "start_ms": int(grid[0]),
                "end_ms_excl": int(mde["primary_region_ms"][1]),
                "n_periods_h15": int(grid.size),
            },
            "symbols_sha256": symbols_sha256(symbols),
            "n_symbols": len(symbols),
            "kappas": list(PINNED_KAPPAS),
            "seeds": list(PINNED_SEEDS),
            "cap_per_symbol": None,
            "headline_cost": cfg.headline_cost,
            "spread_deciles": "per-symbol (runs_manifest/m6_spread_deciles.json)",
            "bootstrap": {"B": 10_000, "seed": 20260704, "ci": "percentile", "alpha": 0.05},
        },
        "checks_run": [
            "window+train_frac vs m6_mde_inputs.json",
            "money mode + blocks-1-5 continuous headline grid == primary_region_ms (VAL excluded)",
            "40-symbol set == symbols_sampled @ sha256 " + PINNED_SYMBOLS_SHA256[:16] + "…",
            "kappa grid == (1.0, 1.5, 2.0, 3.0)",
            "seeds == (0, 1, 2)",
            "cap_per_symbol ABSENT",
            "per-symbol spread deciles in use, == committed artifact, non-flat",
            "shuffle_micro train/eval seed-threading bit-identical (seeds 0,1,2 probed)",
            "headline cost 0.30%",
            "bootstrap constants B=10000 / seed=20260704 / alpha=0.05",
        ],
    }
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"CONFORMANCE PASS — all {len(receipt['checks_run'])} checks; receipt → {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
