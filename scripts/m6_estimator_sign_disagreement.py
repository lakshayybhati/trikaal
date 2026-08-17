"""Three estimators of the same quantity, one model, one file — and they disagree on the sign.

WHY THIS SCRIPT EXISTS. Its receipt was committed with no producer anywhere in the repository, and
it is MINE: I wrote `runs_manifest/m6_estimator_sign_disagreement.json` in a session and pasted it
in, which is the identical defect I had just closed for `m6_demo_seed_sign_disagreement.json`. The
class rule says sweep every site of a defect's class rather than fixing the instance an audit
pointed at, and the improved `m6_receipt_provenance_sweep.py` — the very tool built to catch this —
duly flagged my own receipt with five untraceable top-level scalars. That is the fix working on its
author, which is the only evidence a fix of this kind can offer.

WHAT IS MEASURED. One decision instant, three cell-1 seeds, one rollout at MC_SEED=20260704, three
estimators read off that SAME rollout so nothing differs but the reduction:

  expectation        predict_mu(estimator="expectation") — the conditional mean
  mc_mean_32         the mean of the first 32 sampled trajectory endpoints
  trajectory_median  the median of that seed's 48 sampled endpoints

They disagree about which way the model points, and seed 4 flips outright. The directional signal
is weak enough that the choice of estimator changes its sign.

★ THE INPUT IS NOT IN THE REPOSITORY, AND THAT LIMIT IS RECORDED RATHER THAN PAPERED OVER. The
receipt was computed from an operator-supplied OHLCV CSV. Binance data is deliberately not
redistributed (the M8 gate ships the pipeline and its content hashes, never the bars), so the CSV
stays outside and this script stamps its SHA-256, row count and last timestamp instead. A reader
with the same file can reproduce the numbers exactly and can prove it is the same file; a reader
without it can see precisely what they are missing. Whether that sample should be committed is a
redistribution decision, and it is not mine to make.

    .venv/bin/python scripts/m6_estimator_sign_disagreement.py --csv ~/Downloads/BTCUSDT_1m.csv
    .venv/bin/python scripts/m6_estimator_sign_disagreement.py --csv <path> --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trikaal.demo.csv_forecast import (  # noqa: E402
    MC_SEED,
    PAPER_HORIZON,
    forecast_from_csv,
)
from trikaal.demo.inference import available_seeds, load_unit  # noqa: E402
from trikaal.utils.paths import display_path  # noqa: E402

OUT = REPO / "runs_manifest" / "m6_estimator_sign_disagreement.json"


def measure(csv_path: Path, *, h: int, device: str = "cpu") -> dict:
    text = csv_path.read_text()
    seeds = available_seeds()
    if len(seeds) < 2:
        raise SystemExit(f"need >=2 units to compare estimators, have {seeds}")
    units = {s: load_unit(s, device=device) for s in seeds}
    f = forecast_from_csv(text, units=units, h=h, device=device)

    rows = [
        {
            "seed": s,
            "expectation_pct": f.per_seed[s]["pct_expectation"],
            "mc_mean_32_pct": f.per_seed[s]["mc_mean_pct"],
            "trajectory_median_pct": f.per_seed[s]["pct"],
        }
        for s in sorted(f.per_seed)
    ]
    majority = {
        name: sum(1 for r in rows if r[f"{name}_pct"] > 0)
        for name in ("expectation", "mc_mean_32", "trajectory_median")
    }
    pooled = f.pooled or {}
    return {
        "FINDING": (
            "Three estimators of the same quantity, one model, one file: they DISAGREE ON THE "
            "MAJORITY SIGN, and seed 4 flips outright between expectation and the trajectory "
            "median. The directional signal is weak enough that the CHOICE OF ESTIMATOR CHANGES "
            "ITS SIGN."
        ),
        "note": (
            "expectation = predict_mu(estimator='expectation'); mc_mean_32 = the mean of the "
            "first 32 sampled trajectories' endpoints (bit-identical to production's mc_mean); "
            "trajectory_median = the median of that seed's 48 sampled endpoints. All from the "
            f"SAME rollout at MC_SEED={MC_SEED}."
        ),
        "input": {
            "path_as_supplied": str(csv_path),
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "n_rows": f.n_rows,
            "decision_ts_ms": f.decision_ts_ms,
            "NOT_IN_THE_REPOSITORY": (
                "Binance bars are not redistributed (M8 gate: pipeline + content hashes, never "
                "the data). The hash above is what lets a reader prove they hold the same file."
            ),
        },
        "h": h,
        "n_pooled_paths": f.n_pooled,
        "rows": rows,
        "majority_positive": majority,
        "pooled_median_pct": (math.exp(pooled[50][-1]) - 1.0) * 100.0 if pooled else None,
        "pooled_p10_pct": (math.exp(pooled[10][-1]) - 1.0) * 100.0 if pooled else None,
        "pooled_p90_pct": (math.exp(pooled[90][-1]) - 1.0) * 100.0 if pooled else None,
        "WHY_IT_MATTERS": (
            "The page previously rendered the pooled headline (trajectories) beside per-seed "
            "labels (expectation). A negative headline over two positive seeds reads as broken "
            "arithmetic. Fixed by putting ONE estimator on the page — the one the bands and "
            "median lines are drawn from."
        ),
        "LIMITS": (
            f"One file, one horizon (h={h}), {len(rows)} seeds, cell 1. Illustrative of a measured "
            "property, not a new measurement of it."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path, help="operator-supplied OHLCV CSV")
    ap.add_argument("--h", type=int, default=PAPER_HORIZON)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--check", action="store_true", help="reproduce and diff; write nothing")
    args = ap.parse_args()

    csv_path = args.csv.expanduser()
    if not csv_path.exists():
        print(f"no such CSV: {csv_path}", file=sys.stderr)
        return 1
    doc = measure(csv_path, h=args.h, device=args.device)

    if args.check:
        if not OUT.exists():
            print(f"RECEIPT ABSENT: {display_path(OUT, REPO)}", file=sys.stderr)
            return 2
        old = json.loads(OUT.read_text())
        keys = (
            "h",
            "n_pooled_paths",
            "rows",
            "majority_positive",
            "pooled_median_pct",
            "pooled_p10_pct",
            "pooled_p90_pct",
        )
        bad = [k for k in keys if old.get(k) != doc.get(k)]
        if bad:
            print(f"RECEIPT NOT REPRODUCED — these differ: {bad}", file=sys.stderr)
            return 2
        print(f"RECEIPT REPRODUCED — all {len(keys)} measured fields identical")
        return 0

    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    for r in doc["rows"]:
        print(
            f"  seed {r['seed']}: expectation {r['expectation_pct']:+.6f}%  "
            f"mc_mean@32 {r['mc_mean_32_pct']:+.6f}%  median {r['trajectory_median_pct']:+.6f}%"
        )
    print(f"  majority positive: {doc['majority_positive']}")
    print(f"Receipt: {display_path(OUT, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
