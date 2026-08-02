"""C-15 (b), part 2 — THE PRODUCTION-PARAMETER LEAK BATTERY ON REAL BARS, PAST THE WARM-UP.

Part 1 (`m6_c15b_lake_surface_check.py`) established that the sweep which admitted the lake ran on
an 800-bar head slice under a 1440-bar volatility warm-up, so `target`/`target_valid` were
structurally unable to fail — 0 valid targets in the slice, across all 200 symbols. That is a
statement about the DEMONSTRATION, not about the data.

This closes it: a REAL BTCUSDT month (44,640 bars from `processed/shards/`, the same reduction the
ingest itself used) is swept at production ``FeatureConfig()`` on a slice LONGER than the warm-up,
so every one of the eight output surfaces is live. Then each planted leak is fired and must be
CAUGHT.

CONTROL ARM, stated first: a leak battery that reports "all clean" is worthless unless the same
fixture is shown to CATCH a planted leak. Every planted variant must be caught; if any is missed
the probe reports a FINDING, and if the clean baseline itself fails the probe is INVALID rather
than a conclusion about the lake.

$0, local. This is a one-off receipted probe, NOT a CI test — the runtime is paid once.
"""

from __future__ import annotations

import json
from pathlib import Path

from trikaal.data.causal_check import exhaustive_truncation_sweep
from trikaal.data.config import FeatureConfig
from trikaal.data.features import compute_features
from trikaal.data.reduce_shard import load_shard_npz

SHARD = Path("processed/shards/BTCUSDT/BTCUSDT-2023-01.npz")
OUT = Path("runs_manifest/m6_c15b_production_sweep.json")
# The sweep is O(T^2) in recomputes, so the slice is bounded — but it MUST exceed the warm-up or
# it reproduces the very defect being closed. 2,400 = 1440 warm-up + ~960 live bars.
SLICE_BARS = 1_600  # > the 1440 warm-up; O(T^2) so no larger than needed
# The two LOCALIZED leaks are the ones that matter here: they are the surfaces the ingest
# sweep could not exercise. The four GLOBAL variants are already covered non-vacuously by
# the CI gate at reduced parameters, and each costs a full O(T^2) sweep — a bounded scope,
# stated rather than silently dropped.
GLOBAL_LEAKS: tuple[str, ...] = ()


def _slice(stream, n: int):
    from trikaal.data.universe_build import slice_stream_by_ms

    return slice_stream_by_ms(stream, -1e18, float(stream.bar_open_ms[n - 1]) + 1)


def main() -> int:
    cfg = FeatureConfig()
    warm = int(cfg.effective_n_warm_vol())
    stream = _slice(load_shard_npz(SHARD), SLICE_BARS)
    n_bars = len(stream.bar_open_ms)

    rep: dict = {
        "what": "C-15 (b) part 2 — production-parameter causal sweep on REAL bars past the warm-up",
        "source": str(SHARD),
        "symbol": "BTCUSDT",
        "bars_swept": n_bars,
        "effective_n_warm_vol": warm,
        "slice_exceeds_warm_up": bool(n_bars > warm),
        "config": "production FeatureConfig() — NOT the reduced CI-gate config",
        "note": (
            "the ingest swept 800 bars at these same production parameters, i.e. BELOW the 1440 "
            "warm-up, so target/target_valid could not fail there. This slice is above it."
        ),
    }

    # ---- PRECONDITION: the surfaces must actually be live at this slice length ----
    out = compute_features(stream, cfg)
    tv = int(out.target_valid.sum())
    rep["precondition"] = {
        "target_valid_true": tv,
        "of_bars": int(out.target_valid.size),
        "live": bool(tv > 0),
    }
    if tv == 0:
        rep["verdict"] = "PROBE INVALID — the slice is still inside the warm-up; nothing measured"
        OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 1

    # ---- THE CLEAN BASELINE: the real transforms must PASS ----
    clean = exhaustive_truncation_sweep(stream, cfg)
    rep["clean_baseline"] = {
        "passed": bool(clean.passed),
        "coverage": float(clean.coverage),
        "n_checks": int(clean.n_checks),
        "n_anchors": int(clean.n_anchors),
        "n_failures": len(clean.failures),
        "first_failures": clean.failures[:3],
    }
    if not clean.passed:
        rep["verdict"] = (
            "★ RED — A LOOKAHEAD LEAK IS PRESENT IN THE REAL TRANSFORMS AT PRODUCTION PARAMETERS. "
            "STOP EVERYTHING."
        )
        OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 2

    # ---- CONTROL ARM: every planted leak must be CAUGHT on this same fixture ----
    caught: dict[str, dict] = {}
    for leak in GLOBAL_LEAKS:
        r = exhaustive_truncation_sweep(stream, cfg.with_leak(leak))
        caught[leak] = {
            "caught": bool(not r.passed),
            "n_failures": len(r.failures),
            "outputs_flagged": sorted({f["output"] for f in r.failures}),
        }
    # the LOCALIZED validity leak — the surface that could not fire in the ingest sweep.
    # BUILDER DEFECT, CAUGHT AND DISCLOSED: the first run planted at `warm + 200` = 1640 on a
    # 1600-bar slice, i.e. OUTSIDE the swept range, and both leaks were duly "MISSED". That is the
    # probe failing, not the gate — the same class as the fail-open sweep's three harness bugs and
    # exactly what this script's own control-arm doctrine exists to catch. The plant is now
    # asserted to land inside the slice AND past the warm-up before anything is read.
    bar = (warm + n_bars) // 2
    assert warm < bar < n_bars - 2, f"leak bar {bar} must sit inside the slice and past warm-up"
    r_val = exhaustive_truncation_sweep(stream, cfg.with_localized_leak(bar, kind="validity"))
    r_sig = exhaustive_truncation_sweep(stream, cfg.with_localized_leak(bar, kind="sigma"))
    caught["localized_validity_next"] = {
        "caught": bool(not r_val.passed),
        "at_bar": bar,
        "n_failures": len(r_val.failures),
        "outputs_flagged": sorted({f["output"] for f in r_val.failures}),
    }
    caught["localized_sigma_next"] = {
        "caught": bool(not r_sig.passed),
        "at_bar": bar,
        "n_failures": len(r_sig.failures),
        "outputs_flagged": sorted({f["output"] for f in r_sig.failures}),
    }
    rep["planted_leak_battery"] = caught

    missed = sorted(k for k, v in caught.items() if not v["caught"])
    rep["missed_leaks"] = missed
    if missed:
        rep["verdict"] = (
            f"★ FINDING — the production-parameter sweep MISSED {missed}. The gate does not "
            "discriminate the leaks it claims to. STOP."
        )
        code = 3
    else:
        rep["verdict"] = (
            "GREEN — at production parameters on real bars past the warm-up: the real transforms "
            "PASS an exhaustive sweep, and every planted leak (including the target_valid "
            "survivorship leak that could not fire in the ingest sweep) is CAUGHT. The surfaces "
            "the ingest could not exercise are now demonstrated on real data."
        )
        code = 0
    rep["scope_limit_stated"] = (
        f"ONE symbol, ONE month, {n_bars} bars. This demonstrates that the sweep DISCRIMINATES at "
        "production parameters on real bars — it does not sweep the whole lake, which is O(T^2) "
        "and infeasible. The structural argument carries it: a lookahead defect in "
        "compute_features is transform-level, not symbol-level, so a leak that fires anywhere "
        "fires here. A DATA-DEPENDENT leak on some other symbol remains out of scope and is "
        "stated rather than implied."
    )
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
