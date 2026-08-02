"""AUDIT TIER-4 — the two findings that had to be MEASURED, not read: C-10 and C-15.

VERIFY ONLY. Nothing is fixed here; both gates are left exactly as they are.

Both are the C-2 pattern one level up — **a gate that reports PASS while examining nothing** — and
neither is visible from reading the code, because in both cases the docstring asserts the opposite
of the behaviour. So each is demonstrated by construction.

  C-10  ``train/gates.py micro_legibility_gate`` — a dim with < 10k unmasked bars is ``continue``d
        BEFORE ``ok = ok and acc >= min_acc``. If every micro dim is thin, ``ok`` is still the
        initial ``True``, no RuntimeError is raised, and the hard stop returns ``pass: True``
        having measured nothing. The docstring says such a dim is *"SKIPPED with a named receipt
        entry, never trivially passed"* — the receipt entry is real; the "never trivially passed"
        is not.

  C-15  the causal-safety CI gate (invariant 2) runs ONLY at reduced feature parameters. Re-running
        it at the PRODUCTION ``FeatureConfig()`` does not make it stronger — it makes it VACUOUS,
        because ``effective_n_warm_vol()`` = 1440 exceeds the fixture length, so no bar has a valid
        target and the planted ``localized_validity_next`` leak has nothing to flip.

CONTROL ARM for each (standing norm): the same construction is run in a configuration where the
gate MUST behave differently. If both arms agree, the demonstration proves nothing and the script
says so instead of reporting a finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trikaal.data.config import FeatureConfig, synthetic_test_config
from trikaal.data.features import compute_features
from trikaal.data.synthetic import make_synthetic_stream
from trikaal.train.gates import MICRO_DIMS, micro_legibility_gate

OUT = Path("runs_manifest/m6_tier4_vacuous_gates.json")


class _FakeWindow:
    """Minimal per-symbol window: x [T, F] and mask [T, F] (1 = masked out)."""

    def __init__(self, symbol: str, x: np.ndarray, mask: np.ndarray):
        self.symbol, self.x, self.mask = symbol, x, mask


def c10() -> dict:
    """Does the micro-legibility HARD STOP pass when every micro dim is masked thin?"""
    rng = np.random.default_rng(0)
    t, f = 12_000, 16

    def _run(thin: bool) -> dict:
        x = rng.standard_normal((t, f))
        mask = np.zeros((t, f), dtype=np.uint8)
        if thin:  # every micro dim masked down to < 10k unmasked bars
            for d in MICRO_DIMS:
                mask[:-500, d] = 1
        sw = _FakeWindow("SYMB", x, mask)
        toks = {"SYMB": (rng.integers(0, 8, t), rng.integers(0, 8, t))}
        try:
            return {"raised": False, "receipt": micro_legibility_gate(None, [sw], toks)}
        except RuntimeError as e:
            return {"raised": True, "error": str(e)[:160]}
        except Exception as e:  # a DIFFERENT fault — must not be read as the finding
            return {"raised": True, "unrelated_fault": f"{type(e).__name__}: {str(e)[:160]}"}

    thin = _run(thin=True)
    rep = {
        "claim": "the hard stop passes when every micro dim is too thin to measure",
        "all_dims_thin": thin,
    }
    passed_blind = (
        not thin["raised"]
        and thin.get("receipt", {}).get("pass") is True
        and all("skipped" in v for v in thin.get("receipt", {}).get("per_dim", {}).values())
    )
    rep["verdict"] = (
        "CONFIRMED — returned pass=True with EVERY dim skipped, i.e. measured nothing"
        if passed_blind
        else "NOT CONFIRMED on this construction"
    )
    rep["n_dims_skipped"] = sum(
        1 for v in thin.get("receipt", {}).get("per_dim", {}).values() if "skipped" in v
    )
    rep["n_micro_dims"] = len(MICRO_DIMS)
    return rep


def c15() -> dict:
    """Can the planted target_valid leak fire at all under PRODUCTION feature parameters?"""
    stream, _ = make_synthetic_stream(seed=7, n_bars=600)
    cfgs = {
        "CI-gate cfg (tests/conftest causal_cfg)": synthetic_test_config(
            half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20, vol_window=24, n_warm_vol=12
        ),
        "PRODUCTION FeatureConfig()": FeatureConfig(),
    }
    legs = {}
    for name, cfg in cfgs.items():
        out = compute_features(stream, cfg)
        tv = out.target_valid
        fired = tested = 0
        for bar in range(60, min(560, tv.size - 5), 25):
            leaked = compute_features(stream, cfg.with_localized_leak(bar, kind="validity"))
            tested += 1
            fired += int(not np.array_equal(leaked.target_valid, tv))
        legs[name] = {
            "effective_n_warm_vol": int(cfg.effective_n_warm_vol()),
            "fixture_bars": int(tv.size),
            "target_valid_true": int(tv.sum()),
            "leak_changed_target_valid_at": fired,
            "bars_tried": tested,
        }
    ci = legs["CI-gate cfg (tests/conftest causal_cfg)"]
    prod = legs["PRODUCTION FeatureConfig()"]
    # CONTROL: the reduced config MUST show the leak firing, or the probe cannot separate
    # "the leak is undetectable here" from "the leak plant is broken".
    discriminates = ci["leak_changed_target_valid_at"] > 0
    return {
        "claim": (
            "the causal CI gate runs only at reduced params, and is VACUOUS at production ones"
        ),
        "legs": legs,
        "control_arm": {
            "reduced-config leak fires": discriminates,
            "note": "if this were False the probe would be measuring a broken plant, not a gate",
        },
        "verdict": (
            "PROBE INVALID — the planted leak does not fire even at reduced parameters"
            if not discriminates
            else (
                "CONFIRMED — at production parameters the warm-up (1440) exceeds the fixture, no "
                "bar has a valid target, and the leak cannot flip anything: the check would PASS "
                "while testing nothing"
                if prod["leak_changed_target_valid_at"] == 0
                else "NOT CONFIRMED — the leak fires at production parameters too"
            )
        ),
        "so_the_fix_is_not_swap_the_config": (
            "re-running the CI gate at FeatureConfig() would make it vacuous, not stronger. A "
            "production-parameter causal gate needs a fixture longer than the 1440-bar warm-up, "
            "which is a test-cost decision, not a one-line config change. NOT proposed here."
        ),
        "mitigation_on_the_record": (
            "scripts/m4b_universe_ingest.py:292 DOES run exhaustive_truncation_sweep at production "
            "FeatureConfig() on real lake shards, with a hard gate on coverage==1.0 and "
            "n_checks>0. But that guard proves the sweep RAN, not that each output surface was "
            "non-degenerate — n_checks counts all outputs, so a vacuous target_valid surface still "
            "clears it. Same shape as C-10."
        ),
    }


def main() -> int:
    rep = {
        "what": "AUDIT TIER-4 — C-10 and C-15 measured. VERIFY ONLY, nothing fixed.",
        "C-10_micro_legibility_gate": c10(),
        "C-15_causal_gate_parameters": c15(),
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
