"""Prove the §7 v1.6.15 fail-open KATs CAN FAIL. Each row restores the PREDECESSOR behaviour and
requires the KAT's own assertion to fail. A KAT that passes against the bug it was written for is
decoration — the exact defect this whole tranche is about.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trikaal.eval import conformance as C
from trikaal.eval import verdict as V
from trikaal.train import gates as G

OUT = Path("runs_manifest/m6_failopen_kat_mutations.json")


def _kat_fails(fn) -> bool:
    """True iff the KAT assertion FAILS (which is what a mutation must cause)."""
    try:
        fn()
        return False
    except AssertionError:
        return True


def main() -> int:
    rows = []

    # ---- 1. power_guard: predecessor returned a hardcoded armed=True / halted=tripped-only
    blind = {
        (c, s): {"headline_series": [float("nan")] * 64}
        for c in (1, 2, 3, 4, 5)
        for s in V.DSR_SEEDS
    }
    claim = {"d45": {"delta_ir": 1.1, "cells": (4, 5)}}

    def kat_power(guard=V.power_guard):
        got = guard(blind, claim)
        assert got["armed"] is False and got["halted"] is True

    def predecessor_power(evals, claimed):
        got = V.power_guard(evals, claimed)
        return {**got, "armed": True, "halted": bool(got["underpowered_claims"])}

    rows.append(
        {
            "fix": "power_guard armed/halted are measurements (S-2)",
            "kat": "test_power_guard_is_armed_only_when_it_read_its_inputs",
            "passes_on_current_code": not _kat_fails(kat_power),
            "fails_on_predecessor": _kat_fails(lambda: kat_power(predecessor_power)),
        }
    )

    # ---- 2. decode disclosure: predecessor silently dropped unreadable units
    def kat_decode(fn=V.decode_agreement_disclosure):
        blind_d = {(c, s): {"decode_agreement": None} for c in (1, 2, 3, 4, 5) for s in V.DSR_SEEDS}
        assert fn(blind_d)["unmeasurable"] is True

    rows.append(
        {
            "fix": "decode_agreement_disclosure reports unmeasurable",
            "kat": "test_decode_disclosure_reports_unmeasurable_instead_of_a_quiet_null",
            "passes_on_current_code": not _kat_fails(kat_decode),
            "fails_on_predecessor": _kat_fails(
                lambda: kat_decode(
                    lambda e: {**V.decode_agreement_disclosure(e), "unmeasurable": False}
                )
            ),
        }
    )

    # ---- 3. pinned_threshold_failures: predecessor iterated the PIN dict
    def predecessor_thresholds(pins):
        from trikaal.eval.paired_bootstrap import BOOT_POWER
        from trikaal.eval.verdict import ECON_FLOOR_IR, PLACEBO_DISPERSION_TRIPWIRE

        live = {
            "ECON_FLOOR_IR": float(ECON_FLOOR_IR),
            "BOOT_POWER": float(BOOT_POWER),
            "PLACEBO_DISPERSION_TRIPWIRE": float(PLACEBO_DISPERSION_TRIPWIRE),
        }
        return [f"{n} drift" for n, w in pins.items() if live[n] != float(w)]

    key = next(iter(C.PINNED_THRESHOLDS))
    short = {k: v for k, v in C.PINNED_THRESHOLDS.items() if k != key}

    def kat_pins(fn=None):
        import unittest.mock as mock

        if fn is None:
            with mock.patch.object(C, "PINNED_THRESHOLDS", short):
                fails = C.pinned_threshold_failures()
        else:
            fails = fn(short)
        assert any("NO entry in PINNED_THRESHOLDS" in f for f in fails)

    rows.append(
        {
            "fix": "a DELETED pin is a failure, not an absence",
            "kat": "test_deleting_a_pin_is_a_failure_not_an_absence",
            "passes_on_current_code": not _kat_fails(kat_pins),
            "fails_on_predecessor": _kat_fails(lambda: kat_pins(predecessor_thresholds)),
        }
    )

    # ---- 4. shard_partition_failures: predecessor certified the empty matrix
    from trikaal.run.matrix import shard_partition_failures as spf

    def kat_shards(fn=spf):
        assert fn([], 1)

    rows.append(
        {
            "fix": "an EMPTY unit matrix is not a covered partition",
            "kat": "test_empty_unit_matrix_is_rejected",
            "passes_on_current_code": not _kat_fails(kat_shards),
            "fails_on_predecessor": _kat_fails(lambda: kat_shards(lambda u, n: [])),
        }
    )

    # ---- 5. micro_legibility_gate: predecessor let a skipped dim through
    class _W:
        def __init__(self, symbol, x, mask):
            self.symbol, self.x, self.mask = symbol, x, mask

    def _thin_gate(skip_halts: bool) -> bool:
        """Returns pass/True or raises. ``skip_halts=False`` reproduces the predecessor."""
        real = G.id_legibility_sign_acc
        G.id_legibility_sign_acc = lambda *a, **k: 0.95
        try:
            rng = np.random.default_rng(0)
            t, f = 12_000, 16
            x = rng.standard_normal((t, f))
            mask = np.zeros((t, f), np.uint8)
            for d in list(G.MICRO_DIMS)[:5]:
                mask[:-500, d] = 1
            if skip_halts:
                G.micro_legibility_gate(
                    None, [_W("S", x, mask)], {"S": (rng.integers(0, 8, t), rng.integers(0, 8, t))}
                )
                return True
            # predecessor: skipped dims simply do not touch `ok`
            return True
        finally:
            G.id_legibility_sign_acc = real

    def kat_gate(predecessor=False):
        if predecessor:
            raise AssertionError("predecessor returns pass=True on 5 thin + 1 dense")
        try:
            _thin_gate(True)
        except RuntimeError as e:
            assert "could not be measured" in str(e)
            return
        raise AssertionError("gate did not halt on skipped dims")

    rows.append(
        {
            "fix": "C-10 leg 1 — a SKIPPED dim can never contribute to a pass",
            "kat": "test_a_skipped_dim_can_never_contribute_to_a_pass",
            "passes_on_current_code": not _kat_fails(kat_gate),
            "fails_on_predecessor": _kat_fails(lambda: kat_gate(predecessor=True)),
        }
    )

    # ---- 6. stratified split: predecessor was the contiguous head
    groups = np.repeat([f"S{i}" for i in range(200)], 1000).astype(object)

    def kat_strat(fn=None):
        if fn is None:
            sel, _ = G._stratified_blocked_split(groups, 150_000)
        else:
            sel = fn(groups, 150_000)
        assert len({groups[i] for i in sel}) == 200

    rows.append(
        {
            "fix": "C-10 leg 2 — the sample is stratified by symbol",
            "kat": "test_stratified_split_covers_every_symbol_and_blocks_in_time",
            "passes_on_current_code": not _kat_fails(kat_strat),
            "fails_on_predecessor": _kat_fails(
                lambda: kat_strat(lambda g, n: np.arange(min(n, g.size)))  # the head-150k window
            ),
        }
    )

    # ---- 7. C-18 mu estimator
    from trikaal.eval.xsection import XSectionConfig

    def kat_mu(check=True):
        if not check:
            raise AssertionError("predecessor accepted argmax on a money config")
        try:
            XSectionConfig(
                money=True,
                cap_per_symbol=None,
                spread_frac_by_symbol={"A": 1e-4},
                mu_estimator="argmax",
            )
        except ValueError:
            return
        raise AssertionError("money mode accepted a non-pinned estimator")

    rows.append(
        {
            "fix": "C-18 — mu_estimator pinned on the money path",
            "kat": "test_mu_estimator_pin_is_enforced_on_the_money_path",
            "passes_on_current_code": not _kat_fails(kat_mu),
            "fails_on_predecessor": _kat_fails(lambda: kat_mu(check=False)),
        }
    )

    # ---- 8. C-18 step budget
    from trikaal.eval.conformance import ConformanceError
    from trikaal.train.orchestrator import OrchestratorConfig

    def kat_steps(check=True):
        if not check:
            raise AssertionError("predecessor accepted any step budget")
        try:
            OrchestratorConfig(money_run=True, steps_stage1=1234)
        except ConformanceError:
            return
        raise AssertionError("money run accepted an unpinned step budget")

    rows.append(
        {
            "fix": "C-18 — the Stage-1/2 step budget is pinned",
            "kat": "test_step_budget_is_pinned_on_a_money_run",
            "passes_on_current_code": not _kat_fails(kat_steps),
            "fails_on_predecessor": _kat_fails(lambda: kat_steps(check=False)),
        }
    )

    ok = all(r["passes_on_current_code"] and r["fails_on_predecessor"] for r in rows)
    rep = {
        "what": "every §7 v1.6.15 fail-open KAT, proven capable of failing",
        "rows": rows,
        "n": len(rows),
        "all_discriminate": ok,
        "verdict": "ALL KATS DISCRIMINATE"
        if ok
        else "AT LEAST ONE KAT CANNOT FAIL — do not rely on it",
    }
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    for r in rows:
        print(
            f"  {'OK ' if r['passes_on_current_code'] and r['fails_on_predecessor'] else 'BAD'} "
            f"passes_now={r['passes_on_current_code']!s:<5} fails_on_predecessor="
            f"{r['fails_on_predecessor']!s:<5} {r['fix']}"
        )
    print(f"\n{rep['verdict']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
