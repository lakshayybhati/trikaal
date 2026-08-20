"""§7 v1.6 C-5 — the A9 fix and the resource preconditions.

TWO RULINGS ARE PINNED HERE.

**A9 vs A4 were conflated, and that is what broke the first dry run.** A9 is LOCAL, $0, COMPLETE
PATH COVERAGE — does every stage execute and hand off. A4 is the GPU validation at reduced scale —
does it produce the right numbers. The first ``--dry-run`` zeroed training steps but ran the REAL
money eval (1,402,560 decisions per unit, 8.4 h on a 4090), i.e. it attempted A4's job on A9's
budget. The dry run now DECLARES ``money_run=False`` — which is what lifts ``xsection.py:71``'s cap
prohibition, since ``money_run`` is a declaration and never an inference — and scores a SHORTENED
grid with REAL scoring, because a synthetic score would skip the path A9 exists to prove.

**A dry-run artifact must be structurally un-quotable, not merely labelled.** Labelling alone
leaves "someone reads the manifest carefully" as the only thing between a path proof and a
reported result, so ``verdict._validate_artifact`` REFUSES any artifact carrying the dry-run
markers, and ``load_cell_evals`` therefore cannot assemble one.

**Resource preconditions are checked before any compute.** The argument is measured, not
hypothetical: 27.8 GiB was needed on a 16.0 GiB host and it did NOT fail — macOS swapped, and one
unit with zero training steps took 2,581 s. An OOM stops the meter; swap-thrash keeps billing.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from trikaal.eval.verdict import (
    DSR_HORIZONS,
    DSR_KAPPAS,
    DSR_SEEDS,
    PRIMARY_H,
    VerdictInputError,
    load_cell_evals,
    write_cell_eval_artifact,
    write_eval_index,
)
from trikaal.utils.preflight import (
    RAW_BYTES_PER_BAR,
    STRATEGY_FAST,
    STRATEGY_LOW,
    memory_requirement_bytes,
    resource_failures,
    resource_record,
    window_bytes_per_bar,
)

# §7 v1.6.22: codebook is a REQUIRED per-(cell, seed) field, gated at 95% utilization.
CODEBOOK_OK = {
    "coarse": {"utilization": 0.99, "vocab": 891},
    "fine": {"utilization": 0.98, "vocab": 1225},
}

# §7 v1.6.11 C-1: the decode-agreement disclosure is REQUIRED on every artifact.
DECODE_AGREEMENT_OK = {
    "sign_agreement_dim0": 0.95,
    "variance_ratio_dim0": 1.0,
    "single_bar_degenerate": False,
}


# §7 v1.6 C-12 M1: the capacity disclosure is REQUIRED on every artifact.
OHLCV_RECON_OK = {"ohlcv_recon_mae": 0.10, "ohlcv_recon_mae_by_dim": [0.1] * 7, "n_ohlcv_dims": 7}

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts/m6_money_run.py"
BENIGN_MU = {"frac_negative": 0.5, "activity_decisions": 0.5}


def _driver():
    spec = importlib.util.spec_from_file_location("m6_money_run", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ the un-quotability guard
def _write_set(tmp, meta_extra):
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    entries = {}
    for c in (1, 2, 3, 4, 5):
        for s in DSR_SEEDS:
            name, sha = write_cell_eval_artifact(
                tmp,
                cell_id=c,
                seed=s,
                grid={"h": PRIMARY_H, "start_ms": 0, "n_periods": 32},
                headline_series=rng.standard_normal(32) * 1e-3,
                kappa_star_by_h={h: 1.5 for h in DSR_HORIZONS},
                val_ir_by_kappa_by_h={h: {k: 0.4 for k in DSR_KAPPAS} for h in DSR_HORIZONS},
                mu_diag=BENIGN_MU,
                ohlcv_recon=OHLCV_RECON_OK,
                decode_agreement=DECODE_AGREEMENT_OK,
                meta=meta_extra(c, s),
                codebook=CODEBOOK_OK,
            )
            entries[name] = sha
    write_eval_index(tmp, entries)
    return tmp


def test_a_clean_set_assembles(tmp_path):
    """FIXTURE DISCRIMINATION FIRST: if this failed, every refusal below would be vacuous."""
    evals, _ = load_cell_evals(
        _write_set(tmp_path / "ok", lambda c, s: {"money_run": True, "grid_pinned": True})
    )
    assert len(evals) == 25


@pytest.mark.parametrize(
    "marker",
    [{"dry_run": True}, {"grid_pinned": False}, {"money_run": False}],
    ids=["dry_run", "grid_not_pinned", "not_money_run"],
)
def test_a_dry_run_artifact_can_never_be_assembled(tmp_path, marker):
    """Any ONE marker is sufficient — the guard does not depend on all three agreeing."""
    with pytest.raises(VerdictInputError, match="DRY-RUN artifact"):
        load_cell_evals(_write_set(tmp_path / "dry", lambda c, s: dict(marker)))


def test_one_dry_run_unit_poisons_the_whole_set(tmp_path):
    """The dangerous case: 24 real units and one dry-run unit must not quietly assemble."""

    def meta(c, s):
        clean = {"money_run": True, "grid_pinned": True}
        return {"dry_run": True} if (c, s) == (3, DSR_SEEDS[2]) else clean

    with pytest.raises(VerdictInputError, match="DRY-RUN artifact"):
        load_cell_evals(_write_set(tmp_path / "mixed", meta))


# ------------------------------------------------------------ A9 bounds the work it does
def test_the_dry_run_declares_itself_not_a_money_run():
    """A9 option (a): the declaration is what legitimately lifts the cap prohibition."""
    src = DRIVER.read_text().replace(" ", "")  # whitespace-insensitive on BOTH sides
    assert "money_run=notargs.dry_run" in src, (
        "the dry run must DECLARE money_run=False; inferring it from the values would re-create "
        "the C-6 hole"
    )
    assert "cap_per_symbol=DRY_RUN_CAP" in src


def test_the_dry_run_bounds_are_small_enough_to_be_a_wiring_proof():
    """A9 must take MINUTES. The money grid is 35,064 periods x 40 symbols."""
    mod = _driver()
    assert mod.DRY_RUN_CAP <= 32, "a wiring proof must not re-become a numerical one"
    assert mod.DRY_RUN_SYMBOLS <= 8
    # the reduction is real, not cosmetic: cap x symbols against the money grid x symbols
    assert mod.DRY_RUN_CAP * mod.DRY_RUN_SYMBOLS < 35_064 * 40 / 1000
    assert not hasattr(mod, "DRY_RUN_PERIODS"), (
        "a grid slice does NOT bound score_cell (it builds its own grids from cfg) — a knob that "
        "only looks like a bound is worse than no knob"
    )


def test_conformance_still_asserts_the_full_pinned_surface_in_a_dry_run():
    """The A3 proof must NOT be weakened by the A9 bounding — only the LOAD is reduced."""
    tree = ast.parse(DRIVER.read_text())
    main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    call = next(
        n
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "assert_conformance"
    )
    kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert kw["symbols"] == "symbols", (
        f"conformance is asserted over {kw['symbols']!r}, not the full pinned set — the dry run "
        "must bound only the LOAD, never the gate"
    )
    assert kw["seeds"] == "tuple(PINNED_SEEDS)"
    # THE ARGUMENT ITSELF, not just the symbol list. The first draft of this test checked only
    # `symbols` and passed while the driver bounded `cfg` BEFORE the gate — conformance caught
    # that, this test did not. The gated object must be the untouched money config.
    assert ast.unparse(call.args[0]) == "cfg", (
        f"conformance is asserted over {ast.unparse(call.args[0])!r} — it must receive the "
        "UNBOUNDED money config"
    )
    src = DRIVER.read_text()
    # and the bounded objects are DIFFERENT names, so the two cannot be confused
    assert "load_symbols = symbols[:DRY_RUN_SYMBOLS]" in src
    assert "eval_cfg = replace(cfg" in src
    assert src.index("assert_conformance(") < src.index("eval_cfg = replace(cfg"), (
        "the bounding must happen AFTER the gate, not before it"
    )


# ------------------------------------------------------------ resource preconditions
def test_the_measured_precedent_reproduces():
    """The A9 finding, as arithmetic: 84.15M bars -> 27.8 GiB fast / 16.3 GiB low."""
    n = 84_153_600
    assert memory_requirement_bytes(n, STRATEGY_FAST)["peak_gib"] == pytest.approx(27.82, abs=0.05)
    assert memory_requirement_bytes(n, STRATEGY_LOW)["peak_gib"] == pytest.approx(16.30, abs=0.05)
    assert RAW_BYTES_PER_BAR == 112
    assert window_bytes_per_bar(STRATEGY_LOW) < window_bytes_per_bar(STRATEGY_FAST)


def test_an_oversized_requirement_fails_fast_with_both_numbers(tmp_path):
    """Swap-thrash bills silently; this must stop BEFORE any compute and say why."""
    fails = resource_failures(n_bars=10**12, strategy=STRATEGY_FAST, out_dir=tmp_path)
    assert fails, "an absurd requirement must not pass"
    msg = " ".join(fails)
    assert "MEMORY" in msg and "GiB" in msg
    assert "physical RAM" in msg, "the message must carry BOTH numbers, not just a refusal"
    assert "2,581" in msg, "the measured precedent belongs in the message"


def test_a_reasonable_requirement_passes(tmp_path):
    """Discrimination: fail-fast must not become fail-always."""
    assert resource_failures(n_bars=10_000, strategy=STRATEGY_FAST, out_dir=tmp_path) == []


def test_low_strategy_can_rescue_a_host_that_fast_cannot(tmp_path):
    """The whole point of making strategy a HOST property rather than a spec one.

    ★ THE HOST USED TO BE PART OF THIS FIXTURE. ``ram`` below is the host being reasoned about,
    but the call read the REAL machine's physical RAM and only ``headroom`` carried the intent —
    so whether this arithmetic passed depended on the developer's laptop. Measured by bisection,
    it passed only between ~12.05 GiB and ~35.12 GiB: under the floor the rescue leg failed, and
    OVER THE CEILING the "fast cannot" leg failed, so a 64 GiB workstation or a large CI runner
    went red on a test that has nothing to do with either machine. The host is now injected."""
    n = 84_153_600
    fast = memory_requirement_bytes(n, STRATEGY_FAST)["peak_bytes"]
    low = memory_requirement_bytes(n, STRATEGY_LOW)["peak_bytes"]
    assert low < fast
    ram = (fast + low) // 2  # the host under test: between the two requirements
    assert resource_failures(n_bars=n, strategy=STRATEGY_FAST, out_dir=tmp_path, ram_bytes=ram), (
        "fast should not fit a host between the two requirements"
    )
    assert not resource_failures(
        n_bars=n, strategy=STRATEGY_LOW, out_dir=tmp_path, ram_bytes=ram
    ), "low should fit that same host — this is the rescue the strategy exists for"


@pytest.mark.parametrize("host_gib", [1, 8, 12, 16, 32, 64, 256, 1024])
def test_the_rescue_holds_on_every_host_size(tmp_path, host_gib):
    """THE PROPERTY IS ABOUT THE ARITHMETIC, NOT THE MACHINE, so assert it across the range that
    used to decide the outcome. Any host too small for `fast` must still be offered `low`, and the
    two verdicts must never both be failures for a host that `low` fits."""
    n = 84_153_600
    ram = host_gib * 2**30
    fast = memory_requirement_bytes(n, STRATEGY_FAST)["peak_bytes"]
    low = memory_requirement_bytes(n, STRATEGY_LOW)["peak_bytes"]
    f_fails = resource_failures(n_bars=n, strategy=STRATEGY_FAST, out_dir=tmp_path, ram_bytes=ram)
    l_fails = resource_failures(n_bars=n, strategy=STRATEGY_LOW, out_dir=tmp_path, ram_bytes=ram)
    mem = lambda fs: [x for x in fs if x.startswith("MEMORY:")]  # noqa: E731
    assert bool(mem(f_fails)) == (fast > ram), (host_gib, mem(f_fails))
    assert bool(mem(l_fails)) == (low > ram), (host_gib, mem(l_fails))
    if mem(l_fails):
        assert mem(f_fails), "low failed where fast passed — the strategies are misordered"


def test_the_record_states_that_both_strategies_are_numerically_identical(tmp_path):
    rec = resource_record(n_bars=1000, strategy=STRATEGY_LOW, out_dir=tmp_path)
    assert "BYTE-IDENTICAL" in rec["strategy_is_a_host_property"]
    assert rec["host_physical_ram_gib"] and rec["host_free_disk_gib"]


def test_disk_is_checked_too(tmp_path):
    """The precondition sweep is not memory-only."""
    fails = resource_failures(
        n_bars=1000, strategy=STRATEGY_LOW, out_dir=tmp_path, disk_needed_bytes=10**18
    )
    assert any("DISK" in f for f in fails)
