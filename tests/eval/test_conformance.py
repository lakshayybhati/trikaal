"""KATs for the money-run conformance gate + money mode (audit item 3 / pre-flight Item 8).

The property under test: EVERY §3a pin the config could silently violate is caught — each
mutation of the surface produces its specific failure line, and the true money config passes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from trikaal.data.universe_loader import eval_block_bounds_ms
from trikaal.eval.conformance import (
    ConformanceError,
    assert_conformance,
    conformance_failures,
    symbols_sha256,
    verdict_dsr_failures,
)
from trikaal.eval.xsection import XSectionConfig, primary_region_grid_ms

MDE = Path("runs_manifest/m6_mde_inputs.json")
DEC = Path("runs_manifest/m6_spread_deciles.json")
HAVE_ARTIFACTS = MDE.exists() and DEC.exists()

pytestmark = pytest.mark.skipif(
    not HAVE_ARTIFACTS, reason="pinned-surface artifacts not present in this checkout"
)


def _money_cfg(**kw) -> XSectionConfig:
    from trikaal.eval.costs import load_spread_deciles

    base = dict(
        h=15,
        seq_len=512,
        cap_per_symbol=None,
        spread_frac_by_symbol=load_spread_deciles(DEC),
        money=True,
    )
    base.update(kw)
    return XSectionConfig(**base)


def _symbols() -> list[str]:
    return list(json.loads(MDE.read_text())["symbols_sampled"])


# ------------------------------------------------------------------ money-mode guards
def test_money_mode_forbids_cap_and_requires_deciles():
    with pytest.raises(ValueError, match="forbids cap_per_symbol"):
        XSectionConfig(money=True, cap_per_symbol=200, spread_frac_by_symbol={"A": 1e-4})
    with pytest.raises(ValueError, match="requires per-symbol spread deciles"):
        XSectionConfig(money=True, cap_per_symbol=None)


def test_primary_region_grid_is_the_mde_basis():
    """The money headline grid must BE the grid scripts/m6_prereg.py computed the MDE on:
    one continuous stride grid anchored at blocks[1][0] — never a per-block concatenation."""
    cfg = _money_cfg()
    grid = primary_region_grid_ms(cfg)
    bounds = eval_block_bounds_ms(
        cfg.window_start, cfg.window_end, train_frac=cfg.train_frac, k=cfg.n_blocks
    )
    assert int(grid[0]) == bounds[1][0]  # anchored at the block-1 edge (VAL excluded)
    assert int(grid[-1]) < bounds[cfg.n_blocks - 1][1]
    assert np.all(np.diff(grid) == cfg.h * 60_000)  # ONE stride phase, no re-anchoring
    region = json.loads(MDE.read_text())["primary_region_ms"]
    assert [int(grid[0]), int(region[1])] == [region[0], region[1]]
    n = json.loads(MDE.read_text())["h15_pooled"]["T"]
    # T counts label-COMPLETABLE pooled periods; only the region's final instant can lack a
    # forward label (its h-bar label would need a bar at the data end) — same grid, exactly
    assert grid.size - n in (0, 1)
    assert grid.size == 35_064 and n == 35_063  # the pinned-surface literals, hand-checked


# ------------------------------------------------------------------ the gate itself
def test_conformance_passes_on_the_real_money_config():
    fails = conformance_failures(_money_cfg(), symbols=_symbols(), seeds=(0, 1, 2))
    assert fails == []
    assert_conformance(_money_cfg(), symbols=_symbols(), seeds=(0, 1, 2))  # no raise


@pytest.mark.parametrize(
    ("mutation", "needle"),
    [
        (dict(kappas=(1.0, 2.0, 3.0)), "kappa grid"),
        (dict(headline_cost=0.0010), "headline_cost"),
        (dict(train_frac=0.8), "train_frac"),
        (dict(money=False, cap_per_symbol=None), "money mode"),
    ],
)
def test_each_config_mutation_is_caught(mutation, needle):
    cfg = replace(_money_cfg(), **mutation)
    fails = conformance_failures(cfg, symbols=_symbols(), seeds=(0, 1, 2))
    assert any(needle in f for f in fails), f"{needle!r} not caught: {fails}"


def test_cap_seeds_symbols_and_flat_deciles_are_caught():
    cfg = _money_cfg()
    # the dev cap sneaking back in (bypass __post_init__ via replace-on-frozen? dataclass is
    # mutable — emulate a drifted driver by setting the field after construction)
    drifted = _money_cfg()
    drifted.cap_per_symbol = 200
    fails = conformance_failures(drifted, symbols=_symbols(), seeds=(0, 1, 2))
    assert any("cap_per_symbol" in f for f in fails)
    # wrong seeds
    fails = conformance_failures(cfg, symbols=_symbols(), seeds=(0, 1))
    assert any("seeds" in f for f in fails)
    fails = conformance_failures(cfg, symbols=_symbols(), seeds=(0, 1, 3))
    assert any("seeds" in f for f in fails)
    # wrong symbol set (drop one)
    fails = conformance_failures(cfg, symbols=_symbols()[:-1], seeds=(0, 1, 2))
    assert any("symbol set" in f for f in fails)
    # flat decile map over the evaluated set
    flat = _money_cfg()
    flat.spread_frac_by_symbol = dict.fromkeys(_symbols(), 1e-4)
    fails = conformance_failures(flat, symbols=_symbols(), seeds=(0, 1, 2))
    assert any("FLAT" in f for f in fails)
    # map missing an evaluated symbol
    holey = _money_cfg()
    holey.spread_frac_by_symbol = {
        s: v for s, v in holey.spread_frac_by_symbol.items() if s != _symbols()[0]
    }
    fails = conformance_failures(holey, symbols=_symbols(), seeds=(0, 1, 2))
    assert any("missing from the decile map" in f for f in fails)
    # decile values diverging from the committed artifact
    skewed = _money_cfg()
    skewed.spread_frac_by_symbol = dict(skewed.spread_frac_by_symbol)
    skewed.spread_frac_by_symbol[_symbols()[0]] = 5e-4
    fails = conformance_failures(skewed, symbols=_symbols(), seeds=(0, 1, 2))
    assert any("diverge from the committed artifact" in f for f in fails)


def test_symbol_hash_pin_matches_the_committed_inputs():
    syms = _symbols()
    assert symbols_sha256(sorted(syms)).startswith("60e24f598de96012")
    with pytest.raises(ConformanceError, match="§3a surface"):
        assert_conformance(_money_cfg(), symbols=syms, seeds=(0, 2, 1))  # order matters


def test_shuffle_threading_probe_has_teeth(monkeypatch):
    """The train/eval threading check must FAIL if either path's seed derivation drifts."""
    import trikaal.eval.conformance as conf

    real = conf.shuffle_micro

    def drifted(x, mask, segment_id, *, symbol, seed):
        return real(x, mask, segment_id, symbol=symbol, seed=seed + 1)  # eval-side drift

    monkeypatch.setattr(conf, "shuffle_micro", drifted)
    fails = conf.conformance_failures(_money_cfg(), symbols=_symbols(), seeds=(0, 1, 2))
    assert sum("threading DIVERGES" in f for f in fails) == 3  # every pinned seed caught


# ------------------------------------------------------------------ the §3-clause-5 DSR pins
def _true_trials() -> dict[tuple[int, int, int, float], float]:
    """The exact pinned 180-trial enumeration with deterministic small values."""
    return {
        (c, s, h, k): 1e-3 * ((c * 7 + s * 3 + int(h) + int(2 * k)) % 5)
        for c in (1, 2, 3, 4, 5)
        for s in (0, 1, 2)
        for h in (5, 15, 60)
        for k in (1.0, 1.5, 2.0, 3.0)
    }


def _var0(trials: dict) -> float:
    return float(np.var(np.array([trials[k] for k in sorted(trials)], dtype=np.float64)))


def test_verdict_dsr_pins_pass_on_the_true_recipe():
    t = _true_trials()
    assert verdict_dsr_failures(n_trials=180, threshold=0.95, trials=t, var_sr=_var0(t)) == []


def test_verdict_dsr_doctored_n_trials_240_is_caught():
    t = _true_trials()
    fails = verdict_dsr_failures(n_trials=240, threshold=0.95, trials=t, var_sr=_var0(t))
    assert any("n_trials 240" in f for f in fails)


def test_verdict_dsr_fourth_horizon_budget_is_caught():
    """The old '4 horizons / 240' bookkeeping error: h=1 trials must fail the cross product."""
    t = _true_trials()
    for c in (1, 2, 3, 4, 5):
        for s in (0, 1, 2):
            for k in (1.0, 1.5, 2.0, 3.0):
                t[(c, s, 1, k)] = 1e-3
    fails = verdict_dsr_failures(n_trials=180, threshold=0.95, trials=t, var_sr=_var0(t))
    assert any("cross product" in f and "outside the pin" in f for f in fails)


def test_verdict_dsr_four_kappa_only_var_basis_is_caught():
    """The M5 run_harness convention (var over ONE cell's 4 κ VAL IRs) is NOT the M6 recipe —
    both the shrunken trial set and the subset-variance must be named."""
    full = _true_trials()
    subset = {
        key: v for key, v in full.items() if key[:3] == (4, 0, 15)
    }  # the 4 κ of one trial row
    fails = verdict_dsr_failures(n_trials=180, threshold=0.95, trials=subset, var_sr=_var0(subset))
    assert any("cross product" in f and "176 missing" in f for f in fails)
    # and: right keys, but var_sr computed over a 4-κ subset → the bit-exact re-derivation fails
    fails2 = verdict_dsr_failures(n_trials=180, threshold=0.95, trials=full, var_sr=_var0(subset))
    assert any("var_sr" in f and "wrong source" in f for f in fails2)


def test_verdict_dsr_wrong_ddof_and_threshold_are_caught():
    t = _true_trials()
    vals = np.array([t[k] for k in sorted(t)], dtype=np.float64)
    fails = verdict_dsr_failures(
        n_trials=180, threshold=0.95, trials=t, var_sr=float(np.var(vals, ddof=1))
    )
    assert any("wrong source" in f or "ddof" in f for f in fails)
    fails2 = verdict_dsr_failures(n_trials=180, threshold=0.90, trials=t, var_sr=_var0(t))
    assert any("threshold 0.9" in f for f in fails2)


# ------------------------------------------------------------------ the §7-v1.3 causal pin
def test_cell_tokenizer_causal_pin_and_mutations():
    """Every M6 cell tokenizer must report encoder_causal=True; a bidirectional config is a
    NAMED failure (the token-causality adjudication — invariant 2 binds eval inputs)."""
    from trikaal.eval.conformance import cell_tokenizer_failures
    from trikaal.tokenizer.model import TokenizerAE
    from trikaal.train.cells import CELLS, build_cell_tokenizer

    tiny = dict(d_model=32, n_layers=1, n_heads=2, d_ff=64, max_len=64)
    # the real construction path passes
    tok = build_cell_tokenizer(CELLS[3], **tiny)
    assert cell_tokenizer_failures(tok.get_config(), run="cell4_seed0") == []
    # a bidirectional checkpoint config is caught, by name
    bidi = TokenizerAE(encoder_causal=False, **tiny)
    fails = cell_tokenizer_failures(bidi.get_config(), run="cell4_seed0")
    assert len(fails) == 1 and "encoder_causal=False" in fails[0] and "v1.3" in fails[0]
    # a doctored/absent flag is also caught (an old-schema checkpoint cannot slip through)
    cfg = dict(tok.get_config())
    del cfg["encoder_causal"]
    assert any("encoder_causal=None" in f for f in cell_tokenizer_failures(cfg, run="r"))
