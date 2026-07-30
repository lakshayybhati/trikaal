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
    PINNED_DSR,
    PINNED_SEEDS,
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
    fails = conformance_failures(_money_cfg(), symbols=_symbols(), seeds=PINNED_SEEDS)
    assert fails == []
    assert_conformance(_money_cfg(), symbols=_symbols(), seeds=PINNED_SEEDS)  # no raise


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
    fails = conf.conformance_failures(_money_cfg(), symbols=_symbols(), seeds=PINNED_SEEDS)
    assert sum("threading DIVERGES" in f for f in fails) == len(PINNED_SEEDS)  # every pinned seed


# ------------------------------------------------------------------ the §3-clause-5 DSR pins
def _true_trials() -> dict[tuple[int, int, int, float], float]:
    """The exact pinned enumeration with deterministic small values.

    §7 v1.5: the per-cell SCALE is deliberately cell-dependent so the PLACEBO-only dispersion and
    the superseded ALL-ARMS dispersion genuinely DIFFER. The previous fixture's values were a
    modular function that made the two bases coincide, which silently disarmed the A.4 mutation
    check — a non-discriminating fixture, caught when the mutation assertion could not fire.
    """
    return {
        (c, s, h, k): 1e-3 * c * ((c * 7 + s * 3 + int(h) + int(2 * k)) % 5)
        for c in (1, 2, 3, 4, 5)
        for s in PINNED_SEEDS
        for h in PINNED_DSR["horizons"]
        for k in PINNED_DSR["kappas"]
    }


def _var0(trials: dict) -> float:
    """§7 v1.5 A.4: the dispersion basis is the PLACEBO arm only, NOT all arms."""
    basis = int(PINNED_DSR["var_sr_basis_cell"])
    keys = sorted(k for k in trials if k[0] == basis)
    return float(np.var(np.array([trials[k] for k in keys], dtype=np.float64)))


def _var_all_arms(trials: dict) -> float:
    """The SUPERSEDED pre-v1.5 all-arms basis — retained only to prove the gate rejects it."""
    return float(np.var(np.array([trials[k] for k in sorted(trials)], dtype=np.float64)))


def test_verdict_dsr_pins_pass_on_the_true_recipe():
    t = _true_trials()
    assert (
        verdict_dsr_failures(
            n_trials=PINNED_DSR["n_trials"], threshold=0.95, trials=t, var_sr=_var0(t)
        )
        == []
    )


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
    fails = verdict_dsr_failures(
        n_trials=PINNED_DSR["n_trials"], threshold=0.95, trials=t, var_sr=_var0(t)
    )
    assert any("cross product" in f and "outside the pin" in f for f in fails)


def test_verdict_dsr_four_kappa_only_var_basis_is_caught():
    """The M5 run_harness convention (var over ONE cell's 4 κ VAL IRs) is NOT the M6 recipe.

    §7 v1.5 gives this THREE independent ways to fail, and all three are asserted: the shrunken
    trial set fails the cross-product; the placebo basis is ABSENT from that subset entirely (a
    cell-4-only row contains no cell-5 trials); and on the full set a subset-variance still fails
    the bit-exact re-derivation against the placebo basis.
    """
    full = _true_trials()
    subset = {key: v for key, v in full.items() if key[:3] == (4, 0, 15)}  # one row's 4 κ
    fails = verdict_dsr_failures(
        n_trials=PINNED_DSR["n_trials"], threshold=0.95, trials=subset, var_sr=0.0
    )
    n_full = (
        len(PINNED_DSR["cells"])
        * len(PINNED_SEEDS)
        * len(PINNED_DSR["horizons"])
        * len(PINNED_DSR["kappas"])
    )
    assert any("cross product" in f and f"{n_full - len(subset)} missing" in f for f in fails)
    # the placebo basis is not even present in a cell-4-only subset — named explicitly
    assert any("basis cell" in f and "absent" in f for f in fails)

    # right keys, but var_sr over a 4-κ subset → the bit-exact re-derivation fails
    sub_var = float(np.var(np.array([full[k] for k in sorted(subset)], dtype=np.float64)))
    fails2 = verdict_dsr_failures(
        n_trials=PINNED_DSR["n_trials"], threshold=0.95, trials=full, var_sr=sub_var
    )
    assert any("var_sr" in f and "placebo" in f for f in fails2)
    # and the SUPERSEDED pre-v1.5 ALL-ARMS basis is rejected too (§7 v1.5 A.4 mutation check).
    # Assert the fixture actually DISCRIMINATES first — a fixture where the two bases coincide
    # would make this mutation check vacuously pass.
    assert _var_all_arms(full) != _var0(full)
    fails3 = verdict_dsr_failures(
        n_trials=PINNED_DSR["n_trials"], threshold=0.95, trials=full, var_sr=_var_all_arms(full)
    )
    assert any("var_sr" in f and "placebo" in f for f in fails3)


def test_verdict_dsr_wrong_ddof_and_threshold_are_caught():
    t = _true_trials()
    vals = np.array([t[k] for k in sorted(t)], dtype=np.float64)
    fails = verdict_dsr_failures(
        n_trials=180, threshold=0.95, trials=t, var_sr=float(np.var(vals, ddof=1))
    )
    assert any("wrong source" in f or "ddof" in f for f in fails)
    fails2 = verdict_dsr_failures(
        n_trials=PINNED_DSR["n_trials"], threshold=0.90, trials=t, var_sr=_var0(t)
    )
    assert any("threshold 0.9" in f for f in fails2)


# ------------------------------------------------------------------ the §7-v1.3 causal pin
def test_cell_tokenizer_causal_pin_and_mutations():
    """Every M6 cell tokenizer must report encoder_causal=True; a bidirectional config is a
    NAMED failure (the token-causality adjudication — invariant 2 binds eval inputs)."""
    from trikaal.eval.conformance import cell_tokenizer_failures
    from trikaal.tokenizer.model import TokenizerAE
    from trikaal.train.cells import CELLS, build_cell_tokenizer

    tiny = dict(d_model=32, n_layers=1, n_heads=2, d_ff=64, max_len=64)
    # the real construction path passes both pins
    tok = build_cell_tokenizer(CELLS[3], **tiny)
    assert cell_tokenizer_failures(tok.get_config(), run="cell4_seed0") == []
    # a bidirectional checkpoint config is caught, by name (other pins kept conformant)
    bidi = TokenizerAE(encoder_causal=False, fine_pointwise=True, micro_point_weight=3.0, **tiny)
    fails = cell_tokenizer_failures(bidi.get_config(), run="cell4_seed0")
    assert len(fails) == 1 and "encoder_causal=False" in fails[0] and "v1.3" in fails[0]
    # a doctored/absent flag is also caught (an old-schema checkpoint cannot slip through)
    cfg = dict(tok.get_config())
    del cfg["encoder_causal"]
    assert any("encoder_causal=None" in f for f in cell_tokenizer_failures(cfg, run="r"))
    # §7 v1.4: a contextual-fine checkpoint config is caught, by name (other pins conformant)
    ctx_fine = TokenizerAE(
        encoder_causal=True, fine_pointwise=False, micro_point_weight=3.0, **tiny
    )
    fails4 = cell_tokenizer_failures(ctx_fine.get_config(), run="cell4_seed0")
    assert len(fails4) == 1 and "fine_pointwise=False" in fails4[0] and "v1.4" in fails4[0]
    cfg4 = dict(tok.get_config())
    del cfg4["fine_pointwise"]
    assert any("fine_pointwise=None" in f for f in cell_tokenizer_failures(cfg4, run="r"))
    # §7 v1.4.1: an off-pin lambda is caught, by name (default 1.0 = the unweighted config)
    off_lam = TokenizerAE(encoder_causal=True, fine_pointwise=True, **tiny)
    fails41 = cell_tokenizer_failures(off_lam.get_config(), run="cell4_seed0")
    assert len(fails41) == 1 and "micro_point_weight=1.0" in fails41[0] and "v1.4.1" in fails41[0]
    cfg41 = dict(tok.get_config())
    del cfg41["micro_point_weight"]
    assert any("micro_point_weight=None" in f for f in cell_tokenizer_failures(cfg41, run="r"))
    # a config violating ALL pins reports every named failure
    allbad = TokenizerAE(encoder_causal=False, fine_pointwise=False, **tiny)
    assert len(cell_tokenizer_failures(allbad.get_config(), run="r")) == 3
