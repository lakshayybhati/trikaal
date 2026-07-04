"""KATs for the multi-symbol universe loader (m6_design §6 item 4).

The load-bearing property: the training draw derives from the SAME purged walk-forward geometry
the eval uses — a training window that crosses (or lands within the embargo of) the train/eval
boundary is structurally IMPOSSIBLE. Plus: α-weighted symbol sampling, resume-able cursor,
Cell-5 integration, and the causal gate on the multi-symbol path."""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.data.causal_check import exhaustive_truncation_sweep
from trikaal.data.config import synthetic_test_config
from trikaal.data.synthetic import make_synthetic_stream
from trikaal.data.universe_loader import (
    BAR_MS,
    MultiSymbolWindowSampler,
    build_symbol_windows,
    calendar_boundary_ms,
    eval_block_bounds_ms,
    fold_valid_starts,
)
from trikaal.train.arms import ARM_MICRO, ARM_MICRO_SHUFFLED, ARM_OHLCV


def _grid(n: int, t0: int = 1_600_000_000_000) -> np.ndarray:
    return (t0 + np.arange(n) * BAR_MS).astype(np.int64)


def _sw(symbol: str, n: int, *, boundary: int, seed: int = 0, arm: str = ARM_MICRO, seq_len=16):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 16)).astype(np.float32)
    m = np.zeros((n, 16), dtype=np.uint8)
    m[:, 13:16] = 1  # funding/OI masked (real-build invariant; the Cell-5 tripwire needs it)
    return build_symbol_windows(
        symbol,
        x,
        m,
        np.zeros(n, dtype=np.int64),
        _grid(n),
        arm=arm,
        seq_len=seq_len,
        boundary_ms=boundary,
        seed=seed,
    )


# ------------------------------------------------ the boundary is IMPOSSIBLE to cross (the KAT)
def test_window_across_train_eval_boundary_is_impossible():
    n, seq_len, embargo = 1000, 32, 120
    ts = _grid(n)
    boundary = int(ts[700])  # the calendar boundary falls at bar 700
    starts = fold_valid_starts(
        ts, np.zeros(n, dtype=np.int64), seq_len=seq_len, boundary_ms=boundary, embargo_bars=embargo
    )
    assert starts.size > 0
    # hand-computed legality: last bar must open strictly before bar (700 - 120) = 580
    # → last legal start = 580 - 1 - (seq_len - 1) = 548
    assert starts.max() == 700 - embargo - seq_len
    last_bar_ts = ts[starts + seq_len - 1]
    assert (last_bar_ts < boundary - embargo * BAR_MS).all()
    # and NO admissible start could produce a boundary-crossing window: every index that would
    # place any window bar at/after the embargoed boundary is absent
    illegal = set(range(700 - embargo - seq_len + 1, n))
    assert illegal.isdisjoint(set(starts.tolist()))


def test_windows_respect_segment_bounds():
    n, seq_len = 200, 16
    ts = _grid(n)
    seg = np.zeros(n, dtype=np.int64)
    seg[100:] = 1  # a segment break at bar 100
    boundary = int(ts[-1]) + BAR_MS * 200  # boundary far beyond → only segments constrain
    starts = fold_valid_starts(ts, seg, seq_len=seq_len, boundary_ms=boundary, embargo_bars=0)
    for s in starts:
        assert seg[s] == seg[s + seq_len - 1]  # never spans the break
    assert not any(100 - seq_len < s < 100 for s in starts.tolist())


def test_calendar_boundary_and_blocks_are_deterministic_and_shared():
    b = calendar_boundary_ms("2021-01-01", "2025-01-01", 0.7)
    assert b == calendar_boundary_ms("2021-01-01", "2025-01-01", 0.7)  # pure function
    blocks = eval_block_bounds_ms("2021-01-01", "2025-01-01", k=6)
    assert len(blocks) == 6
    assert blocks[0][0] == b  # forward region starts AT the boundary
    for (_lo1, hi1), (lo2, _hi2) in zip(blocks, blocks[1:], strict=False):
        assert hi1 == lo2  # contiguous
    # the boundary is GLOBAL: any two symbols receive the same instant by construction
    # (it is a pure function of the window config — nothing per-symbol enters)


# ------------------------------------------------ α-weighted draw + resume cursor
def test_alpha_weighting_matches_hand_computed_sqrt():
    boundary = int(_grid(2000)[1800])
    a = _sw("AAA", 500, boundary=boundary)  # fewer windows
    b = _sw("BBB", 2000, boundary=boundary)  # more windows
    samp = MultiSymbolWindowSampler([b, a], alpha=0.5, seed=0)  # order normalized internally
    n_a, n_b = a.n_windows, b.n_windows
    expect = np.array([n_a**0.5, n_b**0.5])
    expect = expect / expect.sum()
    np.testing.assert_allclose(samp.weights, expect, atol=1e-12)  # AAA first (sorted)
    # α=0: uniform regardless of size (the tunable range's other end)
    samp0 = MultiSymbolWindowSampler([a, b], alpha=0.0, seed=0)
    np.testing.assert_allclose(samp0.weights, [0.5, 0.5])


def test_sampler_cursor_resume_is_exact():
    boundary = int(_grid(2000)[1800])
    sws = [_sw("AAA", 800, boundary=boundary), _sw("BBB", 1200, boundary=boundary)]
    s1 = MultiSymbolWindowSampler(sws, seed=7)
    _ = s1.sample_batch(8, 16)
    saved = s1.state()
    x_next, m_next, names_next = s1.sample_batch(8, 16)
    s2 = MultiSymbolWindowSampler(sws, seed=7)
    s2.load_state(saved)
    x_re, m_re, names_re = s2.sample_batch(8, 16)
    np.testing.assert_array_equal(x_next, x_re)
    np.testing.assert_array_equal(m_next, m_re)
    assert names_next == names_re
    assert s2.n_drawn == s1.n_drawn


def test_sampler_rejects_empty_symbols():
    boundary = int(_grid(100)[10])  # boundary so early no window fits
    with pytest.raises(ValueError, match="zero fold-legal"):
        MultiSymbolWindowSampler([_sw("AAA", 100, boundary=boundary)], seed=0)


# ------------------------------------------------ Cell-5 + arm integration through the loader
def test_loader_arm_integration_cell5_and_ohlcv():
    boundary = int(_grid(600)[550])
    micro = _sw("AAA", 600, boundary=boundary, arm=ARM_MICRO, seed=3)
    cell5 = _sw("AAA", 600, boundary=boundary, arm=ARM_MICRO_SHUFFLED, seed=3)
    ohlcv = _sw("AAA", 600, boundary=boundary, arm=ARM_OHLCV, seed=3)
    assert micro.x.shape[1] == 16 and cell5.x.shape[1] == 16 and ohlcv.x.shape[1] == 7
    np.testing.assert_array_equal(micro.x[:, :7], cell5.x[:, :7])  # OHLCV untouched by the shuffle
    assert not np.array_equal(micro.x[:, 7:13], cell5.x[:, 7:13])  # micro shuffled
    np.testing.assert_array_equal(ohlcv.x, micro.x[:, :7])  # removal, not zero-fill


# ------------------------------------------------ causal gate on the multi-symbol path
@pytest.mark.gate
def test_multi_symbol_loader_path_causal_sweep_green():
    """The loader consumes compute_features output — the exhaustive truncation sweep must stay
    green on BOTH streams of a 2-symbol setup (the multi-symbol path's causal gate)."""
    cfg = synthetic_test_config(half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20)
    for seed in (11, 12):
        stream, _ = make_synthetic_stream(seed=seed, n_bars=420, inject_anomalies=True)
        rep = exhaustive_truncation_sweep(stream, cfg, stride=1)
        assert rep.passed and rep.coverage == 1.0, rep.summary()
