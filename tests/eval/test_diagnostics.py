"""Known-answer tests for the secondary diagnostics (§8.D)."""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.eval.diagnostics import (
    calibration_error,
    cell_codebook_diagnostic,
    codebook_usage,
    discriminative_score,
    mean_pinball,
    picp_mpiw,
    pinball_loss,
    tstr_trtr,
    vol_mae_r2,
)


def test_vol_mae_r2_perfect_and_mean():
    st = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = vol_mae_r2(st, st)
    assert perfect["mae"] == 0.0 and np.isclose(perfect["r2_vs_mean"], 1.0)
    at_mean = vol_mae_r2(np.full(4, 2.5), st)
    assert np.isclose(at_mean["r2_vs_mean"], 0.0) and np.isclose(at_mean["mae"], 1.0)


def test_pinball_loss_hand_computed():
    assert np.isclose(pinball_loss(np.array([1.0]), np.array([0.0]), 0.9), 0.9)  # under-prediction
    assert np.isclose(pinball_loss(np.array([0.0]), np.array([1.0]), 0.9), 0.1)  # over-prediction
    mp = mean_pinball(np.array([1.0]), {0.9: np.array([0.0]), 0.1: np.array([0.0])})
    assert np.isclose(mp, (0.9 + 0.1) / 2)


def test_picp_mpiw_and_calibration():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    picp, mpiw = picp_mpiw(y, np.zeros(5), np.full(5, 2.0))
    assert np.isclose(picp, 0.4) and np.isclose(mpiw, 2.0)  # only 1,2 covered
    assert np.isclose(calibration_error(y, np.zeros(5), np.full(5, 2.0), 0.8), 0.4)


def test_discriminative_score_separable_vs_identical():
    rng = np.random.default_rng(0)
    real = rng.normal(0.0, 1.0, (120, 5))
    synth_sep = rng.normal(8.0, 1.0, (120, 5))  # trivially separable
    assert discriminative_score(real, synth_sep, seed=0) > 0.45
    synth_same = rng.normal(0.0, 1.0, (120, 5))  # same distribution → indistinguishable
    assert discriminative_score(real, synth_same, seed=0) < 0.15


def test_tstr_carries_structure_vs_noise():
    rng = np.random.default_rng(3)
    w = np.array([1.0, -2.0, 0.5])
    test_f = rng.normal(size=(200, 3))
    test_r = test_f @ w + 0.1 * rng.normal(size=200)
    real_f = rng.normal(size=(300, 3))
    real_r = real_f @ w + 0.1 * rng.normal(size=300)
    synth_f = rng.normal(size=(300, 3))
    synth_r = synth_f @ w + 0.1 * rng.normal(size=300)  # same structure
    good = tstr_trtr(test_f, test_r, real_f, real_r, synth_f, synth_r)
    assert 0.7 < good["tstr_trtr_ratio"] < 1.3
    noise = tstr_trtr(test_f, test_r, real_f, real_r, synth_f, rng.normal(size=300))
    assert abs(noise["tstr_rankic"]) < 0.2  # synthetic noise carries no real structure


# ------------------------------------------------------- codebook usage / effective bits (CO2 #5)
def test_codebook_usage_exact_entropies():
    # uniform over the whole codebook: utilization 1, H = log2(V) exactly, perplexity = V
    u = codebook_usage(np.tile(np.arange(8), 100), vocab=8)
    assert u["utilization"] == 1.0 and u["n_used"] == 8
    assert u["entropy_bits"] == pytest.approx(3.0, abs=1e-12)
    assert u["perplexity"] == pytest.approx(8.0, abs=1e-9)
    assert u["efficiency"] == pytest.approx(1.0, abs=1e-12)
    # one repeated code: utilization 1/V, H = 0 (total collapse)
    c = codebook_usage(np.full(500, 3, dtype=np.int64), vocab=8)
    assert c["utilization"] == pytest.approx(1 / 8) and c["entropy_bits"] == 0.0
    # uniform over HALF the codebook: utilization 0.5, H = 2 bits at V=8
    h = codebook_usage(np.tile(np.arange(4), 100), vocab=8)
    assert h["utilization"] == 0.5 and h["entropy_bits"] == pytest.approx(2.0, abs=1e-12)


def test_cell_codebook_diagnostic_combines_coarse_and_fine():
    b_c = np.tile(np.arange(8), 50)  # 3 bits effective
    b_f = np.tile(np.arange(4), 100)  # 2 bits effective, capacity log2(16) = 4
    d = cell_codebook_diagnostic(b_c, b_f, v_c=8, v_f=16)
    assert d["effective_bits_per_token"] == pytest.approx(5.0, abs=1e-12)
    assert d["capacity_bits_per_token"] == pytest.approx(3.0 + 4.0, abs=1e-12)
    assert d["coarse"]["utilization"] == 1.0 and d["fine"]["utilization"] == 0.25


def test_codebook_usage_guards():
    with pytest.raises(ValueError, match="non-empty"):
        codebook_usage(np.array([], dtype=np.int64), vocab=8)
    with pytest.raises(ValueError, match="integer"):
        codebook_usage(np.array([0.5, 1.5]), vocab=8)
    with pytest.raises(ValueError, match="outside"):
        codebook_usage(np.array([0, 8], dtype=np.int64), vocab=8)  # id == vocab
    with pytest.raises(ValueError, match="outside"):
        codebook_usage(np.array([-1, 2], dtype=np.int64), vocab=8)
