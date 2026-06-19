"""Secondary diagnostics — subordinate to the headline net-IR (§8.D).

These localize *why* a cell wins or loses; they are never headlined over net-IR. IC/RankIC live in
``ic_screen.py``; here: realized-vol forecast MAE/R² (vs a persistence baseline), multi-horizon
quantile quality (pinball + PICP/MPIW), and the generative pair (discriminative score + TSTR). The
ML probes (discriminative, TSTR) are deliberately tiny, from-scratch, seed-pinned, and dependency-
free — M5 exercises their code path; M6 produces the numbers. KAT in ``test_diagnostics.py``.
"""

from __future__ import annotations

import numpy as np

from trikaal.eval.ic_screen import spearman_rankic


def vol_mae_r2(
    sigma_hat: np.ndarray, sigma_true: np.ndarray, *, baseline_pred: np.ndarray | None = None
) -> dict[str, float]:
    """Realized-vol forecast MAE + R². R² is vs the realized-vol mean and, if given, vs a
    persistence/EWMA ``baseline_pred`` (the honest bar to beat — report the gap)."""
    sh = np.asarray(sigma_hat, dtype=np.float64)
    st = np.asarray(sigma_true, dtype=np.float64)
    mae = float(np.abs(sh - st).mean())
    ss_tot = float(((st - st.mean()) ** 2).sum())
    ss_res = float(((st - sh) ** 2).sum())
    r2_mean = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    out = {"mae": mae, "r2_vs_mean": r2_mean}
    if baseline_pred is not None:
        bp = np.asarray(baseline_pred, dtype=np.float64)
        ss_base = float(((st - bp) ** 2).sum())
        out["r2_vs_baseline"] = 1.0 - ss_res / ss_base if ss_base > 0 else float("nan")
        out["mae_baseline"] = float(np.abs(bp - st).mean())
    return out


def pinball_loss(y: np.ndarray, q_pred: np.ndarray, q_level: float) -> float:
    """Mean pinball (quantile) loss at level ``q``: q·(y−ŷ) if y≥ŷ else (1−q)·(ŷ−y)."""
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(q_pred, dtype=np.float64)
    diff = y - yhat
    return float(np.where(diff >= 0.0, q_level * diff, (q_level - 1.0) * diff).mean())


def mean_pinball(y: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> float:
    """Pinball averaged over quantile levels (the proper scoring rule; lower is better)."""
    return float(np.mean([pinball_loss(y, p, q) for q, p in quantile_preds.items()]))


def picp_mpiw(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """Prediction-interval coverage probability and mean interval width."""
    y = np.asarray(y, dtype=np.float64)
    lo = np.asarray(low, dtype=np.float64)
    hi = np.asarray(high, dtype=np.float64)
    picp = float(((lo <= y) & (y <= hi)).mean())
    mpiw = float((hi - lo).mean())
    return picp, mpiw


def calibration_error(y: np.ndarray, low: np.ndarray, high: np.ndarray, nominal: float) -> float:
    """|PICP − nominal| for the central (1−α) interval."""
    picp, _ = picp_mpiw(y, low, high)
    return abs(picp - nominal)


def _standardize(x: np.ndarray) -> np.ndarray:
    mu, sd = x.mean(axis=0), x.std(axis=0)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def discriminative_score(
    real: np.ndarray, synth: np.ndarray, *, seed: int = 0, iters: int = 300, lr: float = 0.1
) -> float:
    """|test-accuracy − 0.5| of a from-scratch logistic classifier separating real vs synthetic
    flattened windows (0 = indistinguishable, 0.5 = trivially separable; lower is better)."""
    real = np.asarray(real, dtype=np.float64).reshape(real.shape[0], -1)
    synth = np.asarray(synth, dtype=np.float64).reshape(synth.shape[0], -1)
    x = _standardize(np.vstack([real, synth]))
    y = np.concatenate([np.zeros(len(real)), np.ones(len(synth))])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    cut = len(x) // 2
    xtr, ytr, xte, yte = x[:cut], y[:cut], x[cut:], y[cut:]
    w = np.zeros(x.shape[1])
    b = 0.0
    for _ in range(iters):
        z = xtr @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - ytr
        w -= lr * (xtr.T @ g / len(xtr))
        b -= lr * float(g.mean())
    pred = (1.0 / (1.0 + np.exp(-(xte @ w + b))) >= 0.5).astype(float)
    acc = float((pred == yte).mean())
    return abs(acc - 0.5)


def _linear_probe_rankic(
    feat_train: np.ndarray, ret_train: np.ndarray, feat_test: np.ndarray, ret_test: np.ndarray
) -> float:
    """Least-squares linear forecaster fit on (feat_train→ret_train), RankIC on the test set."""
    a = np.asarray(feat_train, dtype=np.float64)
    a = np.hstack([a, np.ones((len(a), 1))])
    coef, *_ = np.linalg.lstsq(a, np.asarray(ret_train, dtype=np.float64), rcond=None)
    pred = np.hstack([np.asarray(feat_test, dtype=np.float64), np.ones((len(feat_test), 1))]) @ coef
    return spearman_rankic(pred, np.asarray(ret_test, dtype=np.float64))


def tstr_trtr(
    feat_test: np.ndarray,
    ret_test: np.ndarray,
    real_feat: np.ndarray,
    real_ret: np.ndarray,
    synth_feat: np.ndarray,
    synth_ret: np.ndarray,
) -> dict[str, float]:
    """TSTR (train-synthetic) vs TRTR (train-real) probe RankIC on the same real test set, + ratio.
    A ratio near 1 means the synthetic data carries the real predictive structure."""
    trtr = _linear_probe_rankic(real_feat, real_ret, feat_test, ret_test)
    tstr = _linear_probe_rankic(synth_feat, synth_ret, feat_test, ret_test)
    ratio = tstr / trtr if abs(trtr) > 1e-9 else float("nan")
    return {"trtr_rankic": trtr, "tstr_rankic": tstr, "tstr_trtr_ratio": ratio}


__all__ = [
    "calibration_error",
    "discriminative_score",
    "mean_pinball",
    "picp_mpiw",
    "pinball_loss",
    "tstr_trtr",
    "vol_mae_r2",
]
