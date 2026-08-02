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


def codebook_usage(tokens: np.ndarray, vocab: int) -> dict[str, float]:
    """Utilization + empirical code entropy (bits) of one token stream against its codebook.

    Referee-preempt (non-gating, CO2 item 5): FSQ-vs-BSQ *effective rates* differ in the
    literature (FSQ ≈ 100 % codebook utilization vs BSQ-style learned codebooks ≈ 94 %), so we
    report ours per cell instead of being asked. ``entropy_bits`` is the Shannon entropy of the
    empirical token distribution — the stream's effective bits/subtoken; ``capacity_bits`` is
    log2(vocab) (the G-parity bpt component); ``perplexity`` = 2^entropy_bits."""
    t = np.asarray(tokens)
    if t.size == 0:
        raise ValueError("codebook_usage needs a non-empty token stream")
    if not np.issubdtype(t.dtype, np.integer):
        raise ValueError(f"tokens must be integer ids, got dtype {t.dtype}")
    if int(t.min()) < 0 or int(t.max()) >= vocab:
        raise ValueError(f"token ids outside [0, {vocab}): min={t.min()}, max={t.max()}")
    counts = np.bincount(t.ravel(), minlength=int(vocab)).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0.0]
    h_bits = float(-(nz * np.log2(nz)).sum())
    return {
        "vocab": int(vocab),
        "n_tokens": int(t.size),
        "n_used": int((counts > 0).sum()),
        "utilization": float((counts > 0).mean()),
        "entropy_bits": h_bits,
        "capacity_bits": float(np.log2(vocab)),
        "efficiency": h_bits / float(np.log2(vocab)),
        "perplexity": float(2.0**h_bits),
    }


def cell_codebook_diagnostic(
    b_c: np.ndarray, b_f: np.ndarray, *, v_c: int, v_f: int
) -> dict[str, object]:
    """Per-cell coarse+fine codebook diagnostic over the full evaluated token streams.

    ``effective_bits_per_token`` = H(coarse) + H(fine) is directly comparable to the G-parity
    bits-per-token capacity (FSQ 20.06 / BSQ 20.00 — invariant 6): the gap is how much of the
    matched budget each quantizer actually spends on real data. Non-gating — reported in every
    cell eval artifact and the smoke output, never thresholded."""
    c = codebook_usage(b_c, v_c)
    f = codebook_usage(b_f, v_f)
    return {
        "coarse": c,
        "fine": f,
        "effective_bits_per_token": c["entropy_bits"] + f["entropy_bits"],
        "capacity_bits_per_token": c["capacity_bits"] + f["capacity_bits"],
    }


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
    "cell_codebook_diagnostic",
    "codebook_usage",
    "discriminative_score",
    "mean_pinball",
    "picp_mpiw",
    "pinball_loss",
    "tstr_trtr",
    "vol_mae_r2",
]


# --------------------------------------------------------------- §7 v1.6 C-12 M1 (ADOPTED)
def ohlcv_recon_diagnostic(
    model, x_arm: np.ndarray, m_arm: np.ndarray, *, n_ohlcv: int = 7, device: str = "cpu"
) -> dict:
    """Reconstruction quality restricted to the OHLCV dims — the C-12 capacity disclosure.

    WHAT IT IS FOR. The Cell-5 placebo does not merely remove microstructure information: the
    block permutation destroys the contemporaneous micro↔OHLCV dependence too (**measured
    0.2037 → 0.0005**), so Cell 5's micro channels become six dims of INDEPENDENT NOISE. Noise
    with a preserved marginal is incompressible and shares no structure with OHLCV, so any bits
    spent representing it are bits taken from OHLCV — and ``micro_point_weight = 3.0`` aims triple
    gradient pressure at fitting it. ΔIR(4−5) therefore contains (information) + (capacity
    handicap) and the two are not separable from the contrast alone.

    THIS CONVERTS AN UNBOUNDED CONFOUND INTO A MEASURED MAGNITUDE. The OHLCV dims are present and
    IDENTICAL in both arms — ``block_time_permute`` never touches them (verified: the OHLCV columns
    come back byte-identical) — so OHLCV reconstruction is a like-for-like comparison. If Cell 5
    reconstructs OHLCV measurably worse than Cell 4 on the same bars, capacity was diverted.

    NON-GATING BY CONSTRUCTION. It is a DISCLOSURE reported beside the headline, never a clause,
    never a threshold. It cannot flip SURVIVES↔NULL and no value here is compared against a bar.
    """
    import torch

    model.eval()
    xt = torch.as_tensor(np.asarray(x_arm, dtype=np.float32), device=device)
    mt = torch.as_tensor(np.asarray(m_arm, dtype=np.float32), device=device)
    if xt.ndim == 2:  # [T, F] -> one window
        xt, mt = xt.unsqueeze(0), mt.unsqueeze(0)
    with torch.no_grad():
        z_hat, _c, _ci, _fi = model.quant(model.latent(xt, mt))
        recon = model.decode_latent(z_hat)
    r = recon.detach().cpu().numpy().astype(np.float64)
    t = xt.detach().cpu().numpy().astype(np.float64)
    k = min(int(n_ohlcv), r.shape[-1])
    per_dim = [float(np.abs(r[..., i] - t[..., i]).mean()) for i in range(k)]
    return {
        "ohlcv_recon_mae": float(np.mean(per_dim)),
        "ohlcv_recon_mae_by_dim": per_dim,
        "ohlcv_recon_mae_dim0_return": per_dim[0] if per_dim else float("nan"),
        "n_ohlcv_dims": k,
        "bars": int(r.shape[0] * r.shape[1]),
        "non_gating": True,
        "what_it_measures": (
            "C-12 M1: OHLCV reconstruction is like-for-like across arms because the placebo "
            "permutation leaves the OHLCV columns byte-identical. Cell 5 reconstructing OHLCV "
            "worse than Cell 4 is capacity diverted to encoding six dims of independent noise "
            "under a 3x micro weight. DISCLOSURE ONLY — never a clause, never a threshold."
        ),
    }


# --------------------------------------------- §7 v1.6.11 C-1 (supervisor-adopted, non-gating)
def single_bar_decode_diagnostic(
    model, x_arm: np.ndarray, m_arm: np.ndarray, *, device: str = "cpu"
) -> dict:
    """Single-bar vs in-window decode agreement — the C-1 handicap channel, on the REAL cells.

    WHY IT IS HERE RATHER THAN ON A TOY. The tokenizer decoder is a sequence Transformer trained on
    FULL WINDOWS, but ``predict._rollout`` calls ``decode_latent`` with a LENGTH-1 sequence and
    takes ``[:, 0, 0]`` — exactly the quantity μ̂ accumulates. The gap is therefore structural. Its
    MAGNITUDE was chased on briefly-trained toys and came back **indeterminate at n=3** (t = 1.82,
    Welch df 2.39, crit 2.62): resolving it on the proxy needs n ≈ 13 per arm, and the proxy may
    not transfer anyway. On the real run the M6 tokenizers already exist and 5 seeds are already
    paid for, so the question is answered at the actual scale for the cost of a forward pass.

    THE SECOND DIAGNOSTIC WIRED TO BOUND A HANDICAP CHANNEL, after ``ohlcv_recon_diagnostic``.
    **Both point the same direction if real:** C-12's capacity handicap and a decode-noise
    handicap would each degrade the PLACEBO arm more than the treatment arm, and each would
    therefore INFLATE ΔIR(4−5). They compound.

    DEGENERACY IS CHECKED BEFORE THE AGREEMENT IS READ (§7 v1.6.10). A single-bar decode collapsed
    to a constant scores a PERFECT sign agreement for free — measured live at
    ``variance_ratio = 0.000`` with agreement 1.0000 — so the variance ratio travels with the
    score and a degenerate row is flagged, never quietly read.

    NON-GATING: reported per (cell, seed), never a clause, never a threshold.
    """
    import torch

    model.eval()
    xt = torch.as_tensor(np.asarray(x_arm, dtype=np.float32), device=device)
    mt = torch.as_tensor(np.asarray(m_arm, dtype=np.float32), device=device)
    if xt.ndim == 2:
        xt, mt = xt.unsqueeze(0), mt.unsqueeze(0)
    with torch.no_grad():
        z_hat, _c, _ci, _fi = model.quant(model.latent(xt, mt))
        win = model.decode_latent(z_hat)
        one = torch.cat(
            [model.decode_latent(z_hat[:, t : t + 1, :]) for t in range(z_hat.shape[1])], dim=1
        )
    w = win.detach().cpu().numpy().astype(np.float64)[..., 0]
    o = one.detach().cpu().numpy().astype(np.float64)[..., 0]
    vw, vo = float(w.var()), float(o.var())
    degenerate = bool(vo <= 1e-8 * max(vw, 1e-30) or float(o.std()) <= 1e-8)
    return {
        "sign_agreement_dim0": float((np.sign(o) == np.sign(w)).mean()),
        "variance_ratio_dim0": (vo / vw) if vw > 0 else float("nan"),
        "single_bar_degenerate": degenerate,
        "mean_abs_delta_dim0": float(np.abs(o - w).mean()),
        "mean_abs_delta_over_sd_dim0": float(np.abs(o - w).mean() / max(float(w.std()), 1e-30)),
        "bars": int(w.size),
        "non_gating": True,
        "read_with": (
            "sign_agreement_dim0 is MEANINGLESS when single_bar_degenerate is true — a collapsed "
            "constant decode scores 1.0 for free. Always read the variance ratio beside it."
        ),
        "what_it_measures": (
            "C-1: the decision path decodes ONE BAR through a decoder trained on FULL WINDOWS. "
            "Lower agreement on the PLACEBO arm than the treatment arm would degrade cell 5's "
            "mu-hat for a reason unrelated to microstructure information, INFLATING dIR(4-5) — "
            "the same direction as the C-12 capacity handicap, with which it compounds. "
            "DISCLOSURE ONLY — never a clause, never a threshold."
        ),
    }
