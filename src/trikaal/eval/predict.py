"""Multi-horizon predictions via the M3 KV-cache rollout (§8.B.1 signal source).

For each decision bar ``t`` we feed the ``L``-bar context tokens through the AR model (filling the
KV cache), then autoregressively roll out ``h`` steps, accumulating the predicted standardized
``ret_close`` at each step. The cumulative is de-normalized by the decision-bar causal vol ``σ_t``
to a raw expected log-return ``μ̂_{t,h}`` over ``[t+1, t+h]``.

**The estimator (§7 v1.4.2 — MEAN, not mode).** ``μ̂`` is pre-registered as the conditional
MEAN of the forward return. The greedy-argmax decode is the conditional MODE; under a skewed
per-step return distribution the mode is biased, and the bias COMPOUNDS over the ``h`` rollout
steps (measured on the acceptance fixture: mode decode drifts to mu_mean −0.051 at h=15 with
92.9 % of decisions negative — sign-saturated — while the correct mean sits near 0 with a
balanced sign). The default estimator is now ``"expectation"``: the token CHAIN still advances
greedily (the cache is deterministic and the argmax path fixes the conditioning), but each
step's μ̂ contribution is the MEAN decode ``decode_latent(E[z])`` with
``E[z] = softmax(logits_c) @ grid_c ⊕ softmax(logits_f | ĉ) @ grid_f`` — the mean-field
(delta-method) expectation over the per-step token distribution. This is deterministic and
seed-stable. ``"argmax"`` (the historical mode estimator) is retained for regression KATs and
to reproduce the pre-v1.4.2 anchor. ``"mc_mean"`` (pinned ``n_samples``/``seed``) is the
unbiased-in-expectation Monte-Carlo alternative, retained as the documented fallback; spec §8's
deferred full MC-decode is thereby PARTIALLY un-deferred as a correctness need — mean only, no
full-distribution scope.

**M5 decode contract.** The decoded standardized ret_close is a vol-relative move multiplied by
the causal σ_t (§3.4). Real model-driven rollout; M5's *numbers* are dev-grade (one symbol),
proving the machine runs leak-free.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from trikaal.data.config import BAR_MS
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE

ESTIMATORS = ("expectation", "argmax", "mc_mean")
MC_DEFAULT_SAMPLES = 32
MC_DEFAULT_SEED = 20260721


@torch.no_grad()
def predict_mu(
    model: TrikaalAR,
    tok: TokenizerAE,
    b_c: np.ndarray,
    b_f: np.ndarray,
    ts: np.ndarray,
    sigma: np.ndarray,
    decisions: np.ndarray,
    *,
    h: int,
    seq_len: int = 128,
    device: str = "cpu",
    chunk: int = 512,
    estimator: str = "expectation",
    mc_samples: int = MC_DEFAULT_SAMPLES,
    mc_seed: int = MC_DEFAULT_SEED,
    prefill: bool = False,  # §7 v1.6 — NON-DEFAULT until condition (a) passes on GPU
) -> np.ndarray:
    """Rollout μ̂_{t,h} (raw cumulative log-return) for each decision bar in ``decisions``.

    ``estimator`` (§7 v1.4.2): ``"expectation"`` (default — the conditional-MEAN decode μ̂ is
    pre-registered as), ``"argmax"`` (the historical mode; regression/anchor-reproduction only),
    or ``"mc_mean"`` (unbiased MC over ``mc_samples`` sampled chains at ``mc_seed`` — the
    documented fallback). All three share the context fill, the k=1 exclusion, and the σ_t scale;
    they differ ONLY in how each step's contribution is formed. ``expectation`` and ``argmax``
    are deterministic; ``mc_mean`` is deterministic given ``(mc_samples, mc_seed)``.

    Each decision ``t`` must have ``t − seq_len + 1 ≥ 0`` and ``t + h`` in range. ``chunk`` bounds
    the KV-cache batch (the uncapped money grid is ~35k decisions/symbol; an unchunked cache at
    seq 512 would be tens of GB). Each decision's rollout is row-independent, but GEMM kernels are
    batch-SHAPE-dependent (ULP-level), so bit-exactness across DIFFERENT chunk sizes is NOT
    claimed; ``chunk`` is a FIXED recorded recipe constant (default 512) under which replay is
    deterministic, and cross-size agreement is KAT'd as approximate.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    model.eval()
    tok.eval()
    max_len = int(getattr(model, "_config", {}).get("max_len", 0))
    if max_len and seq_len + h - 1 > max_len:
        # the rollout steps positions up to seq_len-1 + h-1; the RoPE cos/sin cache is built at
        # max_len, so exceeding it is an index crash at eval time. M5's seq_len=128 never reached
        # it; the money config (seq 512, h=60 → position 571) DOES — caught at rehearsal sizing.
        raise ValueError(
            f"rollout needs positions up to {seq_len + h - 1} but the model was built with "
            f"max_len={max_len} — construct it with max_len >= seq_len + h (e.g. "
            f"backbone_kwargs={{'max_len': {seq_len + 64}}})"
        )
    d_all = np.asarray(decisions, dtype=np.int64)
    if d_all.shape[0] == 0:
        return np.array([], dtype=np.float64)
    parts = [
        _predict_mu_batch(
            model,
            tok,
            b_c,
            b_f,
            ts,
            sigma,
            d_all[c0 : c0 + chunk],
            h=h,
            seq_len=seq_len,
            device=device,
            estimator=estimator,
            mc_samples=mc_samples,
            mc_seed=mc_seed,
            prefill=prefill,
        )
        for c0 in range(0, d_all.shape[0], chunk)
    ]
    return np.concatenate(parts)


def _fill_context_prefill(model, cc, cf, cts, seq_len):
    """PREFILL: one causal forward over the whole KNOWN context, then read the last position.

    §7 v1.6. The sequential ``_fill_context`` issues ``seq_len`` single-token ``model.step`` calls
    to populate a KV cache from context that is entirely known in advance — measured at 96.8-98.5%
    of eval wall-clock, the whole of the ~50x gap against the hardware floor. A causal model gives
    each position exactly its predecessors either way, so this computes the SAME object with one
    kernel launch per block instead of ``seq_len``.

    NON-DEFAULT until condition (a) passes on GPU: the discrete chain (``coarse_cond``) is
    identical and the continuous delta is ~1e-6 on mu-hat — inside the 1e-4 tolerance the project
    already sanctions BETWEEN THESE TWO PATHS (tests/model/test_rollout.py:53-56) — but "decisions
    do not move" must be demonstrated at a sample size where flips are possible, not assumed.
    """
    import torch.nn.functional as _F

    from trikaal.model.attention import apply_rope
    from trikaal.model.predictor import BackboneOutput

    caches = model.init_caches()
    x = model.token_embed(cc, cf) + model.temporal(cts)
    for blk, cache in zip(model.blocks, caches, strict=True):
        xn = blk.norm1(x)
        attn = blk.attn
        b, length, _ = xn.shape
        q, k, v = attn._project(xn)
        cos, sin = attn.rope_cos[:length], attn.rope_sin[:length]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        cache["k"], cache["v"] = k, v
        a = _F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h_ = x + attn.out(a.transpose(1, 2).reshape(b, length, attn.d_model))
        x = h_ + blk.ffn(blk.norm2(h_))
    h = model.norm_final(x)[:, -1:, :]
    logits_c = model.w_c(h)
    coarse_cond = logits_c.argmax(dim=-1)
    return caches, BackboneOutput(
        h, logits_c, logits_f=model.fine_logits(h, coarse_cond), coarse_cond=coarse_cond
    )


def _fill_context(model, cc, cf, cts, seq_len):
    caches = model.init_caches()
    out = None
    for p in range(seq_len):  # populate KV cache with the real context; out predicts bar t+1
        out = model.step(cc[:, p : p + 1], cf[:, p : p + 1], cts[:, p : p + 1], caches, p)
    return caches, out


@torch.no_grad()
def _predict_mu_batch(
    model: TrikaalAR,
    tok: TokenizerAE,
    b_c: np.ndarray,
    b_f: np.ndarray,
    ts: np.ndarray,
    sigma: np.ndarray,
    d: np.ndarray,
    *,
    h: int,
    seq_len: int,
    device: str,
    estimator: str,
    mc_samples: int,
    mc_seed: int,
    prefill: bool = False,
) -> np.ndarray:
    """One chunk of decisions through the KV-cache rollout, under the selected estimator."""
    dev = torch.device(device)
    n = d.shape[0]
    offs = np.arange(-seq_len + 1, 1)  # [-L+1 .. 0]
    idx = d[:, None] + offs[None, :]
    cc = torch.from_numpy(b_c[idx].astype(np.int64)).to(dev)
    cf = torch.from_numpy(b_f[idx].astype(np.int64)).to(dev)
    cts = torch.from_numpy(ts[idx].astype(np.int64)).to(dev)
    sig_t = sigma[d].astype(np.float64)
    ts_dec = cts[:, -1:]  # decision-bar timestamp [n,1]

    if estimator == "mc_mean":
        cum = _rollout_mc_mean(
            model, tok, cc, cf, cts, ts_dec, n, h, seq_len, dev, mc_samples, mc_seed
        )
    else:
        cum = _rollout_greedy(
            model, tok, cc, cf, cts, ts_dec, n, h, seq_len, dev, estimator, prefill
        )
    return cum.cpu().numpy().astype(np.float64) * sig_t  # vol-relative → raw (§3.4 contract)


def _rollout_greedy(
    model, tok, cc, cf, cts, ts_dec, n, h, seq_len, dev, estimator, prefill=False
) -> torch.Tensor:
    """Greedy token chain; per-step contribution = mode decode (``argmax``) or mean decode
    (``expectation``: ``decode_latent(E[z])``). Both advance the cache identically."""
    fill = _fill_context_prefill if prefill else _fill_context
    _caches, out = fill(model, cc, cf, cts, seq_len)
    caches = _caches
    cum = torch.zeros(n, dtype=torch.float32, device=dev)
    if estimator == "expectation":
        q = tok.quant
        grid_c = tok.group_grid("coarse", device=dev)  # [v_c, len(coarse)]
        grid_f = tok.group_grid("fine", device=dev)  # [v_f, len(fine)]
        c_idx, f_idx = list(q.coarse_idx), list(q.fine_idx)
    for k in range(1, h + 1):
        pc = out.coarse_cond  # [n,1] greedy coarse (fixes the conditioning + cache path)
        pf = out.logits_f.argmax(dim=-1)  # [n,1] greedy fine | coarse
        # μ̂ covers [t+1, t+h] = log(C_{t+h}/C_{t+1}): EXCLUDE k=1 (entry at C_{t+1}, §8.B.3).
        if k >= 2:
            if estimator == "argmax":
                cum = cum + tok.decode_tokens(pc, pf)[:, 0, 0]
            else:  # expectation: mean decode of the mean-field latent
                p_c = F.softmax(out.logits_c.float(), dim=-1)  # [n,1,v_c]
                p_f = F.softmax(out.logits_f.float(), dim=-1)  # [n,1,v_f] | ĉ
                z = torch.zeros(n, 1, q.dim, dtype=torch.float32, device=dev)
                z[..., c_idx] = (p_c @ grid_c.float()).to(z.dtype)
                z[..., f_idx] = (p_f @ grid_f.float()).to(z.dtype)
                cum = cum + tok.decode_latent(z)[:, 0, 0]
        if k < h:
            out = model.step(pc, pf, ts_dec + k * BAR_MS, caches, seq_len - 1 + k)
    return cum


def _rollout_mc_mean(model, tok, cc, cf, cts, ts_dec, n, h, seq_len, dev, mc_samples, mc_seed):
    """The unbiased-in-expectation fallback: average μ̂ over ``mc_samples`` sampled chains."""
    gen = torch.Generator(device="cpu").manual_seed(mc_seed)
    acc = torch.zeros(n, dtype=torch.float64, device="cpu")
    for _ in range(mc_samples):
        _caches, out = _fill_context(model, cc, cf, cts, seq_len)
        caches = _caches
        cum = torch.zeros(n, dtype=torch.float32, device=dev)
        for k in range(1, h + 1):
            pc_p = F.softmax(out.logits_c.float(), dim=-1).reshape(n, -1).cpu()
            pc = torch.multinomial(pc_p, 1, generator=gen).reshape(n, 1).to(dev)
            logits_f = model.fine_logits(out.h_final, pc)
            pf_p = F.softmax(logits_f.float(), dim=-1).reshape(n, -1).cpu()
            pf = torch.multinomial(pf_p, 1, generator=gen).reshape(n, 1).to(dev)
            if k >= 2:
                cum = cum + tok.decode_tokens(pc, pf)[:, 0, 0]
            if k < h:
                out = model.step(pc, pf, ts_dec + k * BAR_MS, caches, seq_len - 1 + k)
        acc = acc + cum.cpu().double()
    return (acc / mc_samples).to(dev)


def sigma_hat(sigma_t: np.ndarray, h: int) -> np.ndarray:
    """Point dispersion estimate over the horizon — random-walk scaling ``σ_t·√h`` (M5 contract)."""
    return np.asarray(sigma_t, dtype=np.float64) * np.sqrt(h)


__all__ = ["predict_mu", "sigma_hat"]
