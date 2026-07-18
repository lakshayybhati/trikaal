"""Multi-horizon predictions via the M3 KV-cache rollout (§8.B.1 signal source).

For each decision bar ``t`` we feed the ``L``-bar context tokens through the AR model (filling the
KV cache), then autoregressively roll out ``h`` steps — each step decoding the greedy next token
back to features via ``tokenizer.decode_tokens`` and accumulating the predicted standardized
``ret_close``. The cumulative is de-normalized by the decision-bar causal vol ``σ_t`` to a raw
expected log-return ``μ̂_{t,h}`` over ``[t+1, t+h]``.

**M5 decode contract (documented; M6 finalizes the head).** The decoded standardized ret_close is
treated as a vol-relative move and multiplied by the causal σ_t (the §3.4 decode contract). This is
a real model-driven prediction through the actual rollout path; the *numbers* are meaningless at M5
(dev-grade model, one symbol) — what M5 proves is that the machine runs and is leak-free.
"""

from __future__ import annotations

import numpy as np
import torch

from trikaal.data.config import BAR_MS
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE


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
) -> np.ndarray:
    """Greedy rollout μ̂_{t,h} (raw cumulative log-return) for each decision bar in ``decisions``.

    Each decision ``t`` must have ``t − seq_len + 1 ≥ 0`` and ``t + h`` in range. Returns
    ``μ̂[len(decisions)]``. Deterministic (greedy argmax), so trivially seed-stable.

    ``chunk`` bounds the KV-cache batch (the uncapped money grid is ~35k decisions/symbol; an
    unchunked cache at seq 512 would be tens of GB — the M5 cap=400 hid this, surfaced by the
    toy-rehearsal sizing and fixed pre-CUDA). Each decision's rollout is mathematically
    row-independent, but GEMM kernels are batch-SHAPE-dependent (ULP-level differences that an
    argmax near-tie could amplify), so bit-exactness across DIFFERENT chunk sizes is NOT
    claimed. The contract instead: ``chunk`` is a FIXED recorded constant of the recipe
    (default 512, never varied inside a run), under which replay is deterministic; cross-size
    agreement is KAT'd as approximate.
    """
    model.eval()
    tok.eval()
    max_len = int(getattr(model, "_config", {}).get("max_len", 0))
    if max_len and seq_len + h - 1 > max_len:
        # the rollout steps positions up to seq_len-1 + h-1; the RoPE cos/sin cache is built at
        # max_len, so exceeding it is an index crash (or silent garbage) at eval time. M5's
        # seq_len=128 never reached it; the money config (seq 512, h=60 → position 571) DOES —
        # caught at rehearsal sizing. RoPE caches are buffers, not params: raising max_len at
        # construction changes NO parameter count.
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
        )
        for c0 in range(0, d_all.shape[0], chunk)
    ]
    return np.concatenate(parts)


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
) -> np.ndarray:
    """One chunk of decisions through the KV-cache rollout (the original unchunked body)."""
    dev = torch.device(device)
    n = d.shape[0]
    # context windows [n, L] ending at each decision bar
    offs = np.arange(-seq_len + 1, 1)  # [-L+1 .. 0]
    idx = d[:, None] + offs[None, :]
    cc = torch.from_numpy(b_c[idx].astype(np.int64)).to(dev)
    cf = torch.from_numpy(b_f[idx].astype(np.int64)).to(dev)
    cts = torch.from_numpy(ts[idx].astype(np.int64)).to(dev)
    sig_t = sigma[d].astype(np.float64)

    caches = model.init_caches()
    out = None
    for p in range(seq_len):  # populate KV cache with the real context; out predicts bar t+1
        out = model.step(cc[:, p : p + 1], cf[:, p : p + 1], cts[:, p : p + 1], caches, p)

    cum = torch.zeros(n, dtype=torch.float32, device=dev)
    ts_dec = cts[:, -1:]  # decision-bar timestamp [n,1]
    for k in range(1, h + 1):
        pc = out.coarse_cond  # [n,1] greedy coarse
        pf = out.logits_f.argmax(dim=-1)  # [n,1] greedy fine | coarse
        x_hat = tok.decode_tokens(pc, pf)  # [n,1,16]
        # μ̂ covers [t+1, t+h] = log(C_{t+h}/C_{t+1}): EXCLUDE the k=1 (t→t+1) move — entry is at
        # C_{t+1}, so it is uncapturable (§8.B.3). The k=1 token still advances the cache.
        if k >= 2:
            cum = cum + x_hat[:, 0, 0]  # standardized ret_close (dim 0) for bars t+2..t+h
        if k < h:
            ts_k = ts_dec + k * BAR_MS
            out = model.step(pc, pf, ts_k, caches, seq_len - 1 + k)
    return cum.cpu().numpy().astype(np.float64) * sig_t  # vol-relative → raw (§3.4 contract)


def sigma_hat(sigma_t: np.ndarray, h: int) -> np.ndarray:
    """Point dispersion estimate over the horizon — random-walk scaling ``σ_t·√h`` (M5 contract)."""
    return np.asarray(sigma_t, dtype=np.float64) * np.sqrt(h)


__all__ = ["predict_mu", "sigma_hat"]
