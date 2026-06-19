"""KV-cached single-step rollout must equal the parallel causal forward (Backbone §5/§6).

The M5/M6 5/15/60-min horizons come from autoregressive rollout, so the cached ``step`` path
must be bit-equivalent (to fp tolerance) to the parallel ``forward`` it replaces — at the trunk
state, the coarse logits, the conditioning coarse, and the fine logits.
"""

from __future__ import annotations

import torch

from trikaal.data.config import BAR_MS
from trikaal.model.predictor import TrikaalAR
from trikaal.utils.seeding import set_determinism


def _ts(b: int, length: int) -> torch.Tensor:
    return (1_600_000_000_000 + torch.arange(length).repeat(b, 1) * BAR_MS).long()


def _rollout(model: TrikaalAR, b_c: torch.Tensor, b_f: torch.Tensor, ts: torch.Tensor):
    caches = model.init_caches()
    hs, lcs, lfs, ccs = [], [], [], []
    for pos in range(b_c.shape[1]):
        o = model.step(
            b_c[:, pos : pos + 1], b_f[:, pos : pos + 1], ts[:, pos : pos + 1], caches, pos
        )
        hs.append(o.h_final)
        lcs.append(o.logits_c)
        lfs.append(o.logits_f)
        ccs.append(o.coarse_cond)
    return (
        torch.cat(hs, dim=1),
        torch.cat(lcs, dim=1),
        torch.cat(lfs, dim=1),
        torch.cat(ccs, dim=1),
    )


def test_kv_cache_rollout_matches_parallel_forward():
    set_determinism(0)
    model = TrikaalAR(mtp_depths=0, max_len=64).eval()
    b, length = 2, 40
    b_c = torch.randint(0, model.v_c, (b, length))
    b_f = torch.randint(0, model.v_f, (b, length))
    ts = _ts(b, length)
    with torch.no_grad():
        par = model(b_c, b_f, ts)  # coarse_cond=None → argmax, matches step's default
        h_inc, lc_inc, lf_inc, cc_inc = _rollout(model, b_c, b_f, ts)
    dh = (par.h_final - h_inc).abs().max().item()
    dlc = (par.logits_c - lc_inc).abs().max().item()
    dlf = (par.logits_f - lf_inc).abs().max().item()
    assert torch.equal(par.coarse_cond, cc_inc), "cached argmax-coarse diverged from parallel"
    assert dh < 1e-4, f"trunk diverged: max|Δ|={dh:.2e}"
    assert dlc < 1e-4, f"coarse logits diverged: max|Δ|={dlc:.2e}"
    assert dlf < 1e-4, f"fine logits diverged: max|Δ|={dlf:.2e}"


def test_kv_cache_rollout_works_with_mtp_built():
    """MTP heads are training-only; the cached rollout must still match forward when D>0."""
    set_determinism(0)
    model = TrikaalAR(mtp_depths=4, max_len=64).eval()
    b, length = 1, 24
    b_c = torch.randint(0, model.v_c, (b, length))
    b_f = torch.randint(0, model.v_f, (b, length))
    ts = _ts(b, length)
    with torch.no_grad():
        par = model(b_c, b_f, ts)
        h_inc, _, _, _ = _rollout(model, b_c, b_f, ts)
    assert (par.h_final - h_inc).abs().max().item() < 1e-4
