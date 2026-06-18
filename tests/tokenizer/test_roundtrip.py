"""G1 stage-1 — overfit a single batch to reconstruction MAE < 1e-3 (§7.0).

The gate proves the FSQ tokenizer can drive a single fixed batch to near-zero reconstruction
error — i.e. the STE, the masked Huber loss, and the coarse/fine decode are correctly wired.
If it floored above 1e-3 the FSQ levels would be too coarse or the STE mis-wired (§7.0 G1).

Note on the step budget: the synthetic fixture is deliberately high-entropy (many independent
random microstructure channels), so reaching the strict 1e-3 takes more steps than the ~2k the
spec anticipates on lower-entropy real K-lines. We keep the strict 1e-3 threshold and allow a
larger cap on the synthetic fixture; the architectural property under test is identical.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.gates import overfit_single_batch
from trikaal.utils.seeding import set_determinism


@pytest.mark.gate
@pytest.mark.slow
def test_stage1_overfit_to_recon_mae_below_1e_3(feature_window):
    set_determinism(0)
    x_np, m_np = feature_window  # [256, 16] real synthetic features from one clean segment
    b, length = 2, 16  # one fixed 32-bar batch
    x = torch.from_numpy(x_np[: b * length]).view(b, length, 16)
    m = torch.from_numpy(m_np[: b * length].astype(np.float32)).view(b, length, 16)

    model = TokenizerAE()  # canonical FSQ [11,9,9,7,7,5,5], per-stage scale on

    def step_fn(mdl: TokenizerAE) -> tuple[torch.Tensor, float]:
        out = mdl(x, m)
        return out["loss"], float(out["recon_mae"])

    res = overfit_single_batch(
        model, step_fn, target=1e-3, max_steps=5000, lr=3e-3, warmup_frac=0.05
    )
    assert res.passed, (
        f"recon MAE floored at {res.metric:.2e} after {res.steps} steps — "
        "FSQ levels too coarse or STE mis-wired"
    )
    # sanity: it really overfit (dropped >2 orders of magnitude from the early steps)
    assert res.history[0] > 1e-1
