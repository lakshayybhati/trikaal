"""Pre-flight gate runner (§7.0) — G0 / G1 / G2, the launcher's hard pre-flight.

No GPU-hours are spent until all gates pass. Run before any full training:

    python scripts/preflight.py

Exits non-zero if any gate fails. Mirrors the test suite but prints a single readable summary.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from trikaal.data.causal_check import check_causal_safety, exhaustive_truncation_sweep
from trikaal.data.config import synthetic_test_config
from trikaal.data.features import compute_features
from trikaal.data.synthetic import make_synthetic_stream
from trikaal.model.predictor import TrikaalAR
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.gates import determinism_smoke, overfit_single_batch
from trikaal.utils.seeding import set_determinism

LEAKS = ("centered_zscore", "global_zscore", "sigma_includes_next", "forward_revert_badtick")


def gate_g0() -> bool:
    cfg = synthetic_test_config(half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20)
    stream, anom = make_synthetic_stream(seed=7, n_bars=600)
    # the GATE: exhaustive truncation sweep (every bar), full coverage is the criterion
    gate = exhaustive_truncation_sweep(stream, cfg)
    print(f"  G0 exhaustive sweep (real transforms): {gate.summary()}")
    # added probe: adversarial perturbation on the high-risk strata
    probe = check_causal_safety(stream, cfg, anom, seed=1337, n_per_stratum=5)
    print(f"  G0 perturbation probe: passed={probe.passed}")
    leaks_caught = []
    for leak in LEAKS:
        rep = check_causal_safety(
            stream,
            cfg.with_leak(leak),
            anom,
            seed=1337,
            n_per_stratum=5,
            stop_on_first_failure=True,
        )
        leaks_caught.append(not rep.passed)
        print(f"    planted leak '{leak}': {'CAUGHT' if not rep.passed else 'MISSED'}")
    return gate.passed and gate.coverage == 1.0 and probe.passed and all(leaks_caught)


def gate_g1_stage1() -> bool:
    set_determinism(0)
    cfg = synthetic_test_config(half_life_fast=48, half_life_slow=96, n_warm=12, w_tau=20)
    stream, _ = make_synthetic_stream(seed=11, n_bars=2400)
    out = compute_features(stream, cfg)
    seg_ids, counts = np.unique(out.segment_id, return_counts=True)
    idx = np.nonzero(out.segment_id == seg_ids[int(np.argmax(counts))])[0][100:]
    sel = idx[:32]
    x = torch.from_numpy(out.x[sel]).view(2, 16, 16)
    m = torch.from_numpy(out.m[sel].astype(np.float32)).view(2, 16, 16)
    model = TokenizerAE()
    res = overfit_single_batch(
        model,
        lambda mdl: (mdl(x, m)["loss"], float(mdl(x, m)["recon_mae"])),
        target=1e-3,
        max_steps=5000,
        lr=3e-3,
        warmup_frac=0.05,
    )
    print(
        f"  G1 stage-1 (tokenizer overfit): recon_mae={res.metric:.2e} @ step {res.steps} "
        f"-> {'PASS' if res.passed else 'FAIL'}"
    )
    return res.passed


def gate_g1_stage2() -> bool:
    set_determinism(0)
    tok = TokenizerAE().eval()
    x = torch.randn(2, 24, 16)
    with torch.no_grad():
        b_c, b_f = tok.encode_tokens(x, torch.zeros(2, 24, 16))
    ts = (1_600_000_000_000 + torch.arange(24).repeat(2, 1) * 60000).long()
    set_determinism(0)
    model = TrikaalAR(mtp_depths=4)
    res = overfit_single_batch(
        model,
        lambda mdl: (
            mdl.compute_loss(b_c, b_f, ts)["loss"],
            float(mdl.compute_loss(b_c, b_f, ts)["nll_main"]),
        ),
        target=0.05,
        max_steps=2000,
        lr=1e-3,
    )
    print(
        f"  G1 stage-2 (backbone+MTP overfit): nll_main={res.metric:.3e} @ step {res.steps} "
        f"-> {'PASS' if res.passed else 'FAIL'}"
    )
    return res.passed


def gate_g2() -> bool:
    rng = np.random.default_rng(0)
    b_c = torch.from_numpy(rng.integers(0, 891, (2, 24)))
    b_f = torch.from_numpy(rng.integers(0, 1225, (2, 24)))
    ts = (1_600_000_000_000 + torch.arange(24).repeat(2, 1) * 60000).long()

    def run() -> list[float]:
        model = TrikaalAR(mtp_depths=2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        model.train()
        for _ in range(40):
            opt.zero_grad(set_to_none=True)
            out = model.compute_loss(b_c, b_f, ts)
            out["loss"].backward()
            opt.step()
            losses.append(float(out["loss"].detach()))
        return losses

    passed, max_div = determinism_smoke(run, seed=2024, steps=40)
    print(f"  G2 determinism: max|Δ|={max_div:.1e} -> {'PASS' if passed else 'FAIL'}")
    return passed


def main() -> int:
    print("=== Trikaal pre-flight gates (§7.0) ===")
    results = {
        "G0": gate_g0(),
        "G1-stage1": gate_g1_stage1(),
        "G1-stage2": gate_g1_stage2(),
        "G2": gate_g2(),
    }
    print("\n=== summary ===")
    for name, ok in results.items():
        print(f"  {name:12s} {'PASS' if ok else 'FAIL'}")
    passed = all(results.values())
    print(f"\npreflight.passed = {passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
