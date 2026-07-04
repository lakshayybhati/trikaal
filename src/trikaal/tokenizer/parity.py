"""G-parity — the cross-arm fairness gate (m6_design §3, invariant 6).

The 2×2 ablation is fair only if the BSQ and FSQ arms are matched on the two capacity axes a
tokenizer controls: **bits-per-token** (the token budget) and **total parameters** (the model
budget). This check is asserted at CELL CONSTRUCTION, before any training step — a violated
parity invalidates every cross-cell comparison, so it fails loudly and immediately.
"""

from __future__ import annotations

from torch import nn

from trikaal.constants import FSQ_BPT_BAND


class GParityError(AssertionError):
    """A cell pair violates the pre-registered bpt/param parity — the ablation would be unfair."""


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def check_arm_parity(
    tok_a: nn.Module,
    tok_b: nn.Module,
    *,
    max_bpt_delta: float = 0.5,
    max_param_frac: float = 0.02,
) -> dict[str, float]:
    """Assert the G-parity gate between two tokenizer arms; returns the measured numbers.

    Gates (m6_design §3 G-parity): both arms' bpt inside ``FSQ_BPT_BAND`` (19.5–20.5),
    ``|Δbpt| ≤ max_bpt_delta`` (0.5 bit), and total-parameter counts within ``±max_param_frac``
    (2 %) of each other. Raises :class:`GParityError` on any violation.
    """
    bpt_a, bpt_b = float(tok_a.quant.bpt), float(tok_b.quant.bpt)
    lo, hi = FSQ_BPT_BAND
    for name, bpt in (("A", bpt_a), ("B", bpt_b)):
        if not (lo <= bpt <= hi):
            raise GParityError(f"arm {name} bpt {bpt:.2f} outside FSQ_BPT_BAND [{lo}, {hi}]")
    d_bpt = abs(bpt_a - bpt_b)
    if d_bpt > max_bpt_delta:
        raise GParityError(
            f"|Δbpt| = {d_bpt:.2f} > {max_bpt_delta} (bpt {bpt_a:.2f} vs {bpt_b:.2f})"
        )
    p_a, p_b = n_params(tok_a), n_params(tok_b)
    frac = abs(p_a - p_b) / max(p_a, p_b)
    if frac > max_param_frac:
        raise GParityError(
            f"param parity {frac:.3%} > {max_param_frac:.0%} ({p_a:,} vs {p_b:,}) — "
            "arm capacities are not matched"
        )
    return {
        "bpt_a": bpt_a,
        "bpt_b": bpt_b,
        "d_bpt": d_bpt,
        "params_a": p_a,
        "params_b": p_b,
        "param_frac": frac,
    }


__all__ = ["GParityError", "check_arm_parity", "n_params"]
