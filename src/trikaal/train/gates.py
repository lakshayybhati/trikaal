"""Pre-flight gate runners (§7.0): G1 overfit-a-single-batch and G2 determinism smoke.

These are the launcher's hard pre-flight gates ("no GPU-hours until all gates pass"). They run
on a single fixed batch / a tiny model so the whole thing finishes in seconds on CPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import nn

from trikaal.utils.seeding import set_determinism


@dataclass
class OverfitResult:
    passed: bool
    metric: float  # achieved value of the gate metric (recon MAE or CE nats/token)
    steps: int
    target: float
    history: list[float] = field(default_factory=list)


def cosine_warmup_lr(
    step: int, peak: float, warmup: int, total: int, floor_frac: float = 0.1
) -> float:
    """Cosine decay with linear warmup (the shared §7.1 schedule)."""
    if step < warmup:
        return peak * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def overfit_single_batch(
    model: nn.Module,
    step_fn: Callable[[nn.Module], tuple[torch.Tensor, float]],
    *,
    target: float,
    max_steps: int = 2000,
    lr: float = 1e-3,
    warmup_frac: float = 0.0,
) -> OverfitResult:
    """Train ``model`` on a fixed batch until ``metric < target`` or ``max_steps``.

    ``step_fn(model) -> (loss, metric)``: one forward producing the differentiable loss and the
    scalar gate metric (e.g. reconstruction MAE, or cross-entropy in nats/token). Dropout / weight
    decay / augmentation are disabled by the caller (single-batch overfit, §7.0 G1). When
    ``warmup_frac > 0`` the §7.1 cosine-with-warmup schedule is used (peak = ``lr``).
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    warmup = int(warmup_frac * max_steps)
    history: list[float] = []
    metric = float("inf")
    step = 0
    for step in range(1, max_steps + 1):
        if warmup_frac > 0.0:
            for g in opt.param_groups:
                g["lr"] = cosine_warmup_lr(step, lr, warmup, max_steps)
        opt.zero_grad(set_to_none=True)
        loss, metric = step_fn(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        history.append(metric)
        if metric < target:
            break
    return OverfitResult(
        passed=metric < target, metric=metric, steps=step, target=target, history=history
    )


def determinism_smoke(
    build_and_run: Callable[[], list[float]], *, seed: int, steps: int = 200
) -> tuple[bool, float]:
    """G2: two back-to-back seeded runs must produce bit-identical loss curves.

    ``build_and_run()`` builds a fresh model and returns the per-step loss list (length ``steps``).
    Determinism is pinned around each call. Returns ``(passed, max_abs_divergence)``.
    """
    set_determinism(seed)
    curve_a = build_and_run()
    set_determinism(seed)
    curve_b = build_and_run()
    assert len(curve_a) == len(curve_b) == steps, "loss curves must have the same length"
    max_div = max(abs(a - b) for a, b in zip(curve_a, curve_b, strict=True))
    return (curve_a == curve_b), max_div
