"""Pre-flight gate runners (§7.0): G1 overfit-a-single-batch and G2 determinism smoke.

These are the launcher's hard pre-flight gates ("no GPU-hours until all gates pass"). They run
on a single fixed batch / a tiny model so the whole thing finishes in seconds on CPU.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
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


# ---- §7 v1.4 per-bar id-legibility instruments (the standing M6-time gate) ------------------
MICRO_DIMS = (7, 8, 9, 10, 11, 12)
MICRO_LEGIBILITY_MIN = 0.9


def digit_onehots(tok, b_c: np.ndarray, b_f: np.ndarray) -> np.ndarray:
    """One-hot per-digit decomposition of (c_id, f_id) — bar t's own id, linear-probeable.

    Radices come from the tokenizer's quantizer (FSQ level counts / BSQ bits), so the probe
    works identically for both arms."""
    q = tok.quant
    cols, radices = [], []
    for ids, group in ((b_c, list(q.coarse_idx)), (b_f, list(q.fine_idx))):
        radix = tok._group_radix(group)
        rem = ids.copy()
        digs = []
        for pos in reversed(range(len(group))):
            digs.append(rem % radix[pos])
            rem = rem // radix[pos]
        for d, r in zip(reversed(digs), radix, strict=True):
            cols.append(d)
            radices.append(r)
    feats = np.zeros((b_c.size, sum(radices)), dtype=np.float32)
    off = 0
    for col, r in zip(cols, radices, strict=True):
        feats[np.arange(b_c.size), off + col] = 1.0
        off += r
    return feats


def id_legibility_sign_acc(
    tok,
    values: np.ndarray,
    b_c: np.ndarray,
    b_f: np.ndarray,
    *,
    n: int = 150_000,
    groups: np.ndarray | None = None,
) -> float:
    """Logistic sign-accuracy of ``values[t]`` from bar t's OWN (c_id, f_id) digits.

    The §7 v1.4 legibility instrument: 80/20 split, 300 Adam steps, torch seed 0 —
    deterministic given its inputs.

    ``groups`` (§7 v1.6.15, C-10 leg 2) labels each row with the stratum it came from — in the
    M6 gate, its SYMBOL. When given, the 80/20 split is taken **within each stratum** rather than
    as one contiguous cut of the concatenation. Both properties matter and they pull in opposite
    directions:

    * the split must stay **BLOCKED IN TIME** — 1-minute bars are autocorrelated, so an
      interleaved or shuffled split puts near-duplicate neighbours on both sides and inflates
      accuracy. The contiguous cut had this right.
    * the split must **COVER THE UNIVERSE** — the contiguous cut did not. Measured on the real
      lake, the head-150k window spans **1 of 200 symbols**
      (``runs_manifest/m6_c10_micro_density.json``), so the gate that decides whether Stage-2
      spend proceeds was reading one symbol and calling it "the run's real training stream".

    Per-stratum blocking gives both: every symbol contributes, and within each symbol train is the
    earlier block and val the later one. ``groups=None`` preserves the old single-block behaviour
    for callers with one stream (``m6_canary.py``), where stratification is meaningless.
    """
    n = min(n, b_c.size)
    if groups is None:
        sel = np.arange(n)
        tr_mask = np.zeros(n, dtype=bool)
        tr_mask[: int(0.8 * n)] = True
    else:
        sel, tr_flags = _stratified_blocked_split(np.asarray(groups), n)
        tr_mask = tr_flags
    feats = digit_onehots(tok, b_c[sel], b_f[sel])
    y = (values[sel] > 0).astype(np.float32)
    xt = torch.from_numpy(feats[tr_mask])
    xv = torch.from_numpy(feats[~tr_mask])
    yt = torch.from_numpy(y[tr_mask])
    torch.manual_seed(0)
    logit = nn.Linear(feats.shape[1], 1)
    opt = torch.optim.Adam(logit.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(logit(xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(((logit(xv).squeeze(-1) > 0).numpy() == (y[~tr_mask] > 0.5)).mean())


def _stratified_blocked_split(groups: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(selected row indices, train mask) — an equal-ish quota per stratum, blocked in time.

    Deterministic: strata in first-appearance order, quotas by integer division with the
    remainder handed to the earliest strata, and within a stratum the rows are taken in their
    existing (time) order with the first 80% as train. No RNG, so the gate stays reproducible.
    """
    order, starts = [], {}
    for i, g in enumerate(groups):
        if g not in starts:
            starts[g] = i
            order.append(g)
    quota, extra = divmod(n, max(1, len(order)))
    sel_parts, tr_parts = [], []
    for k, g in enumerate(order):
        idx = np.flatnonzero(groups == g)
        take = min(len(idx), quota + (1 if k < extra else 0))
        if take <= 0:
            continue
        chosen = idx[:take]
        flags = np.zeros(take, dtype=bool)
        flags[: max(1, int(0.8 * take))] = True
        sel_parts.append(chosen)
        tr_parts.append(flags)
    if not sel_parts:  # fail closed rather than return an empty design matrix
        raise ValueError("stratified split selected no rows — the grouping carried no members")
    return np.concatenate(sel_parts), np.concatenate(tr_parts)


def micro_legibility_gate(
    tok,
    per_symbol,
    tokens: dict,
    *,
    min_acc: float = MICRO_LEGIBILITY_MIN,
    run_name: str = "",
) -> dict:
    """The STANDING M6-time per-bar legibility gate (gate-2 ruling, 2026-07-20): after
    Stage-1 and BEFORE any Stage-2 AR spend, each of the SIX MICRO DIMS (7-12) must be
    linearly recoverable from bar t's own id at logistic sign-accuracy >= ``min_acc`` on
    the run's real training stream. Raises RuntimeError on failure — a hard stop.

    SIGN-ACCURACY SEMANTICS FOR THE MAGNITUDE DIMS (gate-2 final ruling, item 4): every
    micro feature enters the model Z-SCORED (§3.2), so ``sign(x_t[dim])`` means
    above-vs-below the causal rolling mean — a balanced, meaningful binary for the
    magnitude-shaped dims (trade count, mean trade size, size dispersion, large-trade
    share) exactly as for the signed dims (TFI). Per-dim base rates are recorded in the
    receipt so any residual class imbalance is visible; only unmasked bars count per dim
    (mask-aware), and a dim masked ~everywhere is SKIPPED with a named receipt entry,
    never trivially passed.

    FIRST REAL-DATA EXECUTION IS A NAMED CHECKPOINT (gate-2 final ruling): if the real
    micro dims cannot clear ``min_acc`` even under the calibrated micro-weighted
    bottleneck leg (§7 v1.4.1), STOP and report with the per-dim receipts — that is a
    re-adjudication, never a silent threshold move.

    FALLBACK HISTORY: the micro-weighted ``w_feat`` bottleneck-leg fallback originally
    recorded here fired under §7 v1.4.1 (2026-07-20) on a trigger STRICTLY STRONGER than
    the one written (deterministic FIXTURE-gate failure with mechanism receipts, 3/3
    seeds, vs the original real-data failure) — see the dated prereg entry."""
    per_dim: dict[str, dict] = {}
    ok = True
    unmeasured: list[int] = []
    for dim in MICRO_DIMS:
        vals, bcs, bfs, grps = [], [], [], []
        for sw in per_symbol:
            b_c, b_f = tokens[sw.symbol]
            keep = sw.mask[:, dim] == 0
            vals.append(sw.x[keep, dim])
            bcs.append(b_c[keep])
            bfs.append(b_f[keep])
            # §7 v1.6.15 C-10 leg 2: carry the SYMBOL through so the probe can stratify by it.
            grps.append(np.full(int(keep.sum()), sw.symbol, dtype=object))
        v = np.concatenate(vals)
        if v.size < 10_000:
            # §7 v1.6.15 C-10 leg 1: SKIPPED IS A THIRD STATE AND IT HALTS. The predecessor
            # `continue`d BEFORE `ok = ok and acc >= min_acc`, so a skipped dim could not lower
            # `ok` — all six thin returned pass=True having measured nothing, and five thin plus
            # one dense passing dim returned pass=True on a gate whose contract is all six. Both
            # reproduced (runs_manifest/m6_tier4_vacuous_gates.json). "We could not measure the
            # channel this whole gate exists to measure" is not a pass.
            per_dim[str(dim)] = {"skipped": "masked (n < 10k unmasked bars)", "n": int(v.size)}
            unmeasured.append(dim)
            ok = False
            continue
        acc = id_legibility_sign_acc(
            tok, v, np.concatenate(bcs), np.concatenate(bfs), groups=np.concatenate(grps)
        )
        per_dim[str(dim)] = {
            "sign_acc": round(acc, 4),
            "n": int(min(150_000, v.size)),
            "n_symbols_in_sample": len({sw.symbol for sw in per_symbol}),
            "base_rate_positive": round(float((v > 0).mean()), 4),
        }
        ok = ok and acc >= min_acc
    receipt = {
        "min_acc": min_acc,
        "per_dim": per_dim,
        "pass": bool(ok),
        "unmeasured_dims": unmeasured,
        "sampling": "stratified by symbol, 80/20 blocked in time WITHIN each symbol (§7 v1.6.15)",
    }
    if not ok:
        why = (
            f"{len(unmeasured)} dim(s) could not be measured at all {unmeasured}"
            if unmeasured
            else "at least one dim fell below min_acc"
        )
        raise RuntimeError(
            f"{run_name}: §7 v1.4 MICRO LEGIBILITY GATE FAILED before Stage-2 ({why}) — {receipt} "
            "(pre-authorized fallback: micro-weighted w_feat in the bottleneck leg, "
            "real-data failures only, dated §7 entry required BEFORE use)"
        )
    return receipt


# ---------------------------------------------------------------- §7 v1.5 item E: the λ slice
LAMBDA_CALIBRATION_SLICE_FRAC = 0.10  # the LAST 10% of the TRAIN region, never an eval block


def lambda_calibration_boundary_ms(
    train_start_ms: int, train_end_ms: int, *, frac: float = LAMBDA_CALIBRATION_SLICE_FRAC
) -> int:
    """Start of the held-out slice carved off the END of the TRAIN region (§7 v1.5 item E rule 3).

    WHY NOT BLOCK 0. If the micro-legibility gate fires, λ may be re-derived ONCE — and the obvious
    place, eval block 0, is wrong twice over. (1) Block 0 already carries κ* selection AND every
    clause-5 DSR trial entry; λ would be a THIRD use of one validation block, and repeated reuse of
    a validation set is a known overfitting channel. (2) λ is a TRAINING hyperparameter and the gate
    fires during training, before any scoring — using forward-region data to set it inverts the
    train/eval separation for no reason. Verified when this was written: the real run has NO
    in-train holdout (``build_symbol_windows`` takes the train/eval calendar split as its
    boundary, and ``HOLD_FRAC`` exists only in ``scripts/m6_canary.py``), so the slice has to be
    created — this is that creation.

    INERT BY DEFAULT: nothing calls this unless the gate fires. It does not move the training
    boundary of a passing run, and it never touches a scored block.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"slice fraction must be in (0,1), got {frac}")
    if train_end_ms <= train_start_ms:
        raise ValueError(f"empty train region: [{train_start_ms}, {train_end_ms})")
    span = train_end_ms - train_start_ms
    return int(train_end_ms - int(span * frac))
