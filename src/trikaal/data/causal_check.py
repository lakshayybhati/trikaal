"""Lookahead-purity harness — the transform-agnostic merge gate (feature-spec §6 / §2.6.2).

Enforces the single testable invariant: every pipeline output for bar ``t`` is a pure function
of raw data with effective timestamp ``<= t+1``. Concretely, each output carries a **read
horizon** — the highest bar index it is allowed to depend on:

  * inputs ``x``, mask ``m``, ``segment_id``, the vol-hook ``raw``, and the divisor ``sigma``
    → horizon ``t`` (may read bars ``<= t`` only);
  * the **target** label ``y_{t→t+1}`` → horizon ``t+1`` (the bar being predicted is a label,
    which is allowed to be future; the divisor inside it is still causal at ``t``).

For each sampled anchor and each output, the harness (1) **truncates** the raw stream to the
output's horizon and asserts the value at ``t`` is byte-identical, and (2) **perturbs** all raw
data strictly beyond the horizon (noise / fabricated revert / sign-flip) and asserts the same.
Any transform that reads beyond its horizon fails. Sampling is **not** uniform: it targets the
high-risk strata (bad-tick-adjacent, segment-boundary, structural-break, stale-run) plus a
random background, because that is where future-dependence hides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trikaal.data.config import FeatureConfig
from trikaal.data.features import PerBarOutput, compute_features
from trikaal.data.synthetic import AnomalyIndex, RawStream

# Each output: (name, attribute, horizon_offset). The output for bar t is fully determined at
# raw-truncation boundary k = t + horizon_offset (inputs/divisor read bars <= t; the target
# LABEL legitimately reads bar t+1, so target/target_valid have offset 1). EVERY model-visible
# input (x, m, ts, raw) plus segment_id, the divisor sigma, and the loss-gating target_valid are
# covered — there is no enumerated "safe transform" carve-out (§6 is transform-agnostic).
_OUTPUTS: tuple[tuple[str, str, int], ...] = (
    ("x", "x_f64", 0),
    ("m", "m", 0),
    ("segment_id", "segment_id", 0),
    ("raw", "raw", 0),
    ("sigma", "sigma", 0),
    ("ts", "ts", 0),  # model input (temporal embedding key) — must be causal too
    ("target", "target", 1),
    ("target_valid", "target_valid", 1),  # the loss-inclusion flag — a survivorship surface
)
_RADIUS = 3  # high-risk region half-width around each flagged bar
_PERTURB_MODES = ("noise", "revert", "signflip")


@dataclass
class CausalReport:
    passed: bool
    n_anchors: int
    n_checks: int
    strata_counts: dict[str, int]
    failures: list[dict] = field(default_factory=list)
    coverage_bars: int = 0  # distinct bars whose outputs were checked
    total_bars: int = 0

    @property
    def coverage(self) -> float:
        return self.coverage_bars / self.total_bars if self.total_bars else 0.0

    def summary(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        strata = ", ".join(f"{k}={v}" for k, v in self.strata_counts.items())
        cov = f"coverage={self.coverage_bars}/{self.total_bars} ({100 * self.coverage:.0f}%)"
        s = f"[causal-safety {head}] {cov} checks={self.n_checks} | {strata}"
        if self.failures:
            f0 = self.failures[0]
            s += f"\n  first failure: output={f0['output']} t={f0['t']} mode={f0['mode']} "
            s += f"max|Δ|={f0['delta']:.3e}"
        return s


def _bit_equal(a: np.ndarray | float, b: np.ndarray | float) -> tuple[bool, float]:
    """Exact float equality (NaN==NaN), returning (equal, max_abs_diff over finite entries)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False, float("inf")
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False, float("inf")
    fin = ~nan_a
    eq = bool(np.array_equal(a[fin], b[fin]))
    delta = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
    return eq, delta


def build_anchor_strata(
    out: PerBarOutput, anom: AnomalyIndex, rng: np.random.Generator, n_per_stratum: int
) -> dict[str, list[int]]:
    """Stratified anchor sets. Anchors clamped to ``[1, T-2]`` so ``t-1`` and ``t+1`` exist."""
    T = len(out)
    lo, hi = 1, T - 2

    def expand(centers: list[int]) -> list[int]:
        s: set[int] = set()
        for c in centers:
            for d in range(-_RADIUS, _RADIUS + 1):
                if lo <= c + d <= hi:
                    s.add(c + d)
        return sorted(s)

    def clamp(centers: list[int]) -> list[int]:
        return [int(c) for c in centers if lo <= c <= hi]

    seg_starts = [0, *(np.nonzero(np.diff(out.segment_id) != 0)[0] + 1).tolist()]
    seg_boundary_centers = clamp(seg_starts) + [s - 1 for s in seg_starts if lo <= s - 1 <= hi]

    # (must_include_centers, full_candidate_pool) — centers are ALWAYS sampled (the exact bars
    # where a localized future-dependent gate fires), the rest fill the radius neighbourhood.
    plan: dict[str, tuple[list[int], list[int]]] = {
        "bad_tick_adjacent": (clamp(anom.bad_tick), expand(anom.bad_tick)),
        "segment_boundary": (
            sorted(set(seg_boundary_centers)),
            expand(seg_starts) + seg_boundary_centers,
        ),
        "structural_break": (clamp(anom.structural_break), expand(anom.structural_break)),
        "stale_run": (clamp(anom.stale_run[:1] + anom.stale_run[-1:]), expand(anom.stale_run)),
        "random_background": ([], list(range(lo, hi + 1))),
    }
    capped: dict[str, list[int]] = {}
    for name, (must, pool) in plan.items():
        must = sorted(set(must))[:n_per_stratum]
        remaining = sorted(set(int(i) for i in pool) - set(must))
        need = max(0, n_per_stratum - len(must))
        if remaining and need:
            sel = rng.choice(np.asarray(remaining), size=min(need, len(remaining)), replace=False)
            must = sorted(set(must) | {int(i) for i in sel})
        capped[name] = must
    return capped


def _value_at(out: PerBarOutput, attr: str, t: int):
    arr = getattr(out, attr)
    return arr[t]


def check_causal_safety(
    stream: RawStream,
    cfg: FeatureConfig,
    anom: AnomalyIndex,
    *,
    seed: int = 1337,
    n_per_stratum: int = 6,
    modes: tuple[str, ...] = _PERTURB_MODES,
    stop_on_first_failure: bool = False,
) -> CausalReport:
    """Run the truncation + perturbation invariant over the stratified anchor set.

    ``stop_on_first_failure`` short-circuits once any leak is caught (used by the planted-leak
    tests, which only need to demonstrate the harness has teeth).
    """
    rng = np.random.default_rng(seed)
    full = compute_features(stream, cfg)
    strata = build_anchor_strata(full, anom, rng, n_per_stratum)
    all_anchors = sorted({t for idxs in strata.values() for t in idxs})

    failures: list[dict] = []
    n_checks = 0
    # group outputs by horizon so we recompute each truncated/perturbed stream once per horizon
    for t in all_anchors:
        for horizon_off in (0, 1):
            outs = [o for o in _OUTPUTS if o[2] == horizon_off]
            h = t + horizon_off
            if h >= len(stream):
                continue
            # truncation
            trunc = compute_features(stream.truncate_to_bar(h), cfg)
            # perturbations (independent recompute per mode)
            perts = {m: compute_features(stream.perturb_after_bar(h, rng, m), cfg) for m in modes}

            for name, attr, _ in outs:
                # Compare EVERY output (no skipping). _bit_equal is NaN-aware, so an invalid
                # target (NaN in both) compares equal; a future-conditioned flip of target_valid
                # or a corrupted target is caught. We never use target_valid to gate a check —
                # that coupling would let a validity leak mask a target leak.
                ref = _value_at(full, attr, t)
                for label, cand in (("truncate", trunc), *perts.items()):
                    n_checks += 1
                    eq, delta = _bit_equal(_value_at(cand, attr, t), ref)
                    if not eq:
                        failures.append(
                            {"output": name, "t": int(t), "mode": label, "delta": delta}
                        )
            if stop_on_first_failure and failures:
                break
        if stop_on_first_failure and failures:
            break
    counts = {k: len(v) for k, v in strata.items()}
    return CausalReport(
        passed=len(failures) == 0,
        n_anchors=len(all_anchors),
        n_checks=n_checks,
        strata_counts=counts,
        failures=failures,
        coverage_bars=len(all_anchors),
        total_bars=len(stream),
    )


def exhaustive_truncation_sweep(
    stream: RawStream, cfg: FeatureConfig, *, stride: int = 1, stop_on_first_failure: bool = False
) -> CausalReport:
    """The merge gate: truncate the raw stream at EVERY bar boundary and assert each output is
    byte-identical to the full-history build at the bar it is just-determined for.

    Coverage is the gating criterion, not a sparse stochastic sample. At boundary ``k`` the stream
    is truncated to bars ``[0..k]``; an output with horizon offset ``o`` is fully determined for bar
    ``t = k - o`` (it may read bars ``<= k``). If a transform at bar ``t`` reads any bar ``> t+o``,
    truncating to ``k = t+o`` removes that bar and the recomputed value diverges — caught here for
    EVERY bar (any look-ahead distance ``>= 1``), which a sparse anchor sample cannot guarantee.

    ``stride > 1`` trades coverage for speed; ``stride == 1`` is the only setting that guarantees
    catching a single-bar, look-ahead-distance-1 leak, so it is the default and the CI gate value.
    """
    full = compute_features(stream, cfg)
    total = len(stream)
    failures: list[dict] = []
    n_checks = 0
    covered: set[int] = set()
    for k in range(0, total, stride):
        trunc = compute_features(stream.truncate_to_bar(k), cfg)
        for name, attr, horizon_off in _OUTPUTS:
            t = k - horizon_off
            if t < 0:
                continue
            covered.add(t)
            n_checks += 1
            eq, delta = _bit_equal(getattr(trunc, attr)[t], getattr(full, attr)[t])
            if not eq:
                failures.append({"output": name, "t": int(t), "mode": "truncate", "delta": delta})
        if stop_on_first_failure and failures:
            break
    return CausalReport(
        passed=len(failures) == 0,
        n_anchors=len(covered),
        n_checks=n_checks,
        strata_counts={"exhaustive_truncation": len(covered)},
        failures=failures,
        coverage_bars=len(covered),
        total_bars=total,
    )
