"""Purged walk-forward folds — train ONCE, evaluate forward (§8.A).

One anchored train span ``[0, train_end)`` followed by ``K=6`` contiguous forward eval blocks over
``[train_end, N)``. The single trained checkpoint is scored on every forward block (no per-fold
retraining). Leakage is controlled at the one train→eval boundary by:

* **PURGE** (§A.3): drop a train sample ``t`` whose feature-or-label interval ``[t−L+1, t+H_max]``
  reaches into the eval region — i.e. ``t + H_max ≥ train_end``.
* **LEADING EMBARGO** (§A.4): additionally drop train anchors in ``[train_end − E, train_end − 1]``
  (``E = H_max + L_corr = 120`` bars) to absorb serial correlation past the label horizon. Since
  ``E > H_max`` the embargo binds, leaving valid train anchors ``t < train_end − E``.

Q1–Q4 (§A.5) cross symbol-holdout × time-holdout. The M6 PRIMARY is **train-universe symbols ×
future blocks 1–5** (prereg §3a: the pinned 40-symbol set, VAL block 0 excluded) — Q2 territory;
"Q4 is the headline" was the M5-era label, and Q3/Q4 (held-out symbols) are secondary robustness
reads.

**§7 v1.6.16 — THE embargo_flatness GATE WAS DELETED, AND THIS PARAGRAPH ADVERTISED IT.**
It was an EMPIRICAL leakage gate (run the headline at E in {60,120,240}, require IR flat past 120)
whose only caller was ever its own test. Wiring it in costs 3x the M6 run, because the embargo
binds ``fold_valid_starts`` and therefore which TRAINING windows are legal — each E needs its own
training. What remains, and what the leakage control actually is:

* the **STRUCTURAL** embargo above, enforced by ``universe_loader.fold_valid_starts`` on EVERY
  training window at load time, not sampled and not optional; and
* its PREMISE, now MEASURED rather than assumed (``runs_manifest/m6_c20_embargo_premise.json``):
  signed-return autocorrelation on the real lake is **+0.0061 mean / 0.0127 worst-symbol at lag
  60**, already ~0 by lag 5, so ``L_corr = 60`` carries roughly a 24x margin over the channel an
  embargo must outrun. (|return| ACF is 0.20 at lag 60 — long memory, but that is volatility
  clustering, not a label-leakage channel, and an embargo sized to it would run to days.)

WHAT IS LOST, STATED PLAINLY: the end-to-end flatness of the headline IR in E is now ASSERTED from
the premise rather than DEMONSTRATED. A leakage channel that is not signed-return autocorrelation
would not be caught by the argument above. The instrument is recoverable from git history if the
3x run is ever funded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

H_MAX_DEFAULT = 60  # autoregressive-rollout label look-forward (drives the purge)
L_CORR_DEFAULT = 60  # 1h serial-correlation margin
EMBARGO_DEFAULT = H_MAX_DEFAULT + L_CORR_DEFAULT  # E = 120 bars (2h)


@dataclass(frozen=True)
class FoldPlan:
    n: int  # total decision bars
    train_end: int  # first index of the forward eval region
    blocks: list[tuple[int, int]]  # K forward eval blocks [lo, hi)
    seq_len: int  # L — feature-window length (for the purge)
    embargo: int = EMBARGO_DEFAULT
    h_max: int = H_MAX_DEFAULT

    def valid_train_anchors(self, anchors: np.ndarray) -> np.ndarray:
        """Train anchors surviving purge + leading embargo: ``t < train_end − E`` (E ≥ H_max)."""
        a = np.asarray(anchors, dtype=np.int64)
        return a[a < self.train_end - self.embargo]

    def purged_only_anchors(self, anchors: np.ndarray) -> np.ndarray:
        """Train anchors surviving the PURGE alone (no embargo): ``t + H_max < train_end``."""
        a = np.asarray(anchors, dtype=np.int64)
        return a[a + self.h_max < self.train_end]

    def block_anchors(self, anchors: np.ndarray, k: int) -> np.ndarray:
        """Eval anchors whose decision bar falls in forward block ``k``."""
        a = np.asarray(anchors, dtype=np.int64)
        lo, hi = self.blocks[k]
        return a[(a >= lo) & (a < hi)]


def make_fold_plan(
    n: int,
    *,
    train_frac: float = 0.7,
    k: int = 6,
    seq_len: int = 512,
    embargo: int = EMBARGO_DEFAULT,
) -> FoldPlan:
    """Anchored train span + ``k`` equal contiguous forward blocks (last absorbs the remainder)."""
    train_end = int(n * train_frac)
    span = n - train_end
    if span < k:
        raise ValueError(f"forward region ({span}) too short for K={k} blocks")
    edges = [train_end + (span * i) // k for i in range(k)] + [n]
    blocks = [(edges[i], edges[i + 1]) for i in range(k)]
    return FoldPlan(n=n, train_end=train_end, blocks=blocks, seq_len=seq_len, embargo=embargo)


def quadrant(symbol_in_train: bool, time_in_train: bool) -> str:
    """§A.5 two-axis label. Q1 in-sample sanity; Q2 temporal-OOS (the M6 primary region lives
    here, prereg §3a); Q3 cross-sectional-OOS; Q4 held-out symbols × future block (a secondary
    robustness read — its M5-era "headline" label is superseded)."""
    if symbol_in_train and time_in_train:
        return "Q1"
    if symbol_in_train and not time_in_train:
        return "Q2"
    if not symbol_in_train and time_in_train:
        return "Q3"
    return "Q4"


__all__ = [
    "EMBARGO_DEFAULT",
    "H_MAX_DEFAULT",
    "L_CORR_DEFAULT",
    "FoldPlan",
    "make_fold_plan",
    "quadrant",
]
