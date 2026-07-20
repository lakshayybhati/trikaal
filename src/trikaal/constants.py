"""Canonical numerical constants and layout for Trikaal.

Single source of truth for the feature-spec §5 stability epsilons, the §1
16-feature layout (names + index groups), and the §4 canonical FSQ config. Every
downstream module imports from here; nothing restates a divergent number.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------------------
# Feature-spec §5 — numerical constants & stability
# --------------------------------------------------------------------------------------
EPS_PX: float = 1e-8  # price-range denominators (features 3-5); guards H==L flat bars
EPS_CNT: float = 1e-9  # TFI / signed-count imbalance; guards zero-trade bar
EPS_SZ: float = 1e-9  # mean/dispersion/large-share denominators; guards zero base-volume
EPS_OI: float = 1.0  # d_oi log-ratio offset; kept consistent with log_oi's +1 form
EPS_VAR: float = 1e-8  # all z-scores; guards zero-variance warm-up
EPS_VOL: float = 1e-8  # vol-scaling variance floor (§3 vol-scaling)
CLIP_LO: float = -5.0  # Kronos-style outlier guard, applied last
CLIP_HI: float = 5.0

# --------------------------------------------------------------------------------------
# Feature-spec §1 — the 16-feature layout (0-indexed x_0..x_15)
# --------------------------------------------------------------------------------------
FEATURE_NAMES: tuple[str, ...] = (
    "ret_close",  # 0
    "range",  # 1
    "body",  # 2
    "upper_wick_frac",  # 3
    "lower_wick_frac",  # 4
    "log_volume",  # 5
    "log_amount",  # 6
    "TFI",  # 7  (trade-flow imbalance — NOT orderbook OFI)
    "signed_count_imbalance",  # 8
    "trade_count",  # 9
    "mean_trade_size",  # 10
    "trade_size_dispersion",  # 11
    "large_trade_share",  # 12
    "funding_rate",  # 13
    "log_oi",  # 14
    "d_oi",  # 15
)
N_FEATURES: int = len(FEATURE_NAMES)
assert N_FEATURES == 16

FEATURE_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}

# §3.2 — features that get a causal z-score (unbounded / scale-varying)
Z_SCORED_IDX: tuple[int, ...] = (0, 1, 5, 6, 9, 10, 11, 14, 15)
# §3.1 — already-bounded / relative features: clip only, no z-score
BOUNDED_IDX: tuple[int, ...] = (2, 3, 4, 7, 8, 12, 13)
assert sorted(Z_SCORED_IDX + BOUNDED_IDX) == list(range(16))

# §3.2 defaults — slow-moving features use the long half-life; the rest the fast one
SLOW_ZSCORE_IDX: tuple[int, ...] = (5, 6, 14, 15)  # log_volume, log_amount, log_oi, d_oi
FAST_ZSCORE_IDX: tuple[int, ...] = (0, 1, 9, 10, 11)

# §1 — OHLCV-only ablation subset (Kronos-equivalent): 5 price/shape + 2 volume/liquidity
OHLCV_ONLY_IDX: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)

# §c finance-weighted features (up-weighted in the tokenizer reconstruction loss)
FINANCE_WEIGHTED_IDX: tuple[int, ...] = (
    FEATURE_INDEX["ret_close"],
    FEATURE_INDEX["range"],
    FEATURE_INDEX["TFI"],
    FEATURE_INDEX["funding_rate"],
    FEATURE_INDEX["d_oi"],
)

# §7 v1.4.1 — THE SIX MICRO DIMS (the aggTrades-derived microstructure block: TFI + trade-flow
# shape). The micro-weighted per-bar bottleneck leg and the standing M6-time legibility gate
# both target exactly this class.
MICRO_DIMS_IDX: tuple[int, ...] = (7, 8, 9, 10, 11, 12)

# §4.5 — mask-bit groups (0-indexed) referenced by the missing-data rules
MASK_BITS_LIQUIDITY: tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11, 12)  # spec "bits 6-13"
MASK_BITS_AGGTRADES: tuple[int, ...] = (7, 8, 9, 10, 11, 12)  # spec "bits 8-13"
MASK_BITS_PERP: tuple[int, ...] = (13, 14, 15)  # funding, log_oi, d_oi  (spec "bits 14-16")

# --------------------------------------------------------------------------------------
# Feature-spec §4 / Tokenizer §c — canonical FSQ config (single source of truth)
# --------------------------------------------------------------------------------------
FSQ_LEVELS: tuple[int, ...] = (11, 9, 9, 7, 7, 5, 5)  # D=7, all odd; bpt = Σ log2(L_i)
FSQ_BPT_BAND: tuple[float, float] = (19.5, 20.5)  # parity band around BSQ k=20


def bits_per_token(levels: tuple[int, ...] | list[int]) -> float:
    """bpt = Σ_i log2(L_i) — the controlled variable of the 2×2 ablation (Tokenizer §g)."""
    return sum(math.log2(level) for level in levels)


def derive_coarse_fine(levels: tuple[int, ...] | list[int]) -> tuple[list[int], list[int]]:
    """Derived coarse/fine grouping (Tokenizer §d) — NOT hardcoded.

    Sort dims by level descending, then choose the split index k that minimises
    ``|bpt_coarse - bpt_fine|`` (matching Kronos's balanced 10/10 split as closely as the
    integer levels allow). Returns ``(coarse_idx, fine_idx)`` as lists of indices into the
    *sorted-descending* dim order. For canonical ``[11,9,9,7,7,5,5]`` this yields coarse
    ``{11,9,9}`` (bpt 9.80) and fine ``{7,7,5,5}`` (bpt 10.26).
    """
    order = sorted(range(len(levels)), key=lambda i: levels[i], reverse=True)
    sorted_levels = [levels[i] for i in order]
    total = bits_per_token(levels)
    best_k, best_gap = 1, math.inf
    for k in range(1, len(sorted_levels)):
        bpt_c = bits_per_token(sorted_levels[:k])
        gap = abs(bpt_c - (total - bpt_c))
        if gap < best_gap:
            best_gap, best_k = gap, k
    coarse_idx = order[:best_k]
    fine_idx = order[best_k:]
    return coarse_idx, fine_idx


def fsq_vocab_sizes(levels: tuple[int, ...] | list[int]) -> tuple[int, int]:
    """(V_c, V_f) — products of the coarse/fine group levels. Canonical → (891, 1225)."""
    coarse_idx, fine_idx = derive_coarse_fine(levels)
    v_c, v_f = 1, 1
    for i in coarse_idx:
        v_c *= levels[i]
    for i in fine_idx:
        v_f *= levels[i]
    return v_c, v_f


# Pinned canonical derived quantities (asserted in tests against the rule above)
FSQ_V_C: int = 891
FSQ_V_F: int = 1225
