"""Paired moving-block bootstrap for ΔIR — the prereg §3 CI clause, §3a pinned recipe.

The primary M6 statistic is ΔIR_info = IR(Cell 4) − IR(Cell 5) on the pooled cross-sectional
portfolio period series (seed-mean, FULL calendar grid with flat periods as 0.0 — xsection's
convention, which keeps the two cells time-aligned regardless of per-cell activity). The CI is
**paired**: one set of moving-block indices per replicate is applied to BOTH cells' series
(equivalently, blocks of the Δr_p series are resampled), so the common market component cancels
inside every replicate — an unpaired CI would be wrongly wide (prereg §3, amendment v1.1).

Pinned recipe (prereg §3a — do not parameterize away): B = 10,000 replicates, bootstrap RNG seed
20260704, block length ⌈√T⌉, percentile CI (the one-sided lower bound is the α-quantile of the
bootstrap ΔIR distribution, α = 0.05), SE_boot = the standard deviation of the same B replicates.
MDE_paired is §3 clause 2 — a variance/nuisance quantity, never a function of the effect's sign —
and under the signed-off §7 v1.5 item-B amendment it is **(t₀.₉₅,ν + t₀.₈₀,ν)·SE_total** with
SE_total = √(SE_boot² + SE_train²) at the Welch–Satterthwaite ν whenever per-seed contrasts are
supplied; the pre-v1.5 **(z₀.₉₅ + z₀.₈₀)·SE_boot** survives only as the dual report's v1.2 leg and
as the fallback when they are not. ★ THIS PARAGRAPH USED TO STATE THE z FORM AS THE RULE, two
lines above the code that computes the t form — the same defect as the clause-2 rule string it
sits beneath, in the same commit-worth of files. Prose about a formula is never the formula:
``mde_terms`` below is the single site, and ``mde_rule_string`` is derived from it so a manifest
cannot describe a recipe it did not run. The per-replicate statistic is IR(a*) − IR(b*) with the
SHARED index draw — the exact ``metrics.information_ratio`` convention (population std, EPS_IR,
√(525600/h) annualization).

Determinism: ALL block starts are drawn in one RNG call before any chunking, so results are
bit-identical for a given (series, h, b, seed) regardless of the memory-chunk size.
Calibration-KAT'd in ``tests/eval/test_paired_bootstrap.py`` (pre-flight Item 8): null →
rejection ≈ α; planted Δ = MDE_paired → power ≈ 0.80.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trikaal.eval.dsr import norm_ppf
from trikaal.eval.metrics import EPS_IR, information_ratio, periods_per_year
from trikaal.eval.tdist import student_t_ppf, welch_satterthwaite_df

BOOT_B = 10_000  # §3a: replicates
BOOT_SEED = 20260704  # §3a: bootstrap RNG seed
BOOT_ALPHA = 0.05  # §3a: one-sided level (the CI lower bound is the α-quantile)
BOOT_POWER = 0.80  # §3 clause 2: the power point inside MDE_paired


def block_length(t: int) -> int:
    """§3a block length: ⌈√T⌉."""
    return int(np.ceil(np.sqrt(t)))


def expand_block_starts(starts: np.ndarray, length: int, t: int) -> np.ndarray:
    """``[B, n_blocks]`` block starts → ``[B, T]`` row indices: concatenate blocks, truncate to T.

    Hand-KAT'd: the standard moving-block reconstruction (overlapping blocks drawn with
    replacement; the final partial block is truncated so every replicate has exactly T rows)."""
    b, nb = starts.shape
    idx = starts[:, :, None] + np.arange(length, dtype=starts.dtype)[None, None, :]
    return idx.reshape(b, nb * length)[:, :t]


def _row_ir(rows: np.ndarray, h: int) -> np.ndarray:
    """Vectorized ``metrics.information_ratio`` over ``[B, T]`` rows (same math, same eps)."""
    mean = rows.mean(axis=1)
    std = rows.std(axis=1, ddof=0)  # population std — the metrics.py convention
    return mean / (std + EPS_IR) * np.sqrt(periods_per_year(h))


@dataclass(frozen=True)
class MdeTerms:
    """The three numbers MDE_paired is the product of, plus the names of the recipe they came from.

    ★ THIS TYPE EXISTS SO THE RULE STRING CANNOT LIE. Clause 2's persisted rule read
    ``MDE_paired = (z0.95+z0.80)·SE_boot`` — the pre-v1.5 formula — inside the string that SHIPS
    IN EVERY EMITTED VERDICT MANIFEST, while the code beside it computed t-quantiles at the
    Welch–Satterthwaite ν over SE_total. The clause-5 comment in ``verdict.py`` had already named
    the fix ("a rule string that cannot disagree with the recipe is the fix") and clause 5 was
    rebuilt from its pins; clause 2's formula half was left standing. Fixing the named instance
    and leaving its sibling is the class-rule failure, so the string is now *derived* from the
    same call the number is.
    """

    quantile_family: str  # "t" (v1.5 amended) | "z" (pre-v1.5, and the dual report's v1.2 leg)
    multiplier: float  # q_{1−α} + q_{power} in that family
    se: float  # the standard error the multiplier is applied to
    se_basis: str  # which SE that is, named
    nu: float  # df of the quantiles (B−1 for the z leg, Welch–Satterthwaite for t)
    mde: float  # multiplier · se — the operative threshold


def mde_terms(
    *,
    se_boot: float,
    se_train: float,
    n_seeds: int,
    b: int,
    alpha: float,
    power: float,
) -> MdeTerms:
    """THE SINGLE SITE where MDE_paired's recipe is decided. Both the number and its prose read it.

    ``se_train > 0`` ⇔ per-seed contrasts were supplied ⇔ the §7 v1.5 item-B amendment applies.
    That branch is the CONTRACT, not a convenience: with no per-seed input there is no training
    variance to estimate and the historical scoring-only rule is the honest fallback, which is
    also what the dual report's v1.2 leg needs.
    """
    if se_train > 0.0:
        nu = welch_satterthwaite_df(se_boot, float(b - 1), se_train, float(n_seeds - 1))
        multiplier = float(student_t_ppf(1.0 - alpha, nu) + student_t_ppf(power, nu))
        se = float(np.sqrt(se_boot * se_boot + se_train * se_train))
        family, basis = "t", "SE_total = sqrt(SE_boot^2 + SE_train^2)"
    else:
        nu = float(b - 1)
        multiplier = float(norm_ppf(1.0 - alpha) + norm_ppf(power))
        se = float(se_boot)
        family, basis = "z", "SE_boot"
    return MdeTerms(
        quantile_family=family,
        multiplier=multiplier,
        se=se,
        se_basis=basis,
        nu=nu,
        mde=float(multiplier * se),
    )


@dataclass(frozen=True)
class PairedBootstrap:
    """One paired ΔIR bootstrap: the point estimate, CI bounds, SE, and the derived MDE."""

    delta_ir: float  # IR(a) − IR(b) on the actual series (metrics.information_ratio)
    ci_lower: float  # one-sided lower bound = α-quantile of the bootstrap ΔIR replicates
    ci_upper: float  # (1−α)-quantile — the §3 clause-3 placebo-health diagnostic needs it
    se_boot: float  # std (ddof=1) of the bootstrap ΔIR replicates
    mde_paired: float  # §3 clause 2 — the OPERATIVE MDE (v1.5: includes the training term)
    # ---- §7 v1.5 item B: the two-term MDE ------------------------------------------------
    # Var(ΔÎR) = E[Var_scoring | models] + Var_models(E[ΔÎR | models])  (law of total variance).
    # Term 1 is se_boot (the bootstrap already conditions on the realized models, and seeds do NOT
    # reduce it — every seed is scored on the SAME data and grid, so scoring noise is common, not
    # independent). Term 2 is se_train, which is CLEAN: all seeds share the eval data, so the
    # across-seed spread of the per-seed contrast is purely model-induced with nothing to subtract.
    mde_paired_scoring_only: float  # the SUPERSEDED v1.2 number, retained for the dual report
    se_train: float  # σ̂_train/√S; 0.0 when per-seed deltas were not supplied
    se_total: float  # √(se_boot² + se_train²) ≥ se_boot ALWAYS
    nu_effective: float  # Welch–Satterthwaite df of se_total
    n_seeds: int  # S — the replicate count behind se_train
    t: int
    block_len: int
    b: int
    seed: int
    alpha: float
    power: float  # the power point inside MDE_paired — stored so the rule string can re-derive it
    passes_ci: bool  # clause 1: ci_lower > 0
    passes_mde: bool  # clause 2: delta_ir ≥ mde_paired

    @property
    def mde_terms(self) -> MdeTerms:
        """Re-derive clause 2's recipe from this result's OWN inputs — never from a belief."""
        return mde_terms(
            se_boot=self.se_boot,
            se_train=self.se_train,
            n_seeds=self.n_seeds,
            b=self.b,
            alpha=self.alpha,
            power=self.power,
        )


def mde_rule_string(pb: PairedBootstrap) -> str:
    """§3 clause 2's rule, WRITTEN FROM THE REALIZED RECIPE, so it cannot describe another one.

    The string carries the multiplier and the SE it multiplies, not just their names — so a reader
    of the manifest can multiply the two stated numbers and land on the stated ``mde_paired``, and
    ``tests/eval/test_mde_rule_string.py`` asserts exactly that identity. A rule string that only
    NAMED its formula is what shipped the superseded one for a month.
    """
    m = pb.mde_terms
    lo, hi = 1.0 - pb.alpha, pb.power
    if m.quantile_family == "t":
        qs = (
            f"(t_{lo:g},nu + t_{hi:g},nu) x {m.se_basis}, "
            f"nu = {m.nu:.6g} (Welch-Satterthwaite over S={pb.n_seeds} seeds)"
        )
    else:
        qs = (
            f"(z_{lo:g} + z_{hi:g}) x {m.se_basis} "
            f"(no per-seed contrasts supplied, so no training-variance term)"
        )
    # The arithmetic tail is the checkable part and its shape is pinned: ":: A x B = C". The
    # 12 significant figures are not decoration — a reader (and tests/eval/test_mde_rule_string.py)
    # multiplies the first two and must land on the third, and .6f rounding broke that identity at
    # the 1e-7 level the first time this was written.
    return f"MDE_paired = {qs} :: {m.multiplier:.12g} x {m.se:.12g} = {m.mde:.12g}"


def paired_delta_ir_bootstrap(
    r_a: np.ndarray,
    r_b: np.ndarray,
    *,
    h: int,
    b: int = BOOT_B,
    seed: int = BOOT_SEED,
    alpha: float = BOOT_ALPHA,
    power: float = BOOT_POWER,
    chunk: int = 256,
    per_seed_deltas: np.ndarray | None = None,
) -> PairedBootstrap:
    """Paired moving-block bootstrap of ΔIR = IR(r_a) − IR(r_b) with SHARED block indices.

    ``r_a``/``r_b`` are the two cells' pooled FULL-calendar-grid period series (seed-mean,
    flat periods = 0.0) on the SAME grid — the §3 pairing contract. ``chunk`` only bounds
    memory; it cannot change the result (all starts are drawn before chunking).

    ``per_seed_deltas`` (§7 v1.5 item B) is the per-seed contrast {ΔIR_s = IR(a_s) − IR(b_s)}. When
    supplied, the MDE carries the across-seed TRAINING-VARIANCE term and uses t-quantiles at the
    Welch–Satterthwaite df — which makes the primary test HARDER, by 1.24–1.60× on the multiplier
    alone plus a strictly positive variance term. When omitted the historical scoring-only
    z-quantile MDE is returned in both fields, which is what the dual report's v1.2 leg needs."""
    a = np.asarray(r_a, dtype=np.float64)
    bb = np.asarray(r_b, dtype=np.float64)
    if a.ndim != 1 or bb.ndim != 1 or a.shape != bb.shape:
        raise ValueError(f"paired series must be 1-d and same-shape; got {a.shape} vs {bb.shape}")
    t = int(a.shape[0])
    if t < 8:
        raise ValueError(f"series too short for a block bootstrap: T={t}")
    if not (np.isfinite(a).all() and np.isfinite(bb).all()):
        raise ValueError(
            "non-finite period returns — the full-calendar-grid convention fills flat "
            "periods with 0.0; NaN here is an upstream bug, not data"
        )
    length = block_length(t)
    n_blocks = int(np.ceil(t / length))
    rng = np.random.default_rng(seed)
    # ONE draw for all replicates → chunk-size-independent determinism
    starts = rng.integers(0, t - length + 1, size=(b, n_blocks), dtype=np.int64)
    deltas = np.empty(b, dtype=np.float64)
    for c0 in range(0, b, chunk):
        c1 = min(c0 + chunk, b)
        idx = expand_block_starts(starts[c0:c1], length, t)
        deltas[c0:c1] = _row_ir(a[idx], h) - _row_ir(bb[idx], h)  # SHARED idx = the pairing
    delta_ir = information_ratio(a, h) - information_ratio(bb, h)
    ci_lower = float(np.quantile(deltas, alpha, method="linear"))
    ci_upper = float(np.quantile(deltas, 1.0 - alpha, method="linear"))
    se_boot = float(np.std(deltas, ddof=1))
    mde_scoring_only = float((norm_ppf(1.0 - alpha) + norm_ppf(power)) * se_boot)

    # §7 v1.5 item B — the training term. nu_boot is B−1: the bootstrap SE is estimated from B
    # replicates, so its own df is large and the WS combination is dominated by the S−1 side.
    se_train, n_seeds = 0.0, 0
    if per_seed_deltas is not None:
        psd = np.asarray(per_seed_deltas, dtype=np.float64).ravel()
        n_seeds = int(psd.size)
        if n_seeds < 2:
            raise ValueError(
                f"per_seed_deltas needs >=2 seeds to estimate a variance, got {n_seeds}"
            )
        if not np.isfinite(psd).all():
            raise ValueError("per_seed_deltas contains non-finite values")
        se_train = float(np.std(psd, ddof=1) / np.sqrt(n_seeds))
    se_total = float(np.sqrt(se_boot * se_boot + se_train * se_train))
    # ONE SITE decides the recipe, and the rule string reads the same one (see mde_terms).
    terms = mde_terms(
        se_boot=se_boot, se_train=se_train, n_seeds=n_seeds, b=b, alpha=alpha, power=power
    )
    nu, mde_paired = terms.nu, terms.mde
    return PairedBootstrap(
        delta_ir=float(delta_ir),
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        se_boot=se_boot,
        mde_paired=mde_paired,
        mde_paired_scoring_only=mde_scoring_only,
        se_train=se_train,
        se_total=se_total,
        nu_effective=float(nu),
        n_seeds=n_seeds,
        t=t,
        block_len=length,
        b=int(b),
        seed=int(seed),
        alpha=float(alpha),
        power=float(power),
        passes_ci=bool(ci_lower > 0.0),
        passes_mde=bool(delta_ir >= mde_paired),
    )


__all__ = [
    "BOOT_ALPHA",
    "BOOT_B",
    "BOOT_POWER",
    "BOOT_SEED",
    "MdeTerms",
    "PairedBootstrap",
    "block_length",
    "expand_block_starts",
    "mde_rule_string",
    "mde_terms",
    "paired_delta_ir_bootstrap",
]
