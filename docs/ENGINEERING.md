# Trikaal — engineering guide

Project overview, the non-negotiable invariants, and the engineering norms this repository is built to. Every norm below was produced by a specific failure, which is named beside it.

## What this is

**Trikaal** — a **TOKENIZER STUDY**, not a foundation model. The object under study is the *tokenizer* for **crypto 1-minute K-lines**; the built-from-scratch decoder-only backbone (**21,301,248 realized params**, Kronos_small class) is the **measurement vehicle** — held fixed and matched across every ablation arm so the only varied factor is the quantizer × input arm. Released as a research artifact (code + weights on HuggingFace + a paper + a live demo). It is a *controlled* evolution of **Kronos** (Shi et al. 2025, arXiv:2508.02739) with exactly **one** novel research contribution: a **microstructure-aware FSQ tokenizer** for financial K-lines. Calling it a foundation model overstates the artifact and misdescribes what the ablation measures — do not reintroduce that framing.

**Current phase: implementation underway.** M1 (synthetic vertical slice) and M2 (BTCUSDT single-symbol real slice + Stage-1 tokenizer) are complete; M3 (Stage-2 AR) is next. The live build order and per-milestone exit gates live in `docs/ROADMAP.md`. The authoritative *design* remains the blueprint spec — build against it, not against assumptions or against Kronos's code.

## Sources of truth (read before doing any work)

- **Blueprint spec (design):** `docs/superpowers/specs/2026-06-18-trikaal-v1-design.md` — the full v1 design (per-bar feature spec, FSQ tokenizer, AR backbone, MTP heads, volatility-scaling, data pipeline, training plan, eval harness). If anything here conflicts with the spec, the spec wins.
- **Execution roadmap (build order):** `docs/ROADMAP.md` — the milestone sequence from here to a shipped paper, each with a testable exit gate, plus the two binding M6 entry gates. Self-sequence against it; update it when a milestone closes.
- **Parent paper:** `/Users/lakshaybhati/Downloads/2508.02739v1 copy.pdf` (Kronos). Trikaal inherits its two-stage tokenizer→autoregressive design and most of its training/eval scaffolding.

## Non-negotiable invariants

These are the things that are easy to get wrong and that would sink either the paper or the model. Do not violate them without explicit user sign-off:

1. **TFI, never OFI.** Microstructure imbalance here is computed from free Binance aggTrades = signed *executed-volume* imbalance = **trade-flow imbalance (TFI)**. True order-flow imbalance (OFI) needs orderbook depth and is explicit **v2** future work. Use the name `TFI` in all code, configs, and prose — a reviewer who sees "OFI" computed from trades alone rejects on the technicality.
2. **Causal-safety / no lookahead is a hard invariant.** Every feature, normalization statistic, volatility scale, and prediction target for bar *t* may use **only** data known by the close of bar *t*. This is enforced by a first-class CI unit test — never weaken, skip, or `xfail` it.
3. **One headline claim.** The single contribution is the microstructure-aware FSQ tokenizer. MTP heads, volatility-scaled targets, and the eval harness are *secondary engineering*, not competing research claims. Resist scope sprawl into a second claim.
4. **The 2×2 ablation is the proof:** {BSQ, FSQ} × {OHLCV-only, +microstructure} at matched params/bits, **plus a required shuffled-microstructure placebo (Cell 5)** that separates microstructure *information* from input *capacity* — if the +micro gain does not exceed the placebo gain, the microstructure leg is withdrawn. Cell 1 (BSQ + OHLCV-only) is **our** BSQ baseline at matched bits-per-token, **externally validated** against published Kronos-small (validation lives only in the eval harness; no Kronos code or weights are part of the model). Keep the AR backbone matched to the Kronos_small dims (8 layers, d_model 512, d_ff 1024, 8 heads) so the comparison stays clean.
5. **Headline metric = cost-aware net Information Ratio** under a forecast-magnitude execution filter, with a 0.1–0.3% per-trade transaction-cost model. IC / RankIC / MAE / R² are *secondary* diagnostics. Purged walk-forward + embargo is mandatory, never optional.
6. **FSQ levels are derived empirically** from reconstruction fidelity; fairness vs BSQ is controlled by **bits-per-token**, NOT by forcing a 2²⁰ vocab. Keep the coarse→fine hierarchy but monitor per-level reconstruction contribution and watch for residual-magnitude decay (mitigation: per-stage learnable scaling).
7. **Determinism is a deliverable, not hygiene (scoped honestly).** Data pipeline, frozen-stats, and prediction replay are **bit-exact** from one config file + a pinned seed + content-hashed inputs; **GPU training is bit-exact only under the deterministic-attention fallback** (FlashAttention-2 is otherwise non-deterministic) and every run records its mode. Every produced dataset is content-hashed so any past prediction can be reconstructed.
8. **Full in-house independence.** No Kronos code or weights are ever part of Trikaal — every component (attention, blocks, tokenizer, FSQ, training loop, eval, metrics) is self-written. Kronos's paper is cited prior art; its public weights appear in exactly one place: the eval harness, as an external validation target for our BSQ baseline (invariant 4). Infrastructure libraries (PyTorch, FlashAttention, DuckDB, …) remain fair game — infrastructure, not research content.

## Architecture (intended, two-stage — mirrors Kronos)

1. **Tokenizer** (`trikaal/tokenizer/`): a small Transformer autoencoder encodes the per-bar **~16-dim microstructure vector** (OHLC-shape + log-volume/amount + TFI/trade-flow + funding + open-interest) → **FSQ** quantizer (per-dimension levels, grouped into a coarse subtoken + a fine subtoken) → decoder. Trained with a hierarchical reconstruction loss (`L_coarse + L_fine`); **no commitment loss** (FSQ does not need one).
2. **AR backbone + MTP** (`trikaal/model/`): decoder-only causal Transformer (**21,301,248 params realized**; the "~27M" in older text was the *target*, never the built artifact — quote the realized number) over token sequences (≤512 bars). RoPE, RMSNorm, Pre-LN, learnable temporal embeddings (minute/hour/day-of-week/day-of-month/month). Hierarchical subtoken prediction (coarse softmax → fine via intra-block cross-attention conditioned on the *sampled* coarse subtoken). DeepSeek-V3-style **MTP** heads (a causal chain, not independent parallel heads) provide multi-horizon (1/5/15/60-min) distributional output; Monte-Carlo trajectory sampling remains available for full distributions.

Supporting subsystems, each first-class:
- **Data pipeline** (`trikaal/data/`): Binance CSV dumps (klines + aggTrades) + futures API (funding, OI) → immutable raw → **Parquet** partitioned by `(symbol, frequency, date)` → **DuckDB** query layer. Includes ingest+inference data-quality gates and dataset content-hashing.
- **Eval harness** (`trikaal/eval/`): co-equal with the model — purged walk-forward + embargo, the 2×2 ablation runner, the cost-aware net-IR backtest, and the secondary diagnostic metrics.

The **per-bar feature vector is the method foundation** — every downstream component depends on its exact definition and causal rules. Read the feature spec section first.

## Tooling & commands (the standard to wire up; repo is pre-code)

- Lint / format: `ruff check .` and `ruff format .`
- Tests: `pytest` (full suite); single test: `pytest tests/<file>::<test_name> -q`
- Pre-commit hooks + GitHub Actions CI running tests on push; the causal-safety/lookahead test is a **required CI gate**.
- **Control-arm rule (standing, prereg §7 v1.4.7):** a probe or gate **must refuse to emit a verdict when its own control arm fails**. If the treatment and the control fail alike, the correct output is `PROBE INVALID` — never a conclusion about the treatment. A probe must also distinguish the failure signature it is testing for from an unrelated fault. This is the same class of defect as an exit code masked by a pipe (see the pipefail rider): a harness that hides its own failure and lets an unverified claim through.
- **Fixture-discrimination rule (standing, prereg §7 v1.5):** a **mutation or negative check must first prove that its own fixture can discriminate the cases it claims to separate**. An assertion that would pass just as well against a gate that does nothing is not a check. Concretely: the v1.5 A.4 mutation check ("the conformance gate rejects the superseded all-arms `var_sr` basis") could not fire, because the conformance fixture's trial values were a modular function that made the placebo-only and all-arms dispersions **numerically identical** — the assertion passed vacuously. It was caught only because the mutation test **failed to fail**. The fix is the norm: the fixture is now cell-dependent and the test asserts `_var_all_arms(full) != _var0(full)` **before** relying on it (`tests/eval/test_conformance.py`).
- **The three are one defect in three costumes — "a check that cannot fail is not a check":** the **pipefail rider** (a `| grep | tail` masked an exit-1 and let a false anchor claim into a commit message; stated in `docs/m6_prereg.md` §7 v1.4.1, cross-referenced above), the **control-arm rule** (a probe emitted a verdict while both arms died of the same `TypeError`), and the **fixture-discrimination rule** (a mutation assertion that no gate could have failed). Each hides a harness's own failure and lets an unverified claim through. Treat any new gate, probe, or negative test as suspect until it has been shown capable of failing.
- Experiment tracking: Weights & Biases.
- **Allowed to import** (infrastructure): PyTorch, CUDA/cuDNN/NCCL, FlashAttention-2, PyArrow/DuckDB/Polars, W&B.
- **Write from scratch** (the research surface — do not import others' implementations): attention/blocks/model, the tokenizer (encoder + FSQ + decoder), the training loop (DDP, checkpointing, schedules), and all eval metrics.

Target training hardware: a single cloud A100/H100 (40–80 GB), bf16, single-process but DDP-ready.

## Out of scope for v1 (firewalled — do not add)

Equities / cross-asset data, Bybit/OKX sources, orderbook depth (→ true OFI), base-class (~100M) scale-up, MLA attention, and architecture-level latency micro-optimization (a serving concern, handled at export/serving). These are captured as v2/v3 in the spec's roadmap; keep them out of v1.
