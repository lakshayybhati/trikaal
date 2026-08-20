# Trikaal

A **tokenizer study**: does microstructure-aware **FSQ** quantization of crypto 1-minute K-lines
buy anything a price-shape tokenizer does not? The object under study is the *tokenizer*; the
decoder-only backbone — **31,795,200 realized parameters** at the FSQ vocabulary, 31,725,568 at
the BSQ one, of which 10,493,952 are the MTP heads — is the **measurement vehicle**, held fixed
across every arm, not a product. A controlled evolution of Kronos; **no Kronos code or weights are
part of it**. The authoritative design is the blueprint spec at
[`docs/specs/2026-06-18-trikaal-v1-design.md`](docs/specs/2026-06-18-trikaal-v1-design.md);
the live build order + milestone exit gates are in [`docs/ROADMAP.md`](docs/ROADMAP.md).

## ★ Status: the designed experiment did not run, and that is the result

**A pre-registered micro-legibility gate fired on real data on 2026-08-12.** It was written on
2026-07-30 — thirteen days earlier, before any of the data it refused had been seen — and §7 v1.5
item E pre-committed the response: gate fires → **stop**, and the primary result becomes the
**mechanism finding**. So the 5-cell {BSQ,FSQ}×{OHLCV,+micro}+placebo ablation **does not run as
designed**. **Stage 2 was never entered for cells 2–5** and none of them produced a scored
artifact; cells 4 and 5 did run through **Stage 1**, and that run is what produced the legibility
receipt the finding rests on. Cell 1 (BSQ + OHLCV-only, seeds 0/2/4) is the only arm with scored
artifacts.

**The finding the gate produced:** the tokenizer keeps the microstructure that *duplicates OHLCV*
and **drops the signed channels** — because an MSE reconstruction objective allocates capacity by
variance **and covariance**, making an independent low-variance channel the worst per-bit
investment available. The gate named the channels: **97.3% of *cell 4's* shortfall** — one seed,
one run — sits on `TFI` and `signed_count_imbalance`, the two **signed** channels, while the four
**magnitude** channels — the ones that co-vary with **volume**, not with price — essentially clear
it. The per-dim arithmetic is printed in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md), and the six
sign accuracies it recomputes from are in `runs_manifest/m6_micro_legibility_stop.json`.

The 18 stored calibration replicates — a **different** file, `runs_manifest/m6_lambda_sweep.json`
— put the same signed share between 77% and 93%, so the direction is robust and the exact
percentage is not. **That file declares itself `NOT_COMPARABLE_TO_THE_GATE_FIRING_RUN`, and no
delta between the 97.3% and the 77–93% spread is a measured quantity.** The spread is quoted as
the dispersion of the statistic across replicates and for nothing else. Measured three ways that share no input (a synthetic fixture, a planted-information canary,
and 40 real symbols at n=150k/dim), and a 512-bar windowed read recovers nothing beyond the per-bar
read — so it is eviction, not smearing.

Milestones M1–M5 (synthetic slice → BTCUSDT real slice → Stage-2 AR → eval harness → the full
200-symbol / 304,625,181-bar universe lake, Merkle `5dfd667d…`) are complete. The environment is
**pinned** (committed `uv.lock`; `uv sync --locked --extra data --extra dev`).

**What the released units were actually trained on — it is not the whole lake.** The draw is **40
of the 200 symbols** and **84,153,600 bars** — 27.6% of the lake. Training consumed 832,096
windows × 512 = 426,033,152 bar-positions at the pinned 26,003 steps × batch 32, i.e. **5.06
passes** over the draw. `train_frac = 0.7` over the 2021-01-01 → 2025-01-01
window puts the split at **2023-10-20**: everything after that date, **including all of 2024, is
held out**. Read the draw back from `draw.drawn_by_symbol_stage1` in any unit's `run_manifest.json`.

## Where to start

| You want | Read |
|---|---|
| **the trained weights themselves** | <https://huggingface.co/lakshayybhati/trikaal-v1-baseline> |
| what the weights are, and what they are **not** | [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) |
| what ships, and its content hashes | [`runs_manifest/m6_weights_release.json`](runs_manifest/m6_weights_release.json) |
| the build order and its exit gates | [`docs/ROADMAP.md`](docs/ROADMAP.md) |
| the M6 design and pre-registration | [`docs/m6_design.md`](docs/m6_design.md), [`docs/m6_prereg.md`](docs/m6_prereg.md) |
| the local forecast dashboard | [`docs/v1_csv_dashboard.md`](docs/v1_csv_dashboard.md) |
| the engineering invariants and working norms | [`docs/ENGINEERING.md`](docs/ENGINEERING.md) |
| what v1 does **not** do, and what v2 would have to fix | [`docs/v2_and_limitations.md`](docs/v2_and_limitations.md) |

**The economics are negative and are reported as such.** Break-even is the mean gross return per
active period: 0.00232% / 0.00538% / 0.02082% against a realistic ~0.10% round trip — 4.8× to 43×
short. The net information ratios (−28 / −67 / −146) are reproduced to within 8–17% by a
*zero-skill* model paying the same costs, i.e. they measure cost drag rather than negative skill.
Nothing here is a trading system.

The sections below describe the original milestone-1 synthetic slice, whose gates and structure
still underpin the repo.

## What the M1 slice proves

The goal of this slice is to prove the **entire model + harness architecture trains and is
leak-free on synthetic data**, with the pre-flight gates **G0 / G1 / G2** green, *before* writing
any Binance ingest (§2.1) or downloading a single byte. The real-data pipeline is deliberately
deferred until this slice is green.

It builds, in order:

1. **A seeded synthetic raw stream** (`trikaal.data.synthetic`) with deliberately injected
   **gaps, bad ticks, structural breaks, and stale runs** — the four high-risk regions the
   lookahead sampler must target.
2. **The real per-bar feature transforms** (`trikaal.data.{segments,normalize,quality,volatility,features,targets}`)
   — feature-spec §1/§3/§4/§5 + vol-scaling §3 — fed by the synthetic stream instead of Binance.
3. **The causal-safety harness** (`trikaal.data.causal_check`) — the transform-agnostic
   truncation + perturbation invariant (§1.6 / §2.6.2) with high-risk-region stratified sampling.
   Four planted-lookahead variants prove the harness has teeth; the real transforms pass.
4. **The FSQ tokenizer** (`trikaal.tokenizer`) — encoder → FSQ (`[11,9,9,7,7,5,5]`, coarse
   `{11,9,9}`→891 / fine `{7,7,5,5}`→1225) → decoder, hierarchical Huber loss, **no commitment term**.
5. **The AR backbone + MTP** (`trikaal.model`) — the Kronos_small *dimensions* (8 layers, `d_model`
   512, `d_ff` 1024, 8 heads), written from scratch: **31,795,200** realized parameters at the FSQ
   vocabulary (**21,301,248** excluding the MTP heads), with hierarchical coarse→fine heads and
   DeepSeek-style MTP causal-chain depths.

## Pre-flight gates (§7.0)

| Gate | What it asserts | Where |
|---|---|---|
| **G0** | **Exhaustive** truncation sweep — *every* bar is a truncation boundary (100% coverage is the gating criterion, not a sparse sample), so a single-bar localized leak cannot pass; plus an adversarial perturbation probe on the high-risk strata; planted leaks (global + localized divisor + localized survivorship) all fail; loader excludes QA-only columns; FSQ/backbone/MTP shape & causality units | `tests/data/test_causal_safety.py`, `tests/{tokenizer,model}/*` |
| **G1 stage-1** | Tokenizer overfits one batch to recon **MAE < 1e-3** | `tests/tokenizer/test_roundtrip.py` |
| **G1 stage-2** | Backbone+MTP overfit one batch to coarse+fine **CE < 0.05 nats/token** | `tests/model/test_overfit.py` |
| **G2** | Two same-seed runs produce **bit-identical** loss curves | `tests/test_determinism.py` |

## Running

The pinned environment, from the committed lockfile — the same command CI runs
(`.github/workflows/ci.yml`) and the same one `scripts/setup_cloud.sh` uses on a rented box:

```bash
uv sync --locked --extra data --extra dev
```

```bash
uv run ruff check . && uv run ruff format --check .   # lint + format
uv run pytest -q -m "not slow"                        # fast suite (G0 units, G1 stage-2, G2)
uv run pytest -q -m slow                              # the longer G1 stage-1 overfit gate
uv run python scripts/preflight.py                    # G0/G1/G2 summary
```

**Both extras are required, and this used to say `pip install -e ".[dev]"`.** That omits
`data`, so `polars` and `duckdb` are absent, four test modules fail to import, and pytest
exits **2 during collection with zero tests run** — the documented setup verified nothing.
`uv sync --locked` additionally installs the *locked* tool versions: `ruff` is now pinned
exactly, because a floor-only constraint resolved to a newer minor that formats Python inside
markdown fences and failed lint on files the locked version passes.

## What was deferred at M1, and what happened to it

This section used to list the Binance ingest and lake, the eval harness, the generation
path and full-corpus training as **not built**. **All four shipped.** The lake is the
200-symbol / 304,625,181-bar universe named above; the eval harness is the seventeen modules
under `src/trikaal/eval/`; generation is `TrikaalAR.init_caches()` / `TrikaalAR.step()`; and
full-corpus training produced the three published cell-1 units. The old block also cited
`BackboneOutput.kv_cache`, which never existed under that name.

It is replaced rather than silently deleted because an agent that reads a present-tense
"not built" and believes it writes a second copy of a subsystem that already exists — which
is the same failure the package docstrings under `src/` carried until 2026-08-20.

## Non-negotiable invariants (from `docs/ENGINEERING.md`)

- **TFI, never OFI** — imbalance is signed executed-volume from aggTrades, not orderbook OFI.
- **Strict causal-safety** — every output for bar `t` is a pure function of raw data with
  effective timestamp `≤ t+1`; the divisor `σ_t` reads bars `≤ t`, the *label* may read `t+1`.
- **One headline claim** — the FSQ tokenizer. MTP, vol-scaling, and the eval harness are secondary.
- **Determinism is a deliverable, scoped honestly** — the data pipeline, frozen statistics and
  prediction replay are bit-exact from one config file, a pinned seed and content-hashed inputs.
  **GPU training is bit-exact only under the deterministic-attention fallback**, and deterministic
  attention is necessary but *not* sufficient; every run records its mode.
- **Full in-house independence** — no Kronos code or weights are part of Trikaal, and none are
  ever pulled. Kronos's paper is cited prior art.

## Licence

Code and weights: **Apache-2.0** ([`LICENSE`](LICENSE)). Binance source data is **not**
redistributed — the ingest pipeline and its content hashes are what reproduce the lake.
