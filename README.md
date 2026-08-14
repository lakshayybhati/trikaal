# Trikaal

A **tokenizer study**: does microstructure-aware **FSQ** quantization of crypto 1-minute K-lines
buy anything a price-shape tokenizer does not? The object under study is the *tokenizer*; the
21.3M-parameter decoder-only backbone is the **measurement vehicle**, held fixed across every
arm, not a product. A controlled evolution of Kronos. The authoritative design is the blueprint
spec at
[`docs/superpowers/specs/2026-06-18-trikaal-v1-design.md`](docs/superpowers/specs/2026-06-18-trikaal-v1-design.md);
the live build order + milestone exit gates are in [`docs/ROADMAP.md`](docs/ROADMAP.md).

**Status:** M1 (synthetic slice), M2 (BTCUSDT real slice + Stage-1 tokenizer), M3 (Stage-2 AR),
M5 (eval harness), and M4 (the full 200-symbol / 304.6 M-bar universe lake, Merkle `5dfd667d…`)
are **complete**; **M6** (the 5-cell {BSQ,FSQ}×{OHLCV,+micro}+placebo ablation) is next — see
[`docs/m6_design.md`](docs/m6_design.md). The environment is **pinned** (committed `uv.lock`;
`uv sync --locked --extra data --extra dev`).

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
5. **The AR backbone + MTP** (`trikaal.model`) — exact Kronos_small (8L/512/1024/8h, **31,795,200** realized total; **21,301,248** backbone excluding MTP
   realized params) with hierarchical coarse→fine heads and DeepSeek-style MTP causal-chain depths.

## Pre-flight gates (§7.0)

| Gate | What it asserts | Where |
|---|---|---|
| **G0** | **Exhaustive** truncation sweep — *every* bar is a truncation boundary (100% coverage is the gating criterion, not a sparse sample), so a single-bar localized leak cannot pass; plus an adversarial perturbation probe on the high-risk strata; planted leaks (global + localized divisor + localized survivorship) all fail; loader excludes QA-only columns; FSQ/backbone/MTP shape & causality units | `tests/data/test_causal_safety.py`, `tests/{tokenizer,model}/*` |
| **G1 stage-1** | Tokenizer overfits one batch to recon **MAE < 1e-3** | `tests/tokenizer/test_roundtrip.py` |
| **G1 stage-2** | Backbone+MTP overfit one batch to coarse+fine **CE < 0.05 nats/token** | `tests/model/test_overfit.py` |
| **G2** | Two same-seed runs produce **bit-identical** loss curves | `tests/test_determinism.py` |

## Running

```bash
pip install -e ".[dev]"

ruff check . && ruff format --check .     # lint + format
pytest -q -m "not slow"                   # fast suite (G0 units, G1 stage-2, G2)
pytest -q -m slow                         # the longer G1 stage-1 overfit gate
python scripts/preflight.py               # G0/G1/G2 summary (the launcher's pre-flight)
```

## Deferred to later milestones (explicitly out of this slice)

- **Binance ingest (§2.1)** + the Parquet/DuckDB lake + content-hashed manifest — the whole
  point of this slice is to be green *before* downloading anything.
- **Inference / generation subsystem (§8)** — the end-to-end KV-cache generation path
  (`TrikaalAR.step`, `BackboneOutput.kv_cache`) and Monte-Carlo rollout. The KV-cache *primitive*
  (`MultiHeadSelfAttention.step`) is implemented and verified bit-equivalent to the full forward
  (`tests/model/test_attention.py`); wiring it through block→model belongs with the eval harness.
- **Eval harness (§8)** — purged walk-forward + embargo, the 2×2 ablation runner, the cost-aware
  net-IR backtest. The training gates here use the parallel (non-cached) forward.
- **Full-corpus Stage-1/Stage-2 training** — this slice overfits *single batches* to prove the
  architecture is bug-free; convergence on the real corpus is a separate run.

This scope was confirmed by a 4-dimension adversarial review (causal-safety / FSQ / backbone-MTP
/ spec-invariants); its high-severity findings on the harness (sparse sampling → exhaustive
coverage; unchecked `target_valid`/`ts`) and on §6.3 sampled-coarse conditioning were fixed here.

## Non-negotiable invariants (from `CLAUDE.md`)

- **TFI, never OFI** — imbalance is signed executed-volume from aggTrades, not orderbook OFI.
- **Strict causal-safety** — every output for bar `t` is a pure function of raw data with
  effective timestamp `≤ t+1`; the divisor `σ_t` reads bars `≤ t`, the *label* may read `t+1`.
- **One headline claim** — the FSQ tokenizer. MTP, vol-scaling, and the eval harness are secondary.
- **Determinism is a deliverable** — one config + a pinned seed reproduces any run.
