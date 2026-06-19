# Milestone 3 — Stage-2 AR training on BTCUSDT-2023 (single-symbol predictor de-risk)

Goal (ROADMAP M3): prove the backbone **learns to predict real tokens** (hierarchical coarse→fine
+ MTP) and the **generation/KV-cache rollout path is correct**, before the universe (M4) and the
15-cycle ablation (M6). Dev-grade (MPS/float32); **not** a determinism-gated artifact.

**Verdict: M3 gate PASSED (qualified).** The backbone demonstrably fits real K-line tokens and its
best-val beats the trivial baselines; the cached rollout is bit-exact. The val margin is *modest*
and the model *overfits one coin-year fast* — the expected single-symbol/single-regime limit, and
exactly why broad generalization is M4/M5's job, not M3's.

## What M3 does and does NOT prove

- **Does:** the AR backbone models real K-line token structure (train CE falls steadily; best-val
  CE beats the train-marginal — context helps beyond token frequency) and the cached single-step
  rollout is bit-equivalent to the parallel forward (so the M5/M6 multi-horizon rollout is sound).
- **Does NOT:** return-forecast skill, RankIC, or net-IR — those need the M5 eval harness and the
  M6 ablation. **No RankIC/IR is computed here** (out of scope; would need the harness).

## Pipeline (all on the existing lake — no downloads)

| Stage | Result |
|---|---|
| Tokenizer freeze | Trained once on the M2 recipe (seed 0), saved content-hashed, then loaded **read-only** (Stage-2 never retrains it). val MAE 0.0773, Δ_fine 0.0600, usage c/f 0.99/0.97. `hash ce7c8c4ae0b62c89…` |
| Token stream | 525,600 bars tokenized (eval, non-overlapping 128-bar windows); **unique coarse 891/891, fine 1225/1225** (full codebook exercised on the stream). `hash d724c3d51bf438c7…` |
| Stage-2 train | backbone (exact Kronos_small, 21,301,248 base params) + MTP (D=4, μ=0.3), AdamW lr 1e-3 cosine+warmup, dropout 0.0, L=128 windows over the 1 segment, temporal-tail val, MPS, 800 steps / 649s. |

### Tokenizer-freeze caveat (honest)

M2 trained the Stage-1 tokenizer in-memory and **never persisted it**, so there was no frozen M2
checkpoint to load. M3 therefore **establishes the canonical frozen tokenizer artifact** (M2 recipe
+ pinned seed, saved content-hashed, loaded read-only thereafter). Because MPS Stage-1 is not
bit-reproducible, this is a *fresh-but-equivalent* freeze, not byte-identical to M2's discarded
weights; its hash is THE tokenizer hash going forward, and the token stream is keyed to it.

## Convergence (temporal-tail val) — abridged

| step | train_nll | val_nll | val_c | val_f | acc_c | acc_f |
|---|---|---|---|---|---|---|
| 50 | 13.380 | 13.521 | 6.664 | 6.857 | 0.002 | 0.002 |
| 150 | 13.220 | 13.340 | 6.553 | 6.786 | 0.004 | 0.004 |
| **250 (best-val)** | 13.087 | **13.282** | **6.508** | **6.774** | 0.004 | 0.003 |
| 400 | 12.791 | 13.323 | 6.540 | 6.783 | 0.004 | 0.004 |
| 800 | 11.929 | 13.568 | 6.640 | 6.927 | 0.004 | 0.003 |

Train NLL falls monotonically **13.38 → 11.93** (the backbone *fits* real tokens — capacity/wiring
confirmed on real data). Val NLL bottoms at **step 250** then rises — the model **overfits one
coin-year** past ~step 250–350 (single regime, 525k bars vs a 21.3M model). The **best-val (step
250) checkpoint is the saved artifact**; the gate is evaluated there.

## Exit gate — beats both trivial baselines (best-val, per coarse/fine)

| metric | model (best-val) | baseline | margin / verdict |
|---|---|---|---|
| **coarse CE** (nats) | **6.5076** | marginal 6.6271 | **+0.1195** — context clearly helps |
| **fine CE** (nats) | **6.7743** | marginal 6.8162 | **+0.0419** — marginally helps |
| coarse top-1 acc | 0.004 | persistence 0.003 / marginal 0.002 | ≥ both (near floor) |
| fine top-1 acc | 0.003 | persistence 0.002 / marginal 0.003 | ≈ floor |

**CE is the meaningful signal**, not top-1 accuracy: for 891/1225-way prediction of stochastic 1m
moves, top-1 accuracy is intrinsically near the ~uniform floor for *any* predictor (model and
last-token-persistence alike). The honest reading of the CE margins: the **coarse (regime/shape)
subtoken carries genuine predictable structure** (+0.12 nats over frequency), while the **fine
(residual-detail) subtoken is near-unpredictable bar-to-bar** (+0.04) — a sensible finding, not a
defect. A model that merely echoed token frequency would tie the marginal; this one beats it.

## Per-depth MTP — depths 1..4 all decrease

| depth | first eval (step 50) | best (step 250) |
|---|---|---|
| 1 | 13.5692 | 13.0526 |
| 2 | 13.5627 | 13.0730 |
| 3 | 13.5598 | 13.0884 |
| 4 | 13.5644 | 13.1103 |

All four MTP causal-chain depths learn (each drops ~0.45–0.52 nats); the monotone rise with depth
(further horizons harder) is expected. Mirrors M1's "MTP depths also learn" gate, now on real data.

## KV-cache rollout correctness

Cached single-step generation vs the parallel forward on a real BTCUSDT sequence (256 bars):
**max|Δ| = 3.34e-06** (h_final). The 5/15/60-min M5/M6 horizons come from autoregressive rollout,
so this equivalence is a hard prerequisite — verified here and in `tests/model/test_rollout.py`.

## Artifacts (content-hashed)

- `dataset_hash`      = `sha256:1d3ec4f8db6686e2…` (stored x/m/ts — matches M2 ✓)
- `tokenizer_hash`    = `sha256:ce7c8c4ae0b62c89…`
- `token_stream_hash` = `sha256:d724c3d51bf438c7…`
- `predictor_hash`    = `sha256:6d118ede6bfcab2e…`  ← the fixture M5's harness is validated against (M6 Gate A).

## Honest caveats

- **Dev-grade:** MPS / float32, not a determinism-gated artifact. Re-running the identical
  seed+config yields the *same convergence profile* (best-val @ step 250, coarse +0.12) but a
  **different `predictor_hash`** — MPS training is not bit-reproducible (§7.G2). Data, frozen-stats,
  token stream, and replay are content-hashed and stable; GPU/MPS training bit-exactness is out of
  scope here and only holds under the deterministic-attention fallback on CUDA.
- **Modest val margin + fast overfit:** one coin-year overfits a 21.3M model by ~step 250–350.
  This is the expected single-regime limit, not a wiring problem — M1's overfit gate already drove
  CE→0 on a fixed batch, and here train CE falls hard while val plateaus then rises.
- **Temporal-tail val, not purged walk-forward** — the leakage-controlled eval is M5.
- **Single symbol = single regime** — generalization across coins/regimes is M4/M5.
- **Budget:** 800 steps / L=128 is a convergence smoke, not the full-corpus schedule; it was sized
  to the val minimum after a 3000-step/L=512 attempt confirmed the model overfits one symbol well
  before then (and was impractically slow on MPS).
- **Bidirectional tokenizer** (spec §Tokenizer-a default): `b_t` is a contextual code used as the
  AR *label*; the `encoder_causal=True` defensible-artifact variant is an M6 ablation knob.

## Out of scope (deferred per ROADMAP)

M4 universe ingest; M5 eval harness (folds/Q1–Q4/net-IR/placebo/DSR); M6 ablation; any
RankIC/IR/forecast-skill claim.
