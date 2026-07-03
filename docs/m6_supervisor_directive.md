# Supervisor Directive — accompanies the Drift Audit

**Attach with:** `drift_audit_2026-06-22.md` (full findings) and `m6_readiness_audit_instruction.md` (builder recon prompt).
**From:** Drift Auditor · **Date:** 2026-06-22 · **Position:** M4→M6 boundary.

---

**Audit verdict: NO DRIFT FIRING.** The build has a clean bill — tests green, FSQ canonical-correct, ablation parity and causal-safety load-bearing in source, Gate A codified with teeth. Four cheap closures are required before any GPU-second. Do the following, in order:

1. **Send the builder the readiness audit** (`m6_readiness_audit_instruction.md`). It already includes the two auditor additions — Gate-A status restatement and the eval-power coverage input. Recon only, no spend. No GPU until its gap-list is back.

2. **The M6 design doc MUST explicitly carry** (these are the binding closures):
   - **(OW-1)** a pre-registered **cross-sectional** effective-N / MDE / placebo **decision threshold** — written before any M6 number is seen. The realized lake is 304.6M bars / thin per-(symbol×regime); the placebo margin must clear a power-aware bar. Input = `_m4b_dq_table.md`.
   - **(OW-2)** the **§8.C.3 Kronos-parity setup gate** — Cell-1 (BSQ) must reach within ~10–15% RankIC of published Kronos-small on a common slice *before* any "FSQ beats BSQ" claim. Currently not named anywhere in the audit chain — do not let it drop.
   - **(OW-4)** per-run **determinism-mode recording** (deterministic-attention vs FlashAttention-2) before CUDA training.
   - seeds ≥ 3 per cell; **bits-per-token parity** across all five cells; placebo (Cell 5) retained.

3. **Drop the "Scale-up training" label for M7.** M6 *is* the training; M7 = honesty pass + write-up; M8 = release. "Scale-up" is a firewalled word (base-class ~100M is v2/v3) — do not carry it forward in any summary.

4. **Do NOT re-open settled decisions.** The 304.6M bar count is documented as below-band-but-adequate (eval breadth, not training fuel; 27M saturates ~1–2B). The external report's "nobody named the bar drop" framing is **factually wrong** — it was flagged in three places (`milestone4b_universe_ingest.md:47`, exit-gate row `:187`, and the DQ table). Take that report's two cheap items (now folded into the audit instruction); ignore its surfacing-failure framing. Likewise off-limits: the FSQ formulation, the Kronos_small backbone dims, the single-claim scope.

**Gate to hold:** no GPU-second until (a) the readiness gap-list is in, (b) the M6 design carries OW-1 / OW-2 / OW-4, and (c) the repo is pushed (`git push -u origin real-data-slice`, verified **private**) and `GH_TOKEN` is held for Phase B.
