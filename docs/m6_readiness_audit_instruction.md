# M6-Readiness Audit — Builder Instruction (recon only)

> Hand this verbatim to the builder. Items 1–5 are the supervisor's original scope (cleaned);
> items **6–7** are the Drift Auditor's additions (see `drift_audit_2026-06-22.md`, Action 1).
> The whole thing is **recon only — NO building, NO training, NO cloud spend.**

---

```
M6-READINESS AUDIT — recon only. NO building, NO training, NO cloud spend.
Repo /Users/lakshaybhati/Downloads/trikaal, branch real-data-slice. The lake is M6-ready
(200 symbols, 304.6M bars, compacted to 7,024 files, anchored 5dfd667d). Before the
supervisor's M6 design, produce a readiness report so the M6 build is precisely scoped.
Build NOTHING — inventory, verify, benchmark, report a gap-list.

1) CODE INVENTORY — M6 = the 5-cell ablation {BSQ,FSQ}×{OHLCV-only,+micro} + shuffled-micro
   placebo (Cell 5), trained on the universe, scored by the M5 harness. For each piece report
   EXISTS / PARTIAL / MISSING with file refs:
   - FSQ tokenizer (encoder+FSQ+decoder) — from M1/M2;
   - BSQ tokenizer variant (the Cell-1/2 baseline at matched bits-per-token) — exists or M6-new?
   - AR backbone + MTP (Kronos_small dims) — from M1/M3;
   - OHLCV-only vs +micro feature-arm selection (which input bits each arm uses) — clean
     config switch or M6-new?
   - shuffled-microstructure placebo (Cell 5) generator — exists or M6-new?
   - MULTI-SYMBOL universe data-loader (compacted lake → batches across 200 symbols, with
     sampling-weight handling for the thin coins) — confirm (likely M6-new);
   - the 5-cell training runner (train each cell, checkpoint, W&B logging) — what's there vs new?
   - the cross-sectional eval driver wiring M5 over the 5 trained models (purged walk-forward +
     placebo comparison) — what's there vs new?

2) M5-ON-UNIVERSE VERIFY: M5 was validated on the single M3 symbol. Load 2–3 symbols + a small
   multi-symbol window through the EXISTING eval path (assemble_window / lake.connect / folds)
   and report what works vs what needs adapting for 200 symbols cross-sectionally. Do NOT fix —
   just report.

3) LAKE-READ BENCHMARK: measure raw read/scan throughput (bars/sec) from the compacted
   7,024-file layout (a DuckDB scan / parquet read of a few symbols). This feeds the GPU choice
   (4090 vs A100) + the G3 throughput gate. Flag if reading would bottleneck a GPU.

4) DETERMINISM HOOK: confirm whether the training/config code already records the per-run
   attention mode (deterministic fallback vs FlashAttention-2, invariant 7), or whether that's
   M6-new (needed before CUDA training).

5) REPORT one readiness doc: the EXISTS/PARTIAL/MISSING table, the M5-on-universe finding, the
   lake-read numbers, the determinism-hook status, and a clean GAP-LIST of exactly what M6 must
   build. Then STOP — the supervisor writes the M6 design from this. Do NOT build, train, or
   rent anything.

6) GATE-A STATUS (explicit): confirm and restate Gate A as a binding-gate RESULT, not just
   "M5 exists." Report: did the FULL M5 harness run end-to-end on the M3 checkpoint
   (folds → rollout → net-IR → DSR/PBO → placebo → diagnostics), and does
   tests/eval/test_gate_a.py pass BOTH its checks — the Q4 causal sweep leak-free at 100%
   coverage AND the planted-lookahead catch (anti-vacuity)? Yes/No with evidence. Do NOT re-run
   training; just confirm the gate is genuinely met.

7) EVAL-POWER INPUT (for the supervisor's pre-registration): attach the per-symbol coverage
   from docs/_m4b_dq_table.md (bars per symbol, regime span, thin-symbol flags). Report the
   realized (symbol × regime) cell counts so the cross-sectional effective-N / MDE / placebo
   decision threshold can be pre-registered BEFORE M6. Report only — do not compute the final
   threshold (that's the design doc's job).

STOP after the report. No code changes, no training, no cloud.
```
