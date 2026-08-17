# M6 MONEY RUN — STATUS REPORT FOR THE SUPERVISOR
**2026-08-12 11:55Z · builder · autonomous run, ~19 h since pool launch**

## THE ONE-LINE ANSWER
Nothing has failed. **Zero of 25 units have completed**, unit 0's eval is **58% past its predicted
duration with no way to observe progress**, and at the current trajectory **the 25-unit matrix no
longer fits the budget even with the promised $20 top-up.** That last point is a decision, and it
is not mine.

## WHAT IS HEALTHY — and it is most things
- **Launch SHA `055d14f`**, rev-list 0, tree clean, HEAD frozen. No commits since.
- Pre-launch: `ruff check` **EXIT 0** · `ruff format --check` **EXIT 0** · suite **753 passed EXIT 0**.
- **All 16 identity keys AGREE across all three boxes.** Verified twice for r2, once from a fresh
  mid-run stamp that came back byte-identical to its provisioning stamp after ~8 h of load.
- **§7 v1.6.29 is working in production**: r2's kernel is `6.8.0-101`, r0/r1 are `5.15.0-52`.
  Recorded on every artifact, never compared; `platform_abi` agrees. Under the old key r2's five
  artifacts would have refused the whole verdict.
- Lake 1920/1920 sha256-verified and token-scrub-proven on all three boxes.
- Training completed cleanly on all three, consistent across boxes:
  | box | unit | s1 | s2 | train wall |
  |---|---|---|---|---|
  | r0 | cell1_bsq_ohlcv_seed0 | 0.059 | 13.089 | 10,254.0 s |
  | r1 | cell1_bsq_ohlcv_seed2 | 0.058 | 12.776 | 11,065.3 s |
  | r2 | cell1_bsq_ohlcv_seed4 | 0.052 | 13.377 | 10,580.7 s |
- 3 boxes live, GPUs 53–98%, 22.8 GiB held, CPU time climbing. Nothing is hung.

## THE PROBLEM: THE EVAL LEG
Predicted **7.80 h/unit** (1,402,560 decisions × 0.020009 s, the marginal you and I both verified
against the throughput probe). **Observed: 12.3 h and still running** — ~31 ms/decision, **58% over**.

**And it cannot be observed.** Between `[ckpt] reloaded OK` and the final `[eval]` line the job
emits nothing; it does zero disk I/O (arrays resident); `py-spy` cannot attach (container lacks
`SYS_PTRACE`). The only signal is "still running". I set a 12 h investigation bound, it fired, and
there was no instrument to act on it with. **A 12-hour GPU loop that emits nothing cannot be
distinguished from a hung one.**

## THE ECONOMICS
Spent **$35.02**. Forward cost for the 22 remaining units = **U × $10.881**.

| eval ends at | unit U | forward | vs $146 (no top-up) | vs $166 (with +$20) |
|---|---|---|---|---|
| 12.5 h | 15.35 h | $167.0 | no | **at the limit** |
| 13.0 h | 15.85 h | $172.5 | no | **short ~$6** |
| 15.0 h | 17.85 h | $194.2 | no | short ~$28 |

Every further hour of eval costs **~$10.9** of forward budget. We are at the edge now.

Reshuffling boxes does not rescue it: retiring r2 (dearest box, $0.6278/h, fewest units) and moving
seed 4 to r0/r1 saves **~$8–10** and adds **~1.1 days**. Total cost is `Σ(units × U × rate)` — the
units do not disappear. **U is the binding constraint, not the allocation.**

**Runway**: 93.7 h on $181; 106.5 h with the top-up. The full run needs ~5.2+ days. **The balance
runs dry mid-run without the top-up, and training has no resume — an interrupted unit is lost.**

## WHAT I HAVE NOT DONE
- Not stopped: that would discard ~12 h × 3 boxes of paid compute, and the unit may close any moment.
- Not started units 4–25: the unit-0 gate has not passed.
- Not committed anything. Not weakened any gate.

## THE DECISION I NEED
If unit 0 closes soon (≤12.5 h eval) the run fits with the top-up and I continue. **If it runs
materially longer, 25 units cannot be funded**, and the options are:
1. **Fund past $160** — needs more than $20.
2. **Stop now**, banking 3 of 25 units plus the real per-unit cost measurement.
3. **Re-scope** the matrix — a prereg change, not a builder decision.

## FINDINGS FOR THE POST-VERDICT COMMIT (11)
1. HF 429 on concurrent lake pulls; retry+resume fixes it (cost me 4 boxes, $0.40).
2. A stagger shorter than the operation it separates separates nothing.
3. Cap must be checked against **billed**, not search prices (~5.8% host-stable uplift).
4. `host_id` ≠ `machine_id`; never infer a machine from a host or IP.
5. `offer_id` is unrecoverable from `vastai show instances` — re-rent depends on the create log.
6. Foreground SSH kills long remote work (my defect, $3.35).
7. Three zsh traps: `${pipestatus[1]}` spelling; unquoted params don't word-split; **a failed glob
   aborts the whole command** — that one silently made an `rm` never run and produced a false
   "resolved" claim from me.
8. `inet_down>1000` did not predict a 65-minute install.
9. `nohup … &` over ssh does not release the connection (held 3h26m).
10. **`m6_eval_throughput_probe.json`'s headline field is a two-point fit through its noisiest
    measurement** (512-point std 59% of mean), ignoring the clean 2048 point, missing it by 35%,
    **2.08× optimistic** — in the field that sizes the largest cost leg in the project.
11. My runner has no retry on the endpoint lookup: one transient API blip made r2's runner declare
    `BOX_GONE` while the box was `running`. Detached design saved the work.

**12 (new, and the most consequential): the eval leg has no progress instrumentation.** Everything
above was found because something emitted a number. This one cannot be, and it is the leg that
consumes ~73% of the run's compute.
