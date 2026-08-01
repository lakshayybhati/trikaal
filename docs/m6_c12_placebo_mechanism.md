# C-12 — What the Cell-5 placebo actually destroys

**AUDIT TIER-2. WRITE-UP ONLY.** The design is FROZEN and nothing here touches it. This states the
mechanism, measures it, separates the two effects it conflates, says what would measure them
apart, and lists the options. It proposes no change and adjudicates nothing. The supervisor has
already flagged that this may be a design-level problem with the headline comparison and may be
**Lakshay's** call rather than theirs.

---

## 1. The mechanism, from the code

`shuffle_micro` (`train/arms.py:51`) → `block_time_permute` (`eval/placebo.py:56`). Within each
segment it draws **one** permutation and applies it to the six micro columns as a block:

```python
perm = rng.permutation(s1 - s0)
out_x[s0:s1, cols] = x[s0:s1, :][perm][:, cols]      # cols = MICRO_DIMS = (7,8,9,10,11,12)
out_m[s0:s1, cols] = mask[s0:s1, :][perm][:, cols]
```

Three properties follow directly and none of them is incidental:

1. the micro block moves **as a unit**, so within-bar structure *among* the micro dims survives;
2. the OHLCV columns are **not in `cols`**, so they are untouched;
3. bar *t*'s OHLCV therefore ends up beside bar *π(t)*'s micro.

## 2. Measured (BTCUSDT, 300k bars, seed 0)

| quantity | before | after | verdict |
|---|---|---|---|
| mean \|corr\| micro ↔ OHLCV, contemporaneous | **0.2037** | **0.0005** | **DESTROYED** |
| mean \|lag-1 autocorr\|, micro | 0.4165 | 0.0006 | DESTROYED *(intended)* |
| mean \|corr\| micro ↔ micro, within-bar | 0.2886 | 0.2886 | preserved |
| mean \|corr\| OHLCV ↔ OHLCV, within-bar | 0.3105 | 0.3105 | preserved |
| micro marginal distribution | — | exact | preserved |
| OHLCV columns | — | byte-identical | untouched |

**The audit's claim holds.** The placebo destroys **two** things where the design describes one.
`shuffle_micro`'s own docstring says *"Marginal distribution preserved, temporal information
destroyed"* — true, and incomplete: the contemporaneous micro↔OHLCV dependence, which is a real
0.20 correlation in the data, goes to zero as well.

## 3. Why that matters, and why λ = 3.0 compounds it

After the shuffle, Cell 5's micro channels are not merely *uninformative about the future* — they
are **statistically independent of everything else in the same bar**. Cell 5 is therefore not
"Cell 4 minus the microstructure information". It is "Cell 4 with six dimensions of independent
noise bolted on".

That is a different kind of arm, and it costs in two ways a pure information-removal would not:

- **Tokenizer capacity.** The FSQ bottleneck has a fixed bits-per-token budget (invariant 6: the
  fairness control *is* bits-per-token). Six dims of independent noise are incompressible and
  cannot be predicted from anything else in the bar, so any capacity spent representing them is
  capacity taken from OHLCV. Cell 4's micro dims, being correlated with OHLCV at 0.20 and
  autocorrelated at 0.42, are partly *free* — they share structure the code is already carrying.
- **λ = 3.0 (`PINNED_MICRO_POINT_WEIGHT`) multiplies the cost.** The per-bar bottleneck leg
  weights the six micro dims **3× in the loss**. In Cell 4 that weighting buys per-bar legibility
  of real state; in Cell 5 it directs three times the gradient pressure at fitting noise. λ was
  calibrated on Cell 4's legibility gate (§7 v1.4.1), not on what it does to a noise arm.

There is also a standing finding that predicts exactly this direction: reconstruction-trained
tokenizers **allocate code capacity by variance and covariance, never by downstream value**.
Independent noise with a preserved marginal has full variance and zero covariance — the worst
possible case for that allocator, and it is weighted 3×.

## 4. What this does to the headline

ΔIR(4−5) is the pre-registered headline. As specified it measures

> (information in temporally-aligned, OHLCV-coupled microstructure) **+** (the capacity and
> λ-weighting handicap of carrying six noise dims)

and the two are not separable from the contrast alone. A positive ΔIR(4−5) is therefore consistent
with the claim being true, **and** consistent with a smaller true information effect plus a
handicap. The direction of the confound is adverse to the null, i.e. it **inflates** the apparent
effect. It does not create one from nothing — Cell 4 must still beat Cell 5 by more than the
pre-registered MDE and clear the 0.5 economic floor — but the size is not clean.

**Clause 3 does not close this, and the audit is right about why.** Clause 3 requires
ΔIR(4−2) > 0, with Cell 2 = FSQ + OHLCV-only. Cell 2 is *information-matched* to the question
(no microstructure) but **not capacity-matched**: it carries 7 dims, not 16, so it never pays the
noise-encoding cost either. Cell 4 vs Cell 2 compares 16-with-information against 7; Cell 4 vs
Cell 5 compares 16-with-information against 16-with-noise. **Neither contrast holds capacity
fixed while varying information.**

## 5. What would MEASURE the handicap separately

Listed as measurements, with their cost and whether they touch the frozen design. **None is
recommended here.**

**(M1) OHLCV reconstruction quality, Cell 4 vs Cell 5 — $0, no design change, uses artifacts the
run already produces.** If capacity is being diverted to noise, Cell 5's tokenizer should
reconstruct the *OHLCV* dims measurably worse than Cell 4's, on the same bars. That is a direct
read of the handicap and it needs no new cell, no new arm, and no re-run — only the per-cell
reconstruction the run already computes, sliced to the OHLCV dims. **This is the cheapest thing
on the list and it is diagnostic, not gating.**

**(M2) Effective-bits-per-token, already recorded.** The artifacts carry
`codebook.effective_bits_per_token`. Cell 5 spending *comparable* bits while reconstructing OHLCV
*worse* is the signature of capacity diverted rather than capacity unused. Combines with M1; also
$0 and already collected.

**(M3) A λ=1.0 Cell-5 variant.** Re-run Cell 5 with the micro weight at 1.0. A material rise in
IR(5) isolates the λ component of the handicap from the noise-encoding component. **Costs one
extra cell-worth of training, and λ is a pinned value** — legitimate only as a diagnostic run
outside the verdict path, which is a ruling, not a builder decision.

**(M4) A capacity-matched sixth cell** — 16 input dims where the six micro slots carry a constant
(zero variance, zero covariance), isolating "carrying six dims" from "carrying six *noisy* dims".
This is the cleanest separation and it is **a design change**: a sixth cell, new compute, and a
new arm. Out of bounds under the freeze; recorded for completeness only.

**(M5) An alternative surrogate.** `eval/placebo.py` already implements `phase_randomize`, which
preserves the power spectrum per channel. It is explicitly marked *"Diagnostic alternative only —
NOT the Cell-5 surrogate"*, and its docstring notes it synthesizes new values so **no mask can
travel with them** — promoting it would re-open the mask-handling question the 2026-07-06 audit
fix closed. Note also that phase randomization would destroy contemporaneous cross-channel
dependence too, so it is **not** a fix for this particular confound. Recorded so nobody reaches
for it assuming it is.

## 6. Options, given the design is FROZEN

**(O1) Run as specified and disclose.** ΔIR(4−5) is reported as the headline exactly as
pre-registered, and the paper states plainly that the placebo removes contemporaneous coupling as
well as temporal alignment, so the contrast bounds information *plus* handicap. Costs nothing,
changes nothing, and is honest. The disclosure belongs in the abstract's limitations, not a
footnote, because it bears on the size of the headline number.

**(O2) O1 plus the free diagnostics (M1 + M2).** Same run, same verdict path, with the OHLCV-recon
and effective-bits comparison reported alongside as a non-gating measurement of the handicap's
size. If the handicap is small, the disclosure in O1 is weakened in the good direction and the
headline is stronger; if it is large, the paper says so. **This is the option that adds the most
information for the least cost and no design risk**, which is why it is stated first among the
non-trivial ones — but it is still not a recommendation.

**(O3) Add M3 or M4 as a post-hoc diagnostic run.** Buys a real separation, costs GPU time and
touches pinned values or adds a cell. A ruling, and probably Lakshay's.

**(O4) Re-spec the placebo.** Forbidden under the freeze. Recorded only so the option set is
complete, and noting that any re-spec re-opens the mask-travel question and would invalidate the
existing Cell-5 conformance receipts.

## 7. What is NOT claimed here

- That ΔIR(4−5) is wrong. It measures what it measures; the issue is what the number *contains*.
- That the effect direction flips. The confound inflates rather than manufactures.
- That the placebo is badly implemented. It does exactly what `block_time_permute` says; the gap
  is between that and how the *design* describes its effect.
- Any adjudication. The measurement is in §2, the reasoning in §3–4, and the decision is not the
  builder's.
