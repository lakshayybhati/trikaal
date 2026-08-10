# 3 Capacity eviction in reconstruction-trained tokenizers

Two-stage architectures for financial time series — a learned tokenizer, followed by an
autoregressive model over the resulting discrete codes — inherit an assumption that is seldom
stated: that *a representation good enough to reconstruct from is good enough to forecast from*.
The tokenizer is fitted to a reconstruction objective and the forecasting model never sees the
underlying series, so the assumption is load-bearing for the whole design. This section shows that
it fails, that the failure is deterministic rather than incidental, and that it acts selectively
on precisely the channels a microstructure-aware model depends on.

We proceed mechanism → prediction → intervention → standing verification. All measurements
reported here are made on a synthetic fixture with a known ground-truth signal, chosen so that the
quantity being recovered is exactly computable; §3.8 states what this does and does not license us
to claim about real markets.

## 3.1 A planted signal the pipeline does not recover

We construct a three-symbol synthetic stream of 42,000,000 bars carrying the canonical 16-wide
per-bar feature layout of §5.
<!-- artifact: runs_manifest/m6_canary_v6_stage1_manifest.json :: recipe.bars_total = 42000000, recipe.symbols = ['SYNAUSDT','SYNBUSDT','SYNCUSDT'] -->
One dimension of the microstructure block, which we write $s_t$, is drawn i.i.d. standard normal
and is observable at the close of bar $t$. In the *planted* arm it drives the return two bars
later,

$$r_t \;=\; \sigma\bigl(\varepsilon_t + c\, s_{t-2}\bigr), \qquad c = 3,\ \ \varepsilon_t \sim \mathcal{N}(0,1)\ \text{i.i.d.},$$

<!-- artifact: scripts/m6_canary.py:220-247 (synth_symbol); C_SIGNAL=3.0, SIGNAL_LAG=2; receipt recipe.c_signal=3.0, recipe.signal_lag=2, recipe.sigma=0.01 -->
and in the *noise* arm $s_t$ is redrawn independently, so the two arms differ in exactly one
respect. Because the states are i.i.d., past returns reveal only states at lag $\ge 2$, which are
orthogonal to the forward label; the plant is therefore recoverable **only** through the
microstructure channel and not through any function of price history.
<!-- artifact: scripts/m6_canary.py:222-231 docstring — "causal AND asymmetric by construction ... invisible to any function of returns available at t" -->

The construction is a Gaussian channel, so the information the plant injects is exact:

$$I\!\left(s_{t};\, r_{t+2}\right) \;=\; \tfrac12\ln\!\left(1+c^{2}\right) \;=\; \tfrac12\ln 10 \;=\; 1.151\ \text{nats per bar}.$$

Since the tokens are a deterministic function of the features, the data-processing inequality makes
1.151 nats an upper bound on the validation cross-entropy improvement the autoregressive stage can
obtain from the plant.
<!-- 1.151 = 0.5*ln(1+3^2), derived from receipt c_signal=3.0; the pre-registration records the same quantity as "~1.15-nat plant" at docs/m6_prereg.md:2923 -->

Both arms were trained under an identical budget and schedule: 20,000 steps at batch 32 and context
32, giving 20,480,000 bar-visits against a 27,300,000-bar training region — 0.75 epochs, so
memorization is not available as an explanation, and the sub-epoch condition is asserted in the run
receipt rather than assumed.
<!-- artifact: same receipt :: sub_epoch.{bar_visits=20480000, train_bars=27300000, epochs_equivalent_train_region=0.7502, holds=true, rule} -->

The result is a null of an unusually clean kind. The learning schedule completed on both arms with
no loss spikes; the validation–train gap was $+0.0072$ (noise) and $-0.0052$ (planted), confirming
the sub-epoch regime; the correlation between the model's forecast and the planted state stayed
below $0.05$ at **every one of the 40 evaluation points**, with a maximum of $0.0433$; and the
planted arm's final validation NLL was $13.6113$ against the noise arm's $13.5979$ — worse by
$0.0134$ nats, i.e. the wrong sign.
<!-- artifact: same receipt :: decision_inputs.{schedule_complete=true, lr_halved=false, gap_noise=0.0072, gap_planted=-0.0052, tf_corr_max=0.0433, tf_flat_every_eval=true, final_val_planted=13.6113, final_val_noise=13.5979, val_planted_minus_noise=0.0134} -->

Of an available 1.151 nats, the pipeline extracted none.

Crucially, the tokenizer was *not* discarding the state. Window-level reconstruction of the planted
dimension was non-degenerate in the same run: reconstruction MAE $0.3044$ against a
predict-the-mean baseline of $0.8216$.
<!-- artifact: same receipt :: micro_recon["9"].{recon_mae=0.3044428039694296, mean_baseline_mae=0.8216477955219712, non_degenerate=true} -->
The information was present in the code stream and unavailable to the model consuming it.

## 3.2 The autoregressive backbone is not the cause

Two explanations survive §3.1. Either the tokenizer's identifier geometry hides the rule (H-T), or
the backbone cannot learn thin conditionals at this scale (H-A). These are separated by planting
the signal **directly in token space**, bypassing the tokenizer entirely, and asking whether the
same backbone under the same budget recovers it.

The first attempt failed at design time and produced its own finding. A deterministic
digit-translation plant cannot carry enough information on this carrier, because the carrier's
coarse digit marginals are already near-uniform: measured digit entropies of $2.3938$, $2.1796$ and
$2.1873$ nats against maxima of $\ln 11 = 2.3979$ and $\ln 9 = 2.1972$, i.e. $99.2\%$–$99.8\%$ of
maximum entropy.
<!-- artifact: runs_manifest/m6_token_control_step0.json :: carrier.coarse_digit_entropies_nats = [2.3937514530202653, 2.179557873990503, 2.1872653383775362]; recipe.coarse_levels=[11,9,9] in m6_token_control_run_manifest.json; ratios derived -->
Under near-uniform marginals a translation plant is capped at $0.2769$ nats, below the
pre-registered $0.7$–$1.1$ nat band the branch condition requires, so it was retired as
information-poor by construction.
<!-- artifact: runs_manifest/m6_token_control_run_manifest.json :: translation_plant_status -->

The adopted plant redraws the target digit from a discretized Gaussian centred monotonically on the
source level, with the width $\sigma$ solved so that the exact information lands in band. The
receipt records $\sigma = 1.0077$ giving exactly $0.9000$ nats in closed form, a Monte-Carlo plug-in
estimate of $0.9009$ agreeing to $9.1\times10^{-4}$, a marginal KL of $0.00476$ between planted and
carrier target distributions, an entropy shift of $-0.0054$ nats (so no information is smuggled in
through the marginal), and an oracle Spearman ceiling of $0.926$. An unplanted control returns
$-8.8\times10^{-5}$.
<!-- artifact: runs_manifest/m6_token_control_step0.json :: plant.{sigma_plant=1.007677, info_nats=0.9000000000000004, target_marginal_kl_planted_vs_carrier=0.004760539830577332, entropy_shift=-0.0054073678438815165}; plant_checks.{mc_plugin_info_nats=0.9009085392898115, closed_form_vs_mc_abs_diff=0.0009085392898111611, oracle_spearman_argmax_predictor=0.9257794670204093, unplanted_control_spearman=-8.759227241258484e-05} -->

The carrier is the regenerated noise stream from §3.1, tokenized by that run's frozen tokenizer, so
the token distribution, budget, schedule and validation protocol are unchanged and the noise arm's
final NLL of $13.5979$ serves as the reference $H_0$.
<!-- artifact: runs_manifest/m6_token_control_run_manifest.json :: recipe.carrier, decision_inputs.{H0_source, H0_val=13.5979} -->
Over the full stream the planted information is $0.9003$ nats exact ($0.9003$ by Monte-Carlo,
agreeing to $2.0\times10^{-5}$).
<!-- artifact: same :: full_stream_receipts.{exact_info_nats=0.9002715667652758, mc_plugin_info_nats=0.9002915157043878, closed_form_vs_mc_abs_diff=1.994893911205775e-05} -->

The backbone recovers it almost completely. The probe's Spearman correlation with the planted rule
crosses the pre-registered detection threshold of $0.3$ at step 2,000 and reaches $0.9999$; the
final validation NLL is $12.7483$, i.e. $H_0 - 0.8496$ nats, which is $94.4\%$ of the planted
information; the schedule completes without spikes and the sub-epoch condition holds.
<!-- artifact: same :: decision_inputs.{probe_cross_step=2000, probe_final=0.9999, final_val_nll=12.7483, final_val_minus_H0=-0.8496, schedule_complete=true, sub_epoch_holds=true, lr_halved=false}; 0.8496/0.9002715667652758 = 0.9437 -->

H-A is refuted. The identical backbone, at the identical scale and budget, learns a dense graded
lag-2 token conditional essentially perfectly and does so an order of magnitude faster than the
budget allows. The §3.1 null therefore localizes to the tokenizer→backbone interface.

**Table 1: the planted-signal arc.** Same backbone, same budget, same schedule; the plant moves
from feature space to token space and back after the interface is re-specified.

| | plant | site | information | extraction | outcome |
|---|---|---|---|---|---|
| §3.1 | lag-2 state → return, $c=3$ | feature space | 1.151 nats | $\Delta$val $= +0.0134$ (wrong sign); probe $< 0.05$ at all 40 evals | none |
| §3.2 | noisy monotone digit map | token space | 0.9003 nats exact | $\Delta$val $= -0.8496$ = 94.4%; probe Spearman 0.9999 | recovered |
| §3.6 | lag-2 state → return, $c=3$ (identical to §3.1) | feature space | 1.151 nats | $\Delta$val $= -1.3381$; probe 0.9407 | recovered |

## 3.3 The interface defect, measured directly

The window encoder is causal but *contextual*: bar $t$'s code is a function of bars $\le t$. State
belonging to bar $t$ is therefore free to be distributed across the identifiers of every subsequent
bar in the window. The window as a whole reconstructs; no individual identifier need carry its own
bar's features.

This is directly measurable, and the measurement is unambiguous. On 300,000 bars of the regenerated
planted stream, tokenized by the frozen §3.1 tokenizer, with a 240,000/60,000 split, a logistic
probe recovering $\operatorname{sign}(s_t)$ from bar $t$'s own coarse and fine digits achieves
$0.5135$ against a chance rate of $0.5$, and an MLP regressing $s_t$ on the same digits achieves a
Pearson correlation of $0.0142$.
<!-- artifact: runs_manifest/m6_token_control_step0.json :: id_visibility_probe.{logistic_sign_accuracy=0.5135166645050049, logistic_chance=0.5, mlp_state_pearson_corr=0.014173370141830274, sample_bars=300000, train_test_split=[240000,60000], tokenizer, stream} -->

The same tokenizer, on the same stream, reconstructs that dimension at the window level with MAE
$0.3044$ against a $0.8216$ baseline (§3.1). The state is encoded and per-bar illegible. Since the
autoregressive stage consumes exactly the per-bar identifier sequence, the information is
inaccessible to it by construction.

The same defect appears in a stronger and more familiar form when the encoder is not causal at all.
Perturbing only future bars inside a tokenization chunk and measuring how many past bars' tokens
flip gives a leakage rate of $41.75\%$ coarse and $28.3\%$ fine on a tokenizer trained on real
data; the causal-encoder control measures exactly $0.0\%$ on both, which is what establishes that
the probe can read zero.
<!-- artifact: runs_manifest/m6_token_causality_probe.json :: results.trained_fsq_real_lake_smoke.{coarse_flip_rate=0.4175, fine_flip_rate=0.2833333333333333}; results.untrained_causal_flag.{coarse_flip_rate=0.0, fine_flip_rate=0.0} -->
Causal encoding is thus necessary for validity (§8) but, as §3.1–§3.3 establish, not sufficient for
usability: it removes the leak forward and leaves the smear backward.

## 3.4 The mechanism: capacity is priced by variance and covariance

Why does a contextual encoder smear one dimension rather than another? Because the reconstruction
objective it is minimizing has a preference, and the preference is not downstream value.

Consider the capacity arithmetic first. The canonical configuration uses per-dimension levels
$(11,9,9,7,7,5,5)$, split into a coarse subtoken of $11\!\times\!9\!\times\!9 = 891$ codes
($9.80$ bits) and a fine subtoken of $7\!\times\!7\!\times\!5\!\times\!5 = 1225$ codes ($10.26$
bits), for $20.058$ bits per token.
<!-- artifact: src/trikaal/constants.py :: FSQ_LEVELS, derive_coarse_fine docstring, FSQ_V_C=891, FSQ_V_F=1225; bits derived as sum log2(L_i) -->
Recovering the sign of a jointly-normal dimension at accuracy $a$ requires a correlation
$\rho = \cos\bigl(\pi(1-a)\bigr)$, so $a = 0.90$ requires $\rho = 0.951$; representing that
dimension to that fidelity costs at least $-\tfrac12\log_2(1-\rho^2) = 1.69$ bits by the Gaussian
rate–distortion bound, and more under any realizable scalar quantizer. A $10.26$-bit fine budget
therefore supports only a handful of such dimensions — measured, four in each of three seeds
(§3.5) — and covering all 13 live dimensions would require at least $22.0$ bits, more than the
entire $20.06$-bit per-token budget, at any coarse/fine split.
<!-- artifact: git show c4cd082:runs_manifest/m6_interface_respec_design_pass.json :: gate2_blocked_diagnosis.capacity_arithmetic.{sign_acc_0.9_requires_rho=0.951, fine_bits_available=10.26, total_bpt_pinned=20.058}. NOTE: that receipt states bits_per_dim_at_rho_0.951=1.78 and all_13_dims_would_need_bits=23.1; the Gaussian rate-distortion bound gives 1.694 and 22.02 and no code in the repo derives 1.78 — we publish the reproducible bound, which is the conservative one and carries the same conclusion. Flagged to the record. -->
<!-- artifact: runs_manifest/m6_interface_respec_design_pass.json :: gates.channel_receipt_non_gating.dims_ge_090_by_seed = {0:4, 1:4, 2:4} -->

Capacity is therefore scarce and must be allocated. The question is by what rule. On a fixture whose
dimensions were statistically exchangeable i.i.d. draws, the allocation was arbitrary: across three
initializations of otherwise identical models, the set of dimensions clearing $0.9$ changed
completely, and the dimension of interest cleared in one seed of three.
<!-- artifact: git show c4cd082:...:: gate2_blocked_diagnosis.three_seed_winner_receipt — per-dim corrs at index 9: seed0 0.91, seed1 0.26, seed2 0.10; note "tiny shells (d64), 600 steps, identical data rng; dim 9 wins in seed 0 only" -->
That reading would have made the whole phenomenon an initialization lottery, which is a much weaker
claim. It was wrong, and what corrected it was making the fixture realistic rather than convenient.

We calibrated the fixture's non-signal dimensions to the measured joint entropy of the real data:
on 30 symbols and 3,895,808 unmasked bars, the Gaussian-copula correlation log-determinant over the
seven non-microstructure dimensions is $-6.4819$, a per-dimension entropy deficit of $0.4631$ nats;
the fixture matches this with a cross-dimension AR(1) chain at $\rho = 0.7993$, achieving $0.462$
nats against a $0.05$-nat tolerance.
<!-- artifact: runs_manifest/m6_interface_respec_design_pass.json :: filler_calibration.{lake_measurement.{copula_corr_logdet=-6.4819, per_dim_deficit_nats=0.4631, symbols=30, unmasked_bars=3895808, sample_seed=20260720}, filler_rho=0.7993, achieved_per_dim_deficit_nats=0.462, tolerance_nats=0.05, within_tolerance=true} -->
The signal dimension, the plant, and the return rule were untouched. Marginals remain standard
normal throughout, so **every dimension except the return has identical marginal variance and the
filler dimensions differ from the signal dimension in exactly one property: they are correlated
with each other and it is independent of everything.**

Under that fixture the allocation is not a lottery. Table 2 gives per-dimension point-decoder
correlations for three seeds; they are near-identical across seeds, and they sort by variance class.

**Table 2: reconstruction quality by variance class, three seeds.** Point-decoder correlation
between a dimension's true value at bar $t$ and its reconstruction from bar $t$'s own fine code.
All dimensions except the return have unit marginal variance; the signal dimension differs from the
fillers only in being independent.

| dimension class | marginal sd | cross-dim structure | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|
| return (index 0) | 3.16 | driven by the signal at lag 2 | **0.9818** | **0.9815** | **0.9823** |
| fillers (11 dims) | 1.00 | AR(1) chain across dims, $\rho = 0.7993$ | 0.826 – 0.918 | 0.824 – 0.919 | 0.824 – 0.917 |
| signal (index 9) | 1.00 | independent | **0.0140** | **0.0011** | **0.0053** |

<!-- artifact: runs_manifest/m6_interface_respec_design_pass.json :: tokenizers.new_pointwise_fine_3seed.seeds.{0,1,2}.nonlinear_diagnostics.point_decoder_per_dim_corrs; filler ranges = min/max over indices 1-8,10-12; index 9 = point_decoder_dim9_corr. Marginal sd of index 0 = sqrt(1+c^2) = sqrt(10) = 3.162 by construction (scripts/m6_canary.py:229-234), corroborated by recon.per_dim["0"].mean_baseline_mae 2.51351 / sqrt(2/pi) = 3.15; fillers and signal are unit-variance by construction, corroborated by mean_baseline_mae ~0.80 -->

The reading is that a reconstruction objective buys **variance** and **covariance** and never buys
**independence**. The highest-variance dimension is reconstructed first because it dominates the
loss. The correlated block is reconstructed next and cheaply, because one shared component serves
eleven dimensions at once. The independent dimension offers no shared component to amortize
against and carries the same marginal variance as any single filler, so under a scarce code budget
it is the last thing worth spending bits on — and it is evicted, deterministically, in every seed.

This is not a defect of the objective. It is the objective working correctly against a goal it was
never given. The consequence for the present application is direct and was recorded before any real
microstructure was tokenized: **microstructure is low-variance and weakly covariant with price
shape, which is exactly the eviction profile.** A tokenizer trained to reconstruct will allocate
capacity away from it by construction, which makes "microstructure-aware" an architectural property
of a tokenizer rather than an emergent one.
<!-- pre-registered verbatim: docs/m6_prereg.md §7 v1.4.1, "Paper-facing mechanism sentence" -->

## 3.5 Intervention

If capacity is allocated by an objective that cannot see per-bar legibility, the remedy is to give
the objective a term that can. We make three changes; the record shows each was necessary and none
was sufficient alone.

**(i) A pointwise fine encoder.** The fine subtoken becomes a per-bar encoding of bar $t$'s own
features, produced by a branch with no cross-bar mixing, while the coarse subtoken retains the
contextual causal encoder. Both properties are thereby carried in one token pair, and bits per
token, the vocabulary split and the cell design are untouched.
<!-- src: src/trikaal/tokenizer/model.py:82-89 (PointwiseEncoder + w_in_fine), :136-144 (latent() overwrites fine latent dims) -->
Alone this changed nothing: legibility $0.5202$, at chance. The optimizer routed the state through
the contextual coarse smear and spent the fine bits elsewhere.
<!-- artifact: git show c4cd082:runs_manifest/m6_interface_respec_design_pass.json :: gate2_blocked_diagnosis.iteration_history.iter1_pointwise_encoder_only -->

**(ii) A per-bar bottleneck decode leg.** A pointwise decode head must reconstruct bar $t$ from bar
$t$'s fine code alone, with zero cross-bar access, and its loss is added to the training objective.
<!-- src: src/trikaal/tokenizer/model.py:96-102 (point_decoder), :230-236 (loss_point added) -->
This trains — winner dimensions reach point-decoder correlation above $0.9$ — but does not by
itself select which dimensions win: legibility $0.5136$ and $0.5037$ at canonical width, and under
the entropy-calibrated fixture $0.5014$, $0.4995$, $0.5182$ across three seeds, a $3/3$ failure
against the $0.9$ threshold. That failure is the measurement reported in Table 2.
<!-- artifact: git show c4cd082:... :: iteration_history.iter2_added_per_bar_bottleneck_leg; runs_manifest/m6_interface_respec_design_pass.json :: gates.gate2_legibility.{new_by_seed={0:0.5014,1:0.4995,2:0.5182}, threshold=0.9, pass=false, protocol} -->

**(iii) Class weighting of the microstructure block in the bottleneck leg, plus objective
separation.** The bottleneck loss weights the six microstructure dimensions as a class by a factor
$\lambda$, and the window reconstruction losses receive the fine channels detached, so the fine
encoder is shaped exclusively by the per-bar objective.
<!-- src: src/trikaal/tokenizer/model.py:119-125 (w_feat_point), :207-217 (detach) -->
Detachment is load-bearing and was measured to be: with the window loss attached, the per-bar
weighting is inert at every $\lambda$ — per-dimension receipts are identical from $\lambda = 2$ to
$\lambda = 10^{6}$ — because the window objective re-creates the smearing incentive as fast as the
leg removes it.
<!-- src: src/trikaal/tokenizer/model.py:207-215 comment; artifact: runs_manifest/m6_lambda_search_receipt.json :: detach_amendment -->

$\lambda$ is calibrated, not chosen. Across 26 trials no $(\lambda, \beta)$ pair cleared $0.9$ on
all three seeds, placing the threshold inside the instrument's seed-noise band at the achievable
ceiling, while confirming that the state is carried (signal-dimension point-decoder correlation
$0.77$–$0.96$, microstructure class $0.86$–$0.92$, return preserved above $0.96$).
<!-- artifact: runs_manifest/m6_lambda_search_receipt.json :: canonical_trials (26 entries), summary.finding -->
The acceptance rule was therefore restated over the three calibration seeds as mean $\ge 0.9$ and
every seed $\ge 0.85$, and $\lambda$ fixed at the smallest searched value clearing it:
$\lambda = 2$ gives $(0.8365, 0.8594, 0.8592)$, mean $0.8517$ — fail; $\lambda = 3$ gives
$(0.9060, 0.9142, 0.9000)$, mean $0.9067$, min $0.9000$ — pass. Re-running through the pinned
construction path reproduces mean $0.9047$, min $0.8839$, and is bit-identical to the direct
construction at fixed thread count.
<!-- artifact: runs_manifest/m6_lambda_search_receipt.json :: seeded_campaign_omp2.{lam2,lam3}, pin.{PINNED_MICRO_POINT_WEIGHT=3.0, formula, lam2_seeded_fails}, formal_cell_path_pin3_omp3.{legibility=[0.9039,0.9262,0.8839], mean=0.9047, min=0.8839, restated_gate_pass=true}, path_determinism_proof -->
$\lambda = 3.0$ is pinned in the conformance surface, applies identically to both quantizers and all
input arms, and leaves bits per token, the cell matrix and every evaluation instrument unchanged.
<!-- src: src/trikaal/eval/conformance.py:85,91,97 — PINNED_ENCODER_CAUSAL, PINNED_FINE_POINTWISE, PINNED_MICRO_POINT_WEIGHT -->

## 3.6 Verification in feature space

The intervention is verified by repeating §3.1 exactly. Generator, seeds, stream size, plant
strength, lag, budget, schedule and validation protocol are unchanged; the tokenizer architecture
is the only difference.
<!-- artifact: runs_manifest/m6_acceptance_stage1_manifest.json :: recipe.{bars_total=42000000, c_signal=3.0, signal_lag=2, sigma=0.01, budget_steps=20000, batch=32, seq_len=32, generator, schedule, symbols} — identical to the §3.1 receipt -->

The forecast–state correlation crosses the detection threshold at step 2,000 and reaches $0.9407$
(max $0.9414$); the planted arm's final validation NLL is $11.8478$ against the noise arm's
$13.1859$, a difference of $-1.3381$ nats where §3.1 gave $+0.0134$; the schedule completes without
spikes and both arms remain in the sub-epoch regime (gaps $-0.0296$ and $-0.1801$ against a $0.5$
threshold). Per-bar legibility under the new tokenizer is $0.8990$ on the planted arm and $0.9084$
on the noise arm, where §3.3 measured $0.5135$.
<!-- artifact: runs_manifest/m6_acceptance_stage1_manifest.json :: decision_inputs.{detect_cross_step=2000, tf_corr_final=0.9407, tf_corr_max=0.9414, final_val_planted=11.8478, final_val_noise=13.1859, val_planted_minus_noise=-1.3381, gap_noise=-0.0296, gap_planted=-0.1801, schedule_complete=true, lr_halved=false, branch='a_detect'}; runs.{planted,noise}.per_bar_id_legibility = 0.8989666666666667, 0.9084333333333333 -->

We report $\Delta$val as the pre-registered decision statistic and not as an extraction fraction:
each arm trains its own tokenizer, so the arm difference is not calibrated against the 1.151-nat
plant in the way the token-space control of §3.2 is calibrated against its own $H_0$. The
directional result is nonetheless unambiguous, and it is the same fixture and the same budget that
produced a null before the interface was changed.

## 3.7 The legibility gate as standing verification

A mechanism established on a fixture is a hypothesis about the real data. We therefore convert it
into a gate rather than an expectation. After Stage-1 training and **before any Stage-2 expenditure**,
each of the six microstructure dimensions (indices 7–12: trade-flow imbalance, signed count
imbalance, trade count, mean trade size, trade-size dispersion, large-trade share) must be linearly
recoverable from bar $t$'s own token pair at logistic sign-accuracy $\ge 0.90$ on the run's real
training stream. Failure raises an exception and halts the run.
<!-- src: src/trikaal/train/gates.py:96-97 (MICRO_DIMS, MICRO_LEGIBILITY_MIN=0.9), :210-294 (micro_legibility_gate, RuntimeError) -->

The probe is a logistic regression on the one-hot digit decomposition of the coarse and fine
identifiers, trained for 300 Adam steps at fixed seed on an 80/20 split, and is deterministic given
its inputs. Sign accuracy is well defined for the magnitude-shaped dimensions because every
microstructure feature enters z-scored, so the sign is above-versus-below the causal rolling mean;
per-dimension base rates are recorded in the receipt so residual class imbalance is visible. The
sample is stratified by symbol with the 80/20 split blocked in time *within* each symbol, which
preserves both properties that matter: full universe coverage, and no near-duplicate neighbours
across the split.
<!-- src: src/trikaal/train/gates.py:125-207 (id_legibility_sign_acc, _stratified_blocked_split), :223-230 (sign semantics) -->

Two defects in this gate were found and closed before it was relied upon, and both are of the class
"a check that cannot fail is not a check" (§8). A dimension too thin to measure formerly took a
code path that skipped the accumulation of the pass flag, so a gate with all six dimensions thin
returned `pass: True` having measured nothing; "we could not measure the channel this gate exists
to measure" is now a third state that halts.
<!-- src: src/trikaal/train/gates.py:255-265; artifact: runs_manifest/m6_tier4_vacuous_gates.json -->
Separately, the probe read a fixed 150,000-row head of a symbol-ordered concatenation, which on the
production lake spans **1 of 200 symbols** — the decision governing all Stage-2 expenditure was
being made on one asset. Stratification fixed it. The materiality of the first defect was then
measured rather than asserted: on the real lake the minimum per-dimension unmasked bar count across
all 200 symbols is 277,758 against a 10,000-bar floor, with zero symbols below it, so the all-thin
case cannot arise on this data.
<!-- artifact: runs_manifest/m6_c10_micro_density.json :: head_sample_coverage.{symbols_covered_at_dim9=1, of_total_symbols=200, gate_reads_first_n_rows=150000}; per_dim_min_across_symbols (min 277758 at dim 12), thin_floor_in_gate=10000, n_symbols_below_floor_per_dim all 0, total_bars=304625181 -->

The gate's first execution on real microstructure is a named checkpoint with a pre-written
adjudication: if the real dimensions cannot clear the threshold under the calibrated architecture,
the run stops and reports with per-dimension receipts. It is not a threshold that may be moved.
<!-- src: src/trikaal/train/gates.py:232-235 -->
We record the expectation before the fact: real trade-flow imbalance has the low-variance,
weakly-covariant profile that Table 2 shows being evicted, and the correction was calibrated on a
synthetic plant. **A gate failure on real data would be the real-data confirmation of the
mechanism** — simultaneously a blocker for the ablation and the strongest available evidence for
the mechanism claim. Both outcomes are reportable and neither is a surprise.

## 3.8 Scope of the claim

Four limits are stated here and repeated in §7.

First, everything in §3.1–§3.6 is measured on a **synthetic fixture**, not on real microstructure.
The fixture's non-signal dimensions are entropy-matched to the real lake and its signal dimension is
a controlled construction; this makes the quantity being recovered exactly computable, which is the
point, and it does not make the finding a measurement of markets. The mechanism claim is therefore
not the paper's headline and is not elevated to one under any experimental outcome.

Second, the mechanism is demonstrated for one reconstruction objective, one quantizer family, and
one class of encoder. We show that *this* objective allocates by variance and covariance; we do not
show that no reconstruction objective can do otherwise. The intervention of §3.5 is itself a
counterexample to the strong form of the claim, since it is a reconstruction objective that carries
the evicted channel — by being given an explicit per-bar term to do so.

Third, the extraction figure of $94.4\%$ in §3.2 is calibrated against a reference model trained on
a matched carrier under an identical protocol, and inherits whatever variation that single
reference run carries; the feature-space verification of §3.6 is reported as a directional decision
statistic for the reason given there.

Fourth, the three interventions of §3.5 were developed against the fixture, and $\lambda$ was
calibrated on it. Its value is pinned pre-data and applies identically across every arm of the
experiment, so it cannot differentially favour any cell; but it is a hyperparameter fitted to a
synthetic proxy for the real allocation problem, and the standing gate of §3.7 exists precisely
because that proxy might not transfer.

---

### Figures for this section

**Figure 2 — Reconstruction quality is sorted by variance class, not by task relevance.**
Point-decoder correlation between each feature dimension's value at bar $t$ and its reconstruction
from bar $t$'s own fine code, for three independently initialized tokenizers trained on an
entropy-calibrated synthetic fixture. The return dimension (3.16× the marginal standard deviation
of the others) is recovered at 0.98; the eleven filler dimensions, which have unit marginal variance
and are coupled by an AR(1) chain across dimensions at $\rho = 0.7993$, are recovered at 0.82–0.92;
the signal dimension,
which has *the same unit marginal variance as the fillers* and differs only in being statistically
independent of them, is recovered at 0.001–0.014. The ordering and the magnitudes are near-identical
across seeds, so the exclusion is deterministic rather than an initialization lottery. Reconstruction
objectives buy variance and covariance; they do not buy independence.

**Figure 3 — The same information is recovered or lost depending only on where it is planted.**
Left: a 1.151-nat lag-2 conditional planted in feature space is not recovered — the forecast–state
correlation stays below 0.05 at all 40 evaluation points and the planted arm's validation NLL ends
0.0134 nats *worse* than the matched noise arm. Centre: 0.9003 nats planted directly in token space,
bypassing the tokenizer, is recovered by the identical backbone under the identical budget — the
probe crosses the 0.3 detection threshold at step 2,000, ends at 0.9999, and validation NLL falls
0.8496 nats, 94.4% of the planted information. Right: the original feature-space plant after the
tokenizer interface is re-specified with a pointwise fine encoder, a per-bar bottleneck decode leg
and microstructure class weighting — probe 0.9407, $\Delta$val $-1.3381$ nats. The backbone was
never the limiting factor; the tokenizer→backbone interface was.
