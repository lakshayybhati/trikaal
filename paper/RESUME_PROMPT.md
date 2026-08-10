# Resume prompt — Trikaal paper

You are writing a research paper for arXiv + ICAIF/ML-workshop submission. Impersonal scientific
register, "we" throughout, no reference to tooling or process. The work is presented as the
author's own.

## Sources of truth, in strict priority order
1. Committed artifacts — `runs_manifest/*.json`, `src/`, `docs/m6_prereg.md` §7 amendment log,
   milestone docs. **Every quantitative claim traces to one of these.**
2. `docs/BUILD_RECORD.md` — the map of what exists and where. Anything marked
   `[recollection — no artifact]` never enters the paper.

Annotate each number with its artifact path in a `%` comment; strip before final.

## Where things are
- `paper/main.tex` · `preprint.sty` (NeurIPS-style; official venue `.sty` swaps in at submission)
- `paper/sections/{03_mechanism,04_design,appendix_mechanism}.tex`
- `paper/figures/make_fig*.py` — one script per figure, all regenerate from receipts
- `paper/figures/figstyle.py` — **the permanent palette; colours are never reassigned**
- `paper/review_s3.pdf` — the §3 export sent for cold review

## Build and verify — always report exit codes explicitly
```bash
cd paper && tectonic -X compile main.tex && cd .. && .venv/bin/ruff check . && .venv/bin/ruff format --check .
```
Also check `grep -c undefined paper/main.log`. A verification you did not see is not a verification.

## Current state
**Done:** §3 (reviewed, both mandatory edits applied) · §4 (complete, awaiting review) ·
Figs 1–10 wired · Appendices A–C · placeholder discipline document-wide.

**Next, in order:** §4 review → Fig 1 revision (drop the bottom Δval strip; Fig 3 owns that
story) → §7 Limitations *in full* (highest priority missing section) → §5 → §8 → §2 →
References → Appendices D/E → §6 stubs with both interpretation paragraphs pre-written and dated.

**Held:** Fig 3 stays a placeholder until §3's review closes. Title and abstract stay unlocked
(draft both mechanism-led and ablation-led variants) until the legibility gate and verdict land.

**Length:** the primary artifact is the full-length arXiv version. Write complete. A venue-length
variant is a separate later pass. Do not pre-shorten.

## Standing rules — every one of these was learned by violating it
- **Placeholder discipline.** Values in prose or captions carry `\PH{...}`. Placeholder *data*
  must be implausible by construction — absurd magnitudes, symmetric about zero, or absent
  entirely. A watermark is not enough: a cropped panel loses the watermark and keeps the numbers.
  A draft that manufactures realistic favourable results is the defect this project exists to stop.
- **A check that cannot fail is not a check.** Any probe, gate or sweep you build must be shown
  capable of failing before its output counts. Mutation-test it; give it a negative control.
- **Separate your probe's bugs from its findings.** The receipt sweep reported 19 findings, then
  8, then 7 as its own defects were removed. The citation checker called truncated code snippets
  "refuted". Report only what survives a corrected instrument.
- **Prose is weaker evidence than a receipt.** When they disagree, say so and cite both.
- **Verify claims against the commit they were made about** (`git show <sha>:<path>`) — line
  numbers drift.
- **State the uncomfortable thing.** The embargo premise fails for absolute returns; θ = κ·c never
  bound on the fixture; NULL or INCONCLUSIVE is the most likely honest outcome. All in the paper.

## Open items carried forward
- **1.78 bits/dim is unsourced** — ruled 2026-08-09, not published; §3.1 uses the reproducible
  1.69 rate–distortion bound. Ruling recorded in-source at the point of use.
- **7 hand-authored receipt blocks** in 4 receipts (`runs_manifest/m6_receipt_provenance_sweep.json`)
  — `cost_split_MEASURED` and the two `m6_integrated_price` blocks are the sharp ones. Not yet
  ledgered in BUILD_RECORD; awaiting the operator's call.
- **51 refuted + 148 manual facts quarantined** (`runs_manifest/m6_s4_fact_verification.json`).
  424 confirmed. Nothing from the quarantine enters the paper without hand-adjudication.
- **Prose-vs-artifact drifts found:** |return| ACF is 0.2323/0.3604, not "0.20"; "41 delisted" is
  the ingest-scan count, 17 is the in-window count — §5 must keep the distinction.
- **Cell 6 is not in the pre-registration** (zero mentions). It is future work, never a committed
  contingency. The capacity handicap stays unquantified in IR units.

## Figure standard
Verdict visible before the caption is read. Vector PDF. Direct labelling over legends. Every
number loaded from its receipt at render time, never transcribed. Captions self-contained — a
reader who sees only captions learns the paper. Status encoded redundantly, never colour alone.
