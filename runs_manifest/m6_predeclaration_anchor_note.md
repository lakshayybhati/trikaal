# COMPANION NOTE TO THE UNITS-1-AND-2 PRE-DECLARATION — HASH ANCHOR AND ARTIFACT-ABSENCE RECORD

**This file is SEPARATE from the hashed pre-declaration on purpose.** `m6_units_1_2_predeclaration.md`
is frozen at the hash below and is not touched again until the units land; anything learned after
the hash goes here and is hashed on its own.

---

## THE ANCHOR

    file    runs_manifest/m6_units_1_2_predeclaration.md
    sha256  8f4dc6471fab2184f2296088542bd6f0812aa387cb0a882f09078c726c66b5df
    time    2026-08-12T20:21:01Z   (independently recomputed and matched 2026-08-12T20:24:28Z)

Computed on both sides and byte-identical. The file cannot be committed before the units land —
that would move `git_commit`, identity key #2, and re-cut the matrix — so the hash is the anchor
in its place. When the post-run batch commits it, the committed bytes can be verified against a
hash published before the data existed.

**The net line (15.6616) has a stronger anchor already:** it is inside pushed commit `448e4fe`,
which provably predates both units. The gross section does not, which is why this note exists.

---

## THE PRECISE ARTIFACT-ABSENCE STATEMENT

> At **2026-08-12T20:21:01Z**, sha256
> `8f4dc6471fab2184f2296088542bd6f0812aa387cb0a882f09078c726c66b5df`, **NO MONEY-RUN cell-4 or
> cell-5 artifact existed anywhere in the repository.** `runs_cloud/results/` held exactly three
> artifacts, all `cell1`, and both boxes reported `artifacts=0`.
>
> **Thirty-two files bearing `cell4_`/`cell5_` eval filenames exist across five fixture
> directories** — `runs/m6_dryrun/eval` (6), `runs/m6_seedflag/eval` (4), `runs/m6_toy/eval` (6),
> `runs/m6_valgate/eval` (10), `runs_cloud/runs/m6_toy/eval` (6). **Every one** is schema
> `m6_cell_eval_v1`, carries **no** `money_run`, `grid_pinned` or `git_commit`, and is **refused by
> `_validate_artifact` on schema before any other clause** (32/32 schema-first). Their `n_periods`
> are **8784 (20 files), 2208 (6) and 1128 (6)** — all against the pinned **35,064**.
>
> Money-run cell-4/cell-5 artifacts anywhere in the repository at that moment: **0**.

---

## HOW THIS STATEMENT WAS ARRIVED AT — THREE SCOPING ERRORS, EACH CAUGHT BY THE NEXT PASS

The value of the statement is that it survived three narrowings, so the narrowings are recorded
rather than the conclusion alone.

1. **"No cell-4/cell-5 artifact exists."** Asserted from `ls runs_cloud/results/*/`, which sees one
   directory. Six files existed under `runs_cloud/`. **"No artifacts exist" would have been false
   in the letter and true in the substance, and the letter is what a reviewer checks.**
2. **"Six files, all n_periods 1128."** Scoped to `runs_cloud/` only, and the `n_periods` claim was
   generalized from the **two** files actually opened. The supervisor's scope was narrower still —
   `runs_cloud/results/*/`. **Neither side had swept the repository.**
3. **The repo-wide sweep** (`**/cell[45]_seed*_eval.json`) returns **32** across five directories,
   and shows `n_periods` is **NOT** uniformly 1128 — it is `{8784: 20, 2208: 6, 1128: 6}`. The
   "1128" clause propagated from my two-file reading into the supervisor's restatement before the
   sweep caught it.

What survives all three passes unchanged: **schema v1 on 32/32, refused schema-first on 32/32, no
`money_run`/`grid_pinned`/`git_commit` on 32/32, and zero money-run cell-4/cell-5 artifacts.** The
substance never moved; only the precision of the sentence did.

---

## PRE-REGISTERED RESOLUTION LIMIT, STATED BEFORE THE NUMBERS

With `SE_gross ≈ 1.0` per measurement, a single paired `ΔIR_gross₀` is resolvable to roughly
**±1.4** at best, while the gross IRs themselves are only **1.1–2.2**. **One seed cannot settle the
gross question.** What it yields is the sign of the contrast, the activities, and whether cells 4
and 5 have gross edge at all. That is worth having and it is **not a power verdict**.

The inverted gross construction (predictions 1.3550 vs 0.7936, wrong order, 1.7× apart) is
recorded in the hashed file and **no replacement construction has been built** — a line whose
hypotheses sit on the wrong side of each other cannot be failed in either direction.
