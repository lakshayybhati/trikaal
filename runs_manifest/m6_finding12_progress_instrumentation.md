# TASK C — EVAL PROGRESS INSTRUMENTATION: DIFF FOR REVIEW, NOT A COMMIT

**HEAD `055d14f` is untouched. Nothing below is applied. No file in the repo was modified.**

Finding 12, now costed: **the eval leg emitted nothing for 14.65 h**, and a 14-hour silent GPU
loop is indistinguishable from a hung one. That is what made this episode unreadable.

---

## 1. WHERE THE TIME ACTUALLY GOES (measured, not assumed)

The supervisor's pointer was to `src/trikaal/run/matrix.py`; the loop is one level down, in
**`src/trikaal/eval/xsection.py`**, inside the `grid_series` closure of `score_cell`.

`grep -rn "wandb" src/trikaal/` returns **`train/orchestrator.py` only** — the supervisor's claim
is exactly right: **nothing in the eval path logs anything, to W&B or to stdout.**

Per unit, `matrix.py:317` calls `score_cell` once per horizon in `DSR_HORIZONS = (5, 15, 60)`,
with `val_only=(h != PRIMARY_H)`. `grid_series` is called at `xsection.py:365` (val) and `:400`
(headline):

| horizon | val_only | grid_series calls | symbol-passes |
|---|---|---|---|
| h=5  | True  | 1 (val)            | 40 |
| h=15 | False | 2 (val + headline) | 80 |
| h=60 | True  | 1 (val)            | 40 |
| | | **4 passes** | **160** |

**160 `predict_mu` invocations per unit over 52,745.9 s ⇒ one every ~330 s (5.5 min).**

This also explains the manifest's own warning: `eval_decisions.by_h = {5:0, 15:1402520, 60:0}`
counts **only the headline pass**, while three of the four passes are VAL κ-searches that score
rollout compute the decision counter never sees. The driver already says so in
`eval_throughput.NOT_COMPARABLE_TO` — I am citing that, not discovering it.

---

## 2. CHOICE OF N — AND WHY NOT INSIDE `predict_mu`

**N = one symbol ≈ 35,063 decisions.**

- The symbol loop is where the work is dispatched; `predict_mu` is called exactly once per symbol.
- **160 callback invocations against 1,402,520 headline decisions is unmeasurable** — roughly
  1 call per 8,766 decisions, each a formatted print.
- **I deliberately did not instrument `predict_mu`'s chunk loop.** It is the hot path, it is a
  v1.4.2/v1.6.15 pinned instrument (`estimator=`), and per-chunk counting would put a Python-level
  branch inside the 512-decision inner loop that produces the money numbers. Granularity of 5.5 min
  does not justify touching the instrument the verdict depends on.

---

## 3. THE DIFF

### 3.1 `src/trikaal/eval/xsection.py` — the only production-logic change

```diff
@@ def score_cell(
 def score_cell(
     run: str,
     model,
     tok,
     arm: str,
     symbols: list[SymbolEval],
     cfg: XSectionConfig,
     *,
     val_only: bool = False,
+    progress: Callable[[dict], None] | None = None,
 ) -> CellScore:
     """Score ONE trained (cell, seed) model cross-sectionally: VAL κ-search → pooled headline.
 
     ``val_only=True`` stops after the VAL block (the per-κ curve + κ*): the verdict artifacts
     need the h∈{5,60} secondary horizons ONLY as VAL trial entries (prereg §3 clause 5), and
     the headline grid at those horizons would be wasted rollout compute. Headline fields come
     back NaN/None and must never be read from a val_only score.
+
+    ``progress`` is an OPTIONAL observer called once per scored symbol with a plain dict. It is
+    a NO-OP BY DEFAULT and touches no number this function returns — §7 finding 12: a 14.65-hour
+    eval leg that emits nothing cannot be distinguished from a hung one. The callback is supplied
+    by the DRIVER, so ``trikaal.eval`` stays free of W&B exactly as it is today (`wandb` is
+    imported in `train/orchestrator.py` and nowhere else).
     """
```

```diff
@@ inside grid_series, the per-symbol loop
         traded_by_k: dict[float, int] = {k: 0 for k in cfg.kappas}
         n_scored = 0
+        t_pass = time.perf_counter()
+        n_syms = len(prepared)
         for sym, d in prepared.items():
             se = d["se"]
             dec = symbol_decisions(se, grid, cfg)
@@
             mu_pool.append(mu)
+            if progress is not None:
+                el = time.perf_counter() - t_pass
+                done = len(n_dec)
+                progress(
+                    {
+                        "run": run,
+                        "phase": "val" if val_only else "grid",
+                        "h": cfg.h,
+                        "symbol": sym,
+                        "symbols_done": done,
+                        "symbols_total": n_syms,
+                        "decisions_done": n_scored,
+                        "elapsed_s": el,
+                        "ms_per_decision": (el / n_scored * 1000.0) if n_scored else float("nan"),
+                        # ETA for THIS PASS ONLY. A unit is 4 passes of unequal size (the headline
+                        # grid spans forward blocks 1..k-1; VAL is block 0), so this must never be
+                        # read as an ETA for the unit.
+                        "eta_pass_s": (el / done * (n_syms - done)) if done else float("nan"),
+                    }
+                )
```

Plus the two imports at the top of the file:

```diff
 from __future__ import annotations
 
+import time
+from collections.abc import Callable
 from dataclasses import dataclass, field
```

### 3.2 `src/trikaal/run/matrix.py` — pass it through

```diff
+        def _progress(ev: dict) -> None:
+            if spec.progress is not None:
+                spec.progress({**ev, "unit": unit, "cell": cell.cell_id, "seed": seed})
+
         val_by_h, kappa_by_h, dec_by_h, head = {}, {}, {}, None
         for h in spec.horizons:
             sc = score_cell(
                 f"{unit}_h{h}",
                 model,
                 tokm,
                 cell.arm,
                 spec.sym_evals,
                 replace(spec.base_eval_cfg, h=h, seed=seed),
                 val_only=(h != PRIMARY_H),
+                progress=_progress,
             )
```

with `progress: Callable[[dict], None] | None = None` added to the spec dataclass that already
carries `horizons` (`matrix.py:147`).

### 3.3 `scripts/m6_money_run.py` — where stdout and W&B are actually produced

```diff
+def _eval_progress(wandb_run):
+    """stdout (→ the shard log) + W&B. The eval path stays W&B-free; this is the seam."""
+
+    def emit(ev: dict) -> None:
+        print(
+            f"[eval:{ev['unit']}] h={ev['h']} {ev['phase']} "
+            f"{ev['symbols_done']}/{ev['symbols_total']} {ev['symbol']} "
+            f"dec={ev['decisions_done']:,} elapsed={ev['elapsed_s']:.0f}s "
+            f"{ev['ms_per_decision']:.2f} ms/dec eta_pass={ev['eta_pass_s']:.0f}s",
+            flush=True,
+        )
+        if wandb_run is not None:
+            wandb_run.log(
+                {
+                    f"eval/{ev['unit']}/h{ev['h']}/{ev['phase']}/decisions": ev["decisions_done"],
+                    f"eval/{ev['unit']}/h{ev['h']}/{ev['phase']}/ms_per_decision": ev["ms_per_decision"],
+                    f"eval/{ev['unit']}/h{ev['h']}/{ev['phase']}/symbols_done": ev["symbols_done"],
+                }
+            )
+
+    return emit
```

**`flush=True` is load-bearing.** The shard log is a redirected file, so stdout is block-buffered;
without the flush the progress lines would sit in a 4 KiB buffer and the log would still look
dead. That is the same defect in a new costume.

---

## 4. THE W&B QUESTION I COULD NOT SETTLE FROM THE CODE

`orchestrator.py:289` creates the W&B run and `:403` calls `wandb_run.finish()`. **I have not
established whether that run is still open when `score_cell` runs, or whether eval happens after
`finish()`.** If it is already finished, the eval leg needs its own `wandb.init` (or the
orchestrator's `finish()` must move), and that is a design choice, not a bug fix.

I am flagging it rather than guessing. **The stdout half works regardless and is the half that
would have made these 14 hours readable** — W&B is the dashboard, the shard log is the record.

---

## 5. WHAT THIS DOES NOT DO

- Does not change any number `score_cell` returns — the callback is an observer and the default
  is `None`.
- Does not touch `predict_mu`, the pinned estimator, or any gate.
- Does not resume, checkpoint, or persist partial eval state. **The eval still writes nothing
  until the end, so a box loss still erases the whole unit.** That is a separate and larger
  change, and I am not smuggling it in here.

## 6. TESTS I WOULD SHIP WITH IT (per the mock rule — assert the CALL, not a stub's reply)

1. `progress=None` (the default) leaves every returned field **bit-identical** to the current
   code on a fixture — the observer is proven inert.
2. A recording callback receives **exactly `len(prepared)` events per pass**, with
   `symbols_done` strictly increasing 1..N and `decisions_done` non-decreasing — asserting the
   **arguments the production code passes**, not a stub's return value.
3. `flush=True` is asserted on the print call, because a buffered progress line is the defect.
4. A negative control: deleting the `progress(...)` call makes test 2 **fail**. A progress test
   that passes against code emitting nothing is a check that cannot fail.
