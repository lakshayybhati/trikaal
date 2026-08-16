# THE CSV DASHBOARD — WHAT IT IS, WHAT IT REFUSES, AND WHAT IT COST

`scripts/m6_csv_dashboard.py`, served at `http://127.0.0.1:8765`. Drop in an OHLCV CSV for any
asset, pick a horizon, get three seeds' forecasts back with the disclosures rendered into the
figure. Local, offline, no new dependency.

---

## 1. THE PRIVACY CLAIM IS ENFORCED, NOT PROMISED

The page renders the sentence *"Runs entirely on this machine … never written to disk and never
uploaded."* Three mechanisms make it true, and the third is the one that survives future edits:

1. **Loopback bind.** `ThreadingHTTPServer(("127.0.0.1", port), …)`.
2. **Nothing written.** The uploaded text is a local variable in `_execute`. `Job` has no field
   that could hold it, and `tests/demo/test_csv_dashboard.py` pins the field set — adding a place
   to keep the bars fails the suite. A separate AST test asserts no `open`/`write_text`/`mkdir`
   anywhere in the module.
3. **Content-Security-Policy.** Every response carries
   `default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:;
   connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'`.
   A future edit that adds a CDN font, an analytics beacon or a remote image is **blocked by the
   browser**, not shipped quietly.

**Verified over HTTP, not by inspection:** the served page contains zero absolute URIs of any kind,
and every network request the browser made during a full run went to `127.0.0.1:8765` — `GET /`,
`POST /run`, `GET /stages`, `GET /result`, `GET /export.csv`. Nothing else.

**The reference design was not copied wholesale for exactly this reason.** It loads Poppins and
JetBrains Mono from `fonts.googleapis.com`. That request tells Google the user opened the tool, on
every load, which would make the rendered sentence false. The page uses a system font stack; the
design survives it.

## 2. HISTORY STORES RESULTS, NEVER INPUTS

In memory only, session-scoped, erased when the server stops. Nothing touches disk.

Kept: the forecast numbers, the rendered figure, the filename, row count, inferred interval,
horizon, timestamp and the stage trace. **The figure contains the 512-bar context path drawn from
the user's file** — that is stated in the History panel verbatim rather than glossed, because
"we keep only results" would otherwise be a half-truth. Never kept: the uploaded bars.
`CLEAR HISTORY` wipes the client array *and* `POST /forget` clears the server's job store; both
were checked (after clearing, `GET /stages?job=…` returns `unknown job`).

## 3. THE LOADING SCREEN IS A TRACE, NOT AN ANIMATION

The stage lines are emitted **by the production pipeline**, from the statements that do the work,
carrying values read off the real objects. `forecast_from_csv` and `parse_csv` take an optional
`progress(stage, detail, state)` callback whose default is a no-op; the dashboard supplies a sink
that appends to the job record. No stage list is pre-rendered — a line appears only once its stage
has actually started.

A run emits **22 stages**. Two are worth calling out:

* `VERIFYING MODEL IDENTITY` runs **before the file is read** and is a real check, not a startup
  ritual replayed: `reverify_identity` recomputes `artifact_hash(module.state_dict(),
  module.get_config())` from the **live modules about to run** and requires
  `recomputed == stored == manifest`. `load_unit` hashes the state dict it read off disk, so a
  weight corrupted in memory afterwards leaves its check untouched. The mutation test perturbs one
  parameter and asserts the stored/manifest strings are *identical to before* while the
  recomputation differs — which is what proves the recomputation is load-bearing. If it fails the
  run stops there.
* `INFERRING BAR INTERVAL` has its own `warn` state. An out-of-distribution interval completes
  successfully and **may not finish quietly green**.

## 4. ★ IT TAKES 5 TO 6 MINUTES PER FORECAST ON CPU, AND THAT IS THE HEADLINE OPERATIONAL FACT

Measured end to end, three seeds, h=15, defaults (`mc_samples=32`, `n_mc_paths=48`):

| input | wall clock |
|---|---|
| 5,999 rows (real BTCUSDT) | **328.1 s** |
| 1,300 rows (synthetic) | **342.7 s** |
| 1,400 rows (synthetic) | **319.9 s** |

Where it goes, per seed: `FORECASTING` ≈ 39–47 s, `SAMPLING TRAJECTORIES` ≈ 47–70 s. Everything
else — parse, features, tokenize, quantiles, render — totals **under a quarter of a second**.

The cost is the Monte-Carlo rollout: `mc_paths` calls `_fill_context` once per sampled path, so a
512-bar forward pass runs 48 times per seed, and `predict_mu(estimator="mc_mean")` does the same 32
times. Row count is nearly irrelevant; sample count is everything.

**This reframes the loading screen from decoration to necessity** — five minutes of a blank page
reads as a hung tool. It is also a real, unrequested optimisation target: the context is *identical*
across samples within a seed, so the caches could be built once and copied. That is a change to a
function gated bit-for-bit against production (`test_mc_paths_endpoint_matches_production`), so it
is **not** made unilaterally. Flagged for a ruling, not actioned.

## 5. THE FIGURE — THREE DEFECTS FOUND BY RENDERING THE OLD ONE AND LOOKING AT IT

None was visible from the code, and 818 tests could not see any of them.

1. **THE SEED LEGEND WAS INVISIBLE.** Legend marks were drawn at y=508; the disclosure box spanned
   492–578 and was appended *after* them, so it painted over them. Deterministic — no font metrics
   involved. The shipped figure had no legend at all.
2. **THE OVER-DISPERSION DISCLOSURE WAS CLIPPED OFF THE RIGHT EDGE.** SVG `<text>` does not wrap. A
   210-character line at 10.5 px ran to x≈1178 in a 1020-wide viewBox and ended mid-word at
   *"…vs the returns it forecast"*. The single caveat the supervisor named as must-survive was **cut
   in half by the image whose job is to carry it.** Now wrapped in a **monospace** face at a
   character budget computed from the box width — monospace advance is exactly 0.6em, so the fit is
   arithmetic rather than eyeballed, and each line is asserted at *its own rendered font size*.
3. **THE FORECAST WAS 2.8% OF THE PLOT WIDTH.** 512 context bars against 15 forecast steps on one
   linear axis. The x-axis is now split — context compressed into 62%, forecast expanded into 38% —
   with the break drawn, labelled `◀ 512 BARS OF CONTEXT, COMPRESSED | NEXT 15 MIN, EXPANDED ▶`,
   and named in the subtitle: *"the two sides are NOT one scale."*

### Where design and honesty pulled against each other, and what won

On a shared y-axis the whole forecast fan renders as **a flat line** — it is ~0.1% against context
bars that move 2–3%. That flatness is the truest thing the figure says and it **stays**. But it also
makes the sampled distribution invisible, so the same numbers are drawn again in an **own-scale
inset** below, with its own y-axis labelled in percent and a caption reading *"THE SAME FORECAST AT
ITS OWN Y-SCALE (%). THE PANEL ABOVE SHOWS ITS TRUE SIZE AGAINST YOUR BARS."*

Both readings are present; neither replaces the other. A cropped inset still carries its own axis
values and its own caption, which is what the context-stripping rule demands of a panel that can
travel alone. A test asserts the main panel still draws both percentile bands for every seed at the
shared scale — if it ever stops, the figure has started hiding its own main finding.

### What is unchanged, at full strength, in the raster

OOD banner (red box, thicker stroke, loud at half size) · `25-446x OVER-DISPERSED` · "read the band,
not the wiggle" · the ~0.55z quantization note · edge size and net-negative after fees · three seeds
never averaged · no P&L · not investment advice. The page-level panel repeats them at 12.5 px in the
`0.86` text ramp, and a test asserts that block contains **no** `--t4` / `0.22` opacity.

## 6. TWO BUGS THE TESTS COULD NOT SEE

* **The filename parse.** A real browser sends `Content-Type: text/csv` after
  `Content-Disposition`; the reader split the whole header block on `;`, so the filename token ran
  past the closing quote and the app displayed `BTCUSDT_1m.csv"\r\nContent-Type: text/csv` as the
  user's filename. **The unit test was green throughout, because the multipart body it hand-built
  had no `Content-Type` line — a fixture narrower than reality.** Found by uploading a file through
  the UI and reading the card. Both shapes are now parametrized, and the pre-fix source was restored
  in an isolated tree to watch the new test fail (`browser-form` fails, `bare-form` passes — which
  is exactly why the original was green).
* **A test selector that quietly changed target.** Two tests picked "the rect with `rx=6`". When the
  own-scale inset was added it also had `rx=6`, came first in document order, and one test began
  measuring the inset instead of the disclosure box — **and still passed**, because enough mono text
  happened to fall inside the inset to satisfy its line count. Both now select by `id`.

## 7. WHAT WAS NOT COPIED FROM THE REFERENCE

CDN fonts (§1) · the component framework (`x-dc`/`DCLogic`, `sc-for`, `sc-if`, `{{ }}` bindings —
the design was extracted, the code was not) · credits and the asset library · the five-item rail
(three items: new forecast, history, about) · **the assistant persona**. There is no invented prose
and no simulated thinking: the system card contains the measured read line and the figure. The
result line is typed in as motion, which is real text arriving; nothing is authored on the model's
behalf. The credit pill became `LOCAL · NOTHING UPLOADED`.

## 8. OPEN, FOR A RULING

1. **The MC speed-up** (§4). Roughly a 3–5× wall-clock cut for a change inside a bit-for-bit-gated
   function. Not made unilaterally.
2. **The own-scale inset** (§5) is my addition, not in the spec. It is defensible and tested, but it
   is a second panel showing the same numbers larger, which is a shape worth a second opinion.
