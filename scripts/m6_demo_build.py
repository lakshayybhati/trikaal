"""Build the v1 demo: real forecasts from the banked cell-1 units, rendered to a static page.

A SELF-CONTAINED STATIC ARTIFACT IS THE HONEST FORM HERE. The bytes shipped are the bytes
reviewed: there is no server that could later be extended to compute something it should not, no
runtime dependency to drift, and the page can be hosted anywhere (a static Space, GitHub Pages,
or opened from disk). Every forecast on it is a real production-path forecast, and the acceptance
gate (``scripts/m6_demo_acceptance.py``) proves that path is bit-identical to the scored one.

REFUSES TO BUILD unless the acceptance receipt exists and reads PASS. A demo that has not been
shown to reproduce the scored model is a demo showing an unverified model, and shipping it would
be verifying EXISTENCE where BEHAVIOUR is required.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from trikaal.demo.inference import (  # noqa: E402
    DEMO_H,
    available_seeds,
    forecast_at,
    load_unit,
    prepare_symbol,
    use_units_from,
)
from trikaal.demo.render import render_svg  # noqa: E402
from trikaal.utils.paths import display_path  # noqa: E402

sys.path.insert(0, str(REPO / "scripts"))
from m6_demo_acceptance import pick_decisions  # noqa: E402

PAGE_CSS = """
body{margin:0;font:15px/1.55 system-ui,sans-serif;background:#f8fafc;color:#0f172a}
main{max-width:1000px;margin:0 auto;padding:28px 18px 64px}
h1{font-size:26px;margin:0 0 6px}
.lede{color:#475569;margin:0 0 18px}
.banner{border:1.5px solid #dc2626;background:#fef2f2;border-radius:8px;
 padding:14px 16px;margin:18px 0}
.banner b{color:#991b1b}
.banner p{margin:6px 0 0;color:#7f1d1d;font-size:14px}
figure{margin:26px 0;background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:12px}
figure svg{width:100%;height:auto;display:block}
figcaption{color:#64748b;font-size:12.5px;margin-top:8px}
code{font:12.5px ui-monospace,Menlo,monospace;background:#f1f5f9;padding:1px 5px;border-radius:4px}
footer{margin-top:36px;color:#64748b;font-size:12.5px;border-top:1px solid #e2e8f0;padding-top:16px}
@media (prefers-color-scheme:dark){
 body{background:#0b1220;color:#e2e8f0}.lede{color:#94a3b8}
 figure{background:#111a2e;border-color:#1e293b}figcaption{color:#94a3b8}
 code{background:#1e293b}footer{color:#94a3b8;border-color:#1e293b}
 .banner{background:#2a1215;border-color:#f87171}.banner b{color:#fecaca}.banner p{color:#fca5a5}
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--per-symbol", type=int, default=2)
    ap.add_argument("--out", type=Path, default=REPO / "demo/index.html")
    ap.add_argument(
        "--acceptance",
        type=Path,
        default=REPO / "runs_manifest/m6_demo_acceptance.json",
    )
    ap.add_argument(
        "--units",
        type=Path,
        default=None,
        help=(
            "Directory holding the three cell-1 units (any layout — the repo tree or a "
            "HuggingFace download). Units are found by content: a folder with run_manifest.json, "
            "predictor.pt and tokenizer.pt. Omit to use the in-repo runs_cloud/ paths."
        ),
    )
    args = ap.parse_args()
    use_units_from(args.units)

    # ── REFUSE WITHOUT A PASSING ACCEPTANCE RECEIPT ───────────────────────────────────────────
    if not args.acceptance.exists():
        print(
            f"REFUSING TO BUILD: no acceptance receipt at {args.acceptance}. Run "
            "scripts/m6_demo_acceptance.py first — an unverified demo shows an unverified model.",
            file=sys.stderr,
        )
        return 2
    acc = json.loads(args.acceptance.read_text())
    if acc.get("verdict") != "PASS" or acc.get("n_mismatch", 1) != 0:
        print(
            f"REFUSING TO BUILD: acceptance verdict is {acc.get('verdict')!r} with "
            f"{acc.get('n_mismatch')} mismatches. The demo path is not proven bit-identical to "
            "the scored path.",
            file=sys.stderr,
        )
        return 2
    if acc.get("mutated_negative_control"):
        print(
            "REFUSING TO BUILD: the acceptance receipt is from a --mutate run (negative control), "
            "not a clean one.",
            file=sys.stderr,
        )
        return 2

    seeds = available_seeds()
    units = {s: load_unit(s) for s in seeds}
    print(f"[units] {seeds} — identity verified (recomputed content hash == stored == manifest)")

    figures: list[str] = []
    requested = [s.strip() for s in args.symbols.split(",") if s.strip()]
    for sym in requested:
        try:
            stamps = pick_decisions(sym, args.per_symbol)
        except Exception as e:
            print(f"[skip]  {sym}: {e}")
            continue
        # NO SILENT DROPS. An empty stamp list is a real outcome (no valid decision instant found)
        # and must be announced, not inferred from a missing line — a build that quietly renders
        # fewer symbols than asked reads as "covered everything" when it did not.
        if not stamps:
            print(f"[skip]  {sym}: no valid decision instants found — NOT rendered")
            continue
        prep = prepare_symbol(sym, units=units)  # ~29 s per seed, once per symbol
        for ms in stamps:
            b = forecast_at(sym, ms, units=units, prepared=prep)
            svg = render_svg(b)
            mus = ", ".join(f"seed {sf.seed} &#956;&#770;={sf.mu_hat:+.5f}" for sf in b.seeds)
            figures.append(
                f"<figure>{svg}<figcaption>{sym} &#183; decision "
                f"{datetime.fromtimestamp(ms / 1000, tz=UTC):%Y-%m-%d %H:%M UTC} &#183; {mus}"
                "</figcaption></figure>"
            )
            print(f"[fig]   {sym} @ {ms}")

    if not figures:
        print("REFUSING TO BUILD: no figures rendered", file=sys.stderr)
        return 2
    print(f"[figs]  {len(figures)} rendered of {args.per_symbol * len(requested)} requested")

    n_par = units[seeds[0]].n_params_realized
    n_mtp = units[seeds[0]].n_params_mtp
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Trikaal &#183; {DEMO_H}-minute K-line forecasts (cell 1 baseline)</title>
<style>{PAGE_CSS}</style></head><body><main>
<h1>Trikaal &#8212; {DEMO_H}-minute forecasts from a microstructure-aware tokenizer study</h1>
<p class="lede">A decoder-only transformer over FSQ/BSQ-tokenised crypto 1-minute K-lines.
{n_par:,} realized parameters ({n_mtp:,} of them in the MTP heads). This page shows
<strong>forecasts</strong> produced by the exact weights the paper reports, through the exact
code path the paper scored.</p>

<div class="banner">
<b>THIS IS A FORECAST DEMO, NOT A TRADING SYSTEM &#8212; AND WE MEASURED IT LOSING MONEY.</b>
<p>Net information ratio on these weights: <code>-28.47</code>, <code>-67.02</code>,
<code>-146.49</code> for seeds 0, 2 and 4, across the entire pre-registered 0.10&#8211;0.30%
cost band. The break-even round-trip cost is <code>0.0023&#8211;0.0208%</code> against a
realistic <code>~0.10%</code> taker round trip &#8212; between 5&#215; and 43&#215; too small.
No profit, position or equity curve is computed anywhere in this demo, by design.</p>
</div>

<p>Every chart shows <strong>three seeds plotted separately and never averaged</strong>. Identical
data and configuration produced headline information ratios spanning 118.03 and activity spanning
24.4&#215;; that spread is one of the study&#8217;s findings, not noise to be smoothed away.</p>

<p>These are <strong>cell 1</strong> weights: BSQ quantizer, OHLCV-only input &#8212; the
<em>baseline</em> arm. They are not the microstructure contribution. The micro-legibility gate
refused cells 4 and 5 before Stage-2 on real data, so those arms have no scored artifacts at all;
that refusal is the paper&#8217;s primary finding.</p>

{"".join(figures)}

<footer>
Forecasts formed only from bars at or before the decision bar (causal safety is a CI-enforced
invariant). Checkpoint identity verified by recomputing the content hash and requiring
agreement with both the stored hash and the run manifest. The demo&#8217;s inference path is
proven <strong>bit-for-bit identical</strong> to the scored production path
(<code>scripts/m6_demo_acceptance.py</code>, {acc["n_decisions"]} decisions &#215;
{len(acc["seeds"])} seeds, float64 exact, no tolerance). Not investment advice.
</footer>
</main></body></html>"""

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    kb = len(page.encode()) / 1024
    print(f"WROTE {display_path(args.out, REPO)}  ({len(figures)} figures, {kb:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
