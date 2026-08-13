"""SVG rendering — every disclosure lives INSIDE the image, by construction.

WHY HAND-WRITTEN SVG AND NOT A PLOTTING LIBRARY. Two reasons, and the first is the binding one:

1. THE CONTEXT-STRIPPING RULE. A caption beside a chart is not protection — figures get cropped,
   screenshotted into slides, and pasted without their surrounding section. Here the title, the
   arm label, the horizon, the seed identities AND the project's headline economic result are SVG
   text nodes inside the same viewBox as the curve. A crop that removes them removes the chart.
   There is no "caption" to lose because there is no caption: it is all one image.

2. NO NEW DEPENDENCY. The project declares numpy + torch (+ data extras) and nothing else. A demo
   is infrastructure, not research surface, but adding matplotlib/PIL to ship one chart is a poor
   trade when the SVG is 200 lines and gives exact control over what is inside the raster.

★ THIS MODULE CANNOT DRAW A P&L. It accepts a ForecastBundle, which carries no return, no
position and no equity series — those are never computed anywhere in ``trikaal.demo``. There is no
code path from this function to a profit curve.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from trikaal.demo.inference import ForecastBundle

# ── THE HEADLINE ECONOMIC RESULT, RENDERED INTO EVERY IMAGE ───────────────────────────────────
# Not a disclaimer bolted on: this is the project's measured finding and it is the single most
# important thing a viewer of a forecast chart could be missing. Sources:
#   net IR       — runs_cloud/results/r{0,1,2}, the three banked cell-1 money units
#   break-even   — runs_manifest/m6_horizon_break_even.json
# If these numbers ever change, they change HERE and in the receipt, and the image changes with
# them. A stale number rendered into a picture is worse than no number.
HEADLINE_NET_IR = (-28.47, -67.02, -146.49)
BREAK_EVEN_PCT_RANGE = (0.0023, 0.0208)
ROUND_TRIP_FEE_PCT = 0.10

SEED_COLOURS = {0: "#2563eb", 2: "#d97706", 4: "#059669"}

_CSS = """
  .bg{fill:#ffffff}
  .frame{fill:none;stroke:#cbd5e1;stroke-width:1}
  .ctx{fill:none;stroke:#0f172a;stroke-width:1.4}
  .real{fill:none;stroke:#64748b;stroke-width:1.4;stroke-dasharray:4 3}
  .title{font:600 17px system-ui,sans-serif;fill:#0f172a}
  .sub{font:400 12px system-ui,sans-serif;fill:#475569}
  .axis{font:400 10px system-ui,sans-serif;fill:#64748b}
  .legend{font:500 11px system-ui,sans-serif}
  .warnbox{fill:#fef2f2;stroke:#dc2626;stroke-width:1.2}
  .warnhead{font:700 11px system-ui,sans-serif;fill:#991b1b}
  .warn{font:400 10.5px system-ui,sans-serif;fill:#7f1d1d}
  .foot{font:400 9.5px ui-monospace,SFMono-Regular,Menlo,monospace;fill:#94a3b8}
  @media (prefers-color-scheme: dark){
    .bg{fill:#0b1220}.frame{stroke:#334155}.ctx{stroke:#e2e8f0}.real{stroke:#94a3b8}
    .title{fill:#f1f5f9}.sub{fill:#cbd5e1}.axis{fill:#94a3b8}
    .warnbox{fill:#2a1215;stroke:#f87171}.warnhead{fill:#fecaca}.warn{fill:#fca5a5}
    .foot{fill:#64748b}
  }
"""


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_svg(b: ForecastBundle, *, width: int = 940, height: int = 560) -> str:
    """One self-contained SVG. Every qualifier is a text node inside this viewBox."""
    pad_l, pad_r, pad_t, pad_b = 62, 22, 92, 150
    pw, ph = width - pad_l - pad_r, height - pad_t - pad_b

    ctx = np.asarray(b.context_close, dtype=np.float64)
    real = np.asarray(b.realized_close, dtype=np.float64)
    anchor = float(ctx[-1])
    # Normalize so the decision bar is 1.0 — the y-axis is RELATIVE, never a currency amount.
    ctx_n, real_n = ctx / anchor, real / anchor
    fcast = np.array([sf.forecast_close / anchor for sf in b.seeds], dtype=np.float64)

    n_ctx, n_real = ctx_n.size, real_n.size
    total_x = n_ctx + max(n_real - 1, b.h)
    ymin = float(min(ctx_n.min(), real_n.min(), fcast.min()))
    ymax = float(max(ctx_n.max(), real_n.max(), fcast.max()))
    span = max(ymax - ymin, 1e-9)
    ymin, ymax = ymin - 0.06 * span, ymax + 0.06 * span

    def X(i: float) -> float:
        return pad_l + (i / max(total_x - 1, 1)) * pw

    def Y(v: float) -> float:
        return pad_t + ph - (v - ymin) / (ymax - ymin) * ph

    ctx_pts = " ".join(f"{X(i):.2f},{Y(v):.2f}" for i, v in enumerate(ctx_n))
    real_pts = " ".join(f"{X(n_ctx - 1 + i):.2f},{Y(v):.2f}" for i, v in enumerate(real_n))

    seed_marks, legend = [], []
    x_f = X(n_ctx - 1 + b.h)
    for k, sf in enumerate(b.seeds):
        c = SEED_COLOURS.get(sf.seed, "#7c3aed")
        y = Y(sf.forecast_close / anchor)
        seed_marks.append(
            f'<line x1="{X(n_ctx - 1):.2f}" y1="{Y(1.0):.2f}" x2="{x_f:.2f}" y2="{y:.2f}" '
            f'stroke="{c}" stroke-width="1.8" stroke-dasharray="5 3"/>'
            f'<circle cx="{x_f:.2f}" cy="{y:.2f}" r="4.5" fill="{c}"/>'
        )
        lx = pad_l + k * 210
        legend.append(
            f'<circle cx="{lx + 5}" cy="{height - 128}" r="4.5" fill="{c}"/>'
            f'<text class="legend" x="{lx + 16}" y="{height - 124}" fill="{c}">'
            f"seed {sf.seed} &#183; &#956;&#770; {sf.mu_hat:+.5f}</text>"
        )

    # y ticks
    ticks = []
    for f in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = ymin + f * (ymax - ymin)
        ticks.append(
            f'<text class="axis" x="{pad_l - 8}" y="{Y(v) + 3:.2f}" text-anchor="end">'
            f"{v:.4f}</text>"
        )

    dt = datetime.fromtimestamp(b.decision_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    ir = " / ".join(f"{v:.2f}" for v in HEADLINE_NET_IR)
    nparam = b.meta.get("n_params_realized", 0)

    warn_y = height - 108
    warn = (
        f'<rect class="warnbox" x="{pad_l}" y="{warn_y}" width="{pw}" height="66" rx="5"/>'
        f'<text class="warnhead" x="{pad_l + 10}" y="{warn_y + 17}">'
        "THIS IS A FORECAST, NOT A TRADING RESULT. THIS MODEL LOSES MONEY AFTER FEES."
        "</text>"
        f'<text class="warn" x="{pad_l + 10}" y="{warn_y + 33}">'
        f"Measured net information ratio on these exact weights: {_esc(ir)} "
        "(seeds 0/2/4) across the whole 0.10-0.30% cost band."
        "</text>"
        f'<text class="warn" x="{pad_l + 10}" y="{warn_y + 47}">'
        f"Break-even round-trip cost {BREAK_EVEN_PCT_RANGE[0]:.4f}-{BREAK_EVEN_PCT_RANGE[1]:.4f}% "
        f"vs a realistic ~{ROUND_TRIP_FEE_PCT:.2f}% taker round trip: 5-43x too small."
        "</text>"
        f'<text class="warn" x="{pad_l + 10}" y="{warn_y + 60}">'
        "No profit, position or equity curve is computed anywhere in this demo. "
        "Not investment advice."
        "</text>"
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" \
width="{width}" height="{height}" role="img" \
aria-label="Trikaal forecast for {_esc(b.symbol)}, three seeds, not a trading result">
<style>{_CSS}</style>
<rect class="bg" x="0" y="0" width="{width}" height="{height}"/>
<text class="title" x="{pad_l}" y="30">Trikaal &#183; {_esc(b.symbol)} &#183; \
{b.h}-minute forecast &#183; cell 1 (BSQ, OHLCV-only &#8212; the BASELINE arm)</text>
<text class="sub" x="{pad_l}" y="50">Tokenizer study, not a foundation model. \
{nparam:,}-parameter measurement vehicle. Decision bar {_esc(dt)}; \
context = {n_ctx} one-minute bars.</text>
<text class="sub" x="{pad_l}" y="68">Three seeds shown separately and never averaged \
&#8212; the spread is one of the paper&#8217;s findings, not noise to smooth.</text>
<rect class="frame" x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}"/>
{"".join(ticks)}
<polyline class="ctx" points="{ctx_pts}"/>
<polyline class="real" points="{real_pts}"/>
{"".join(seed_marks)}
<line x1="{X(n_ctx - 1):.2f}" y1="{pad_t}" x2="{X(n_ctx - 1):.2f}" y2="{pad_t + ph}" \
stroke="#94a3b8" stroke-width="1" stroke-dasharray="2 3"/>
<text class="axis" x="{X(n_ctx - 1) + 5:.2f}" y="{pad_t + 12}">decision bar</text>
<text class="axis" x="{pad_l}" y="{pad_t - 8}">relative price (decision bar = 1.0000) \
&#8212; dashed grey = what actually happened</text>
{"".join(legend)}
{warn}
<text class="foot" x="{pad_l}" y="{height - 14}">predictor \
{_esc(b.seeds[0].predictor_hash[:26])}&#8230; &#183; estimator \
{_esc(b.meta.get("estimator", "?"))} &#183; forecast formed from bars \
&#8804; the decision bar only</text>
</svg>"""
