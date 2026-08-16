"""Local CSV forecast dashboard — stdlib http.server, binds to 127.0.0.1, fully offline.

    python scripts/m6_csv_dashboard.py        # then open http://127.0.0.1:8765

★ NOTHING LEAVES THE MACHINE. Loopback-only bind, no outbound request anywhere in the process,
and the uploaded CSV is parsed in memory and dropped — never written to disk. A user's price data
is theirs, and "there is no code path that could send it" is a stronger statement than a privacy
promise in a footer.

★ NO NEW DEPENDENCY. ``http.server`` + hand-written SVG, the same choice as the demo and for the
same reason: the disclosures must live INSIDE the figure. A crop that removes the
out-of-distribution banner removes the chart with it.

★ WHY A PATH AND A BAND, NOT DECODED CANDLES. Decoding predicted tokens back to full bar features
and drawing future OHLC bodies would look far more impressive and would be dishonest. Our own
measured OHLCV reconstruction MAE for this cell is 0.3470 in z-units (runs_cloud/results/r2), so
the drawn high/low/open would be dominated by reconstruction error while implying the model
predicts them. The model emits ONE scalar per horizon — an expected log-return — and the figure
shows exactly that and nothing more.

★ NO P&L, structurally. No profit, position, equity curve or Sharpe is computed here;
``tests/demo/test_no_pnl.py`` parses this package's AST and fails if one appears.
"""

from __future__ import annotations

import argparse
import html
import http.server
import socketserver
import sys
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from trikaal.demo.csv_forecast import (  # noqa: E402
    MIN_ROWS,
    PAPER_HORIZON,
    TRAINED_HORIZONS,
    CsvRefused,
    forecast_csv_export,
    forecast_from_csv,
    plain_english,
)
from trikaal.demo.inference import available_seeds, load_unit  # noqa: E402

SEED_COLOURS = {0: "#2563eb", 2: "#d97706", 4: "#059669"}
_UNITS: dict = {}
_LAST_EXPORT = {"csv": ""}

_SVG_CSS = (
    ".bg{fill:#fff}.fr{fill:none;stroke:#cbd5e1}"
    ".hs{fill:none;stroke:#0f172a;stroke-width:1.6}"
    ".ti{font:600 16px system-ui,sans-serif;fill:#0f172a}"
    ".sb{font:12px system-ui,sans-serif;fill:#475569}"
    ".ax{font:10px system-ui,sans-serif;fill:#64748b}"
    ".lg{font:500 12px system-ui,sans-serif}"
    "@media(prefers-color-scheme:dark){.bg{fill:#0b1220}.hs{stroke:#e2e8f0}"
    ".ti{fill:#f1f5f9}.sb{fill:#cbd5e1}.fr{stroke:#334155}.ax{fill:#94a3b8}}"
)


def _svg(f) -> str:
    """History solid, forecast dotted. Every qualifier is a text node inside this viewBox."""
    w, hgt = 1020, 620
    pl, pr, pt, pb = 66, 24, 96, 164
    pw, ph = w - pl - pr, hgt - pt - pb

    close = np.asarray(f.context_close, float)
    anchor = float(close[-1])
    hist = close / anchor
    fc = [f.per_seed[s]["price"] / anchor for s in sorted(f.per_seed)]
    blo = f.band["lo_price"] / anchor
    bhi = f.band["hi_price"] / anchor
    n = hist.size
    total = n + f.h
    ymin = min(hist.min(), min(fc), blo)
    ymax = max(hist.max(), max(fc), bhi)
    sp = max(ymax - ymin, 1e-9)
    ymin -= 0.08 * sp
    ymax += 0.08 * sp

    def xx(i: float) -> float:
        return pl + (i / max(total - 1, 1)) * pw

    def yy(v: float) -> float:
        return pt + ph - (v - ymin) / (ymax - ymin) * ph

    hist_pts = " ".join(f"{xx(i):.1f},{yy(v):.1f}" for i, v in enumerate(hist))
    x0, xf = xx(n - 1), xx(total - 1)
    parts = [
        f'<rect class="bg" width="{w}" height="{hgt}"/>',
        f'<rect class="fr" x="{pl}" y="{pt}" width="{pw}" height="{ph}"/>',
    ]
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = ymin + q * (ymax - ymin)
        parts.append(
            f'<text class="ax" x="{pl - 8}" y="{yy(v) + 3:.1f}" text-anchor="end">{v:.4f}</text>'
        )
    # ── SAMPLE-PERCENTILE BANDS + MEDIAN PATH, per seed. Never individual trajectories. ──────
    import math as _m

    for s in sorted(f.per_seed):
        c = SEED_COLOURS.get(s, "#7c3aed")
        qd = f.quantiles.get(s)
        if not qd:
            continue
        for lo_q, hi_q, op in ((10, 90, 0.09), (25, 75, 0.14)):
            up = [yy(_m.exp(qd[hi_q][k])) for k in range(f.h)]
            dn = [yy(_m.exp(qd[lo_q][k])) for k in range(f.h)]
            pts = " ".join(f"{xx(n - 1 + k + 1):.1f},{up[k]:.1f}" for k in range(f.h))
            pts += " " + " ".join(
                f"{xx(n - 1 + k + 1):.1f},{dn[k]:.1f}" for k in range(f.h - 1, -1, -1)
            )
            parts.append(
                f'<polygon points="{x0:.1f},{yy(1.0):.1f} {pts}" '
                f'fill="{c}" fill-opacity="{op}" stroke="none"/>'
            )
    parts.append(f'<polyline class="hs" points="{hist_pts}"/>')
    for k, s in enumerate(sorted(f.per_seed)):
        c = SEED_COLOURS.get(s, "#7c3aed")
        qd = f.quantiles.get(s)
        if qd:
            med = " ".join(
                f"{xx(n - 1 + j + 1):.1f},{yy(_m.exp(qd[50][j])):.1f}" for j in range(f.h)
            )
            parts.append(
                f'<polyline points="{x0:.1f},{yy(1.0):.1f} {med}" fill="none" '
                f'stroke="{c}" stroke-width="2.2"/>'
            )
        y = yy(f.per_seed[s]["price"] / anchor)
        parts.append(f'<circle cx="{xf:.1f}" cy="{y:.1f}" r="4.5" fill="{c}"/>')
        # MTP anchors — horizons actually trained; between them is interpolation
        for ha, mu in sorted(f.mtp_anchors.get(s, {}).items()):
            if 1 <= ha <= f.h:
                parts.append(
                    f'<rect x="{xx(n - 1 + ha) - 3:.1f}" y="{yy(_m.exp(mu)) - 3:.1f}" '
                    f'width="6" height="6" fill="none" stroke="{c}" stroke-width="1.4"/>'
                )
        lx = pl + k * 230
        parts.append(
            f'<circle cx="{lx + 5}" cy="{hgt - 112}" r="4.5" fill="{c}"/>'
            f'<text class="lg" x="{lx + 16}" y="{hgt - 108}" fill="{c}">'
            f"seed {s} {f.per_seed[s]['pct']:+.3f}%</text>"
        )
    parts.append(
        f'<line x1="{x0:.1f}" y1="{pt}" x2="{x0:.1f}" y2="{pt + ph}" '
        'stroke="#94a3b8" stroke-dasharray="2 3"/>'
        f'<text class="ax" x="{x0 + 5:.1f}" y="{pt + 12}">now</text>'
    )

    ood = f.inferred_bar_ms != 60_000
    box_fill = "#fef2f2" if ood else "#f8fafc"
    box_stroke = "#dc2626" if ood else "#94a3b8"
    head_fill = "#991b1b" if ood else "#334155"
    body_fill = "#7f1d1d" if ood else "#475569"
    tail = "  <- NOT 60s: OUT OF DISTRIBUTION" if ood else "  (matches training)"
    warn_lines = [
        "READ THE BAND, NOT THE WIGGLE. THIS IS NOT A PREDICTION OF THE PRICE.",
        "Bands are SAMPLE PERCENTILES of the model's own sampled paths, NOT confidence "
        "intervals. Our own measurement: mu-hat is 25-446x OVER-DISPERSED vs the returns "
        "it forecasts, so this fan is far WIDER than reality.",
        "Lines are the MEDIAN of "
        f"{f.n_mc_samples} sampled trajectories per seed - not 'the prediction'. Squares = the "
        "4 TRAINED MTP horizons; between them is interpolation.",
        "Measured edge ~0.005-0.02% per trade and NET NEGATIVE after realistic fees. "
        "Per-step shape is quantized at ~0.55z on the worst-reconstructed channel.",
        f"TRAINED ON CRYPTO 1-MINUTE BARS. Inferred interval {f.inferred_bar_ms / 1000:g}s{tail}."
        "  No profit, position or P&L is computed. Not investment advice.",
    ]
    wy = hgt - 128
    parts.append(
        f'<rect x="{pl}" y="{wy}" width="{pw}" height="86" rx="5" '
        f'fill="{box_fill}" stroke="{box_stroke}"/>'
    )
    for i, txt in enumerate(warn_lines):
        weight = "700 11px" if i == 0 else "10.5px"
        fill = head_fill if i == 0 else body_fill
        parts.append(
            f'<text x="{pl + 10}" y="{wy + 16 + i * 14}" '
            f'style="font:{weight} system-ui,sans-serif;fill:{fill}">'
            f"{html.escape(txt)}</text>"
        )

    dt = datetime.fromtimestamp(f.decision_ts_ms / 1000, tz=UTC)
    stamp = dt.strftime("%Y-%m-%d %H:%M UTC")
    head = [
        f'<text class="ti" x="{pl}" y="30">Trikaal &#183; {f.h}-minute forecast '
        "&#183; cell 1 (BSQ, OHLCV-only &#8212; the BASELINE arm)</text>",
        f'<text class="sb" x="{pl}" y="50">Your CSV, {f.n_rows:,} rows. '
        f"Last bar {html.escape(stamp)}. Solid = your data; dotted = forecast; "
        "shaded = sample percentiles (10-90, 25-75).</text>",
        f'<text class="sb" x="{pl}" y="68">Three seeds, never averaged. The band is the '
        "SPREAD BETWEEN MODELS, not a calibrated confidence interval.</text>",
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {hgt}" '
        f'width="{w}" height="{hgt}" role="img" '
        f'aria-label="Trikaal {f.h}-minute forecast, three seeds, not a trading result">'
        f"<style>{_SVG_CSS}</style>" + "".join(parts) + "".join(head) + "</svg>"
    )


_PAGE_CSS = (
    "body{margin:0;font:15px/1.55 system-ui,sans-serif;background:#f8fafc;color:#0f172a}"
    "main{max-width:1040px;margin:0 auto;padding:24px 18px 60px}"
    "h1{font-size:22px;margin:0 0 4px}.l{color:#475569;margin:0 0 16px}"
    "form{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;"
    "margin-bottom:18px}"
    "label{font-weight:600;font-size:13px;display:block;margin-bottom:6px}"
    ".row{display:flex;gap:18px;align-items:flex-end;flex-wrap:wrap}"
    "button{background:#0f172a;color:#fff;border:0;border-radius:7px;padding:9px 18px;"
    "font-size:14px;cursor:pointer}"
    ".err{background:#fef2f2;border:1.5px solid #dc2626;color:#7f1d1d;border-radius:8px;"
    "padding:14px;margin:16px 0;white-space:pre-wrap}"
    ".read{background:#fff;border:1px solid #e2e8f0;border-left:4px solid #0f172a;"
    "border-radius:8px;padding:14px;margin:16px 0;font-size:15px}"
    "svg{width:100%;height:auto;background:#fff;border:1px solid #e2e8f0;border-radius:10px}"
    "a.dl{display:inline-block;margin-top:12px;font-size:14px}small{color:#64748b}"
    "@media(prefers-color-scheme:dark){body{background:#0b1220;color:#e2e8f0}"
    "form,.read{background:#111a2e;border-color:#1e293b}"
    "svg{background:transparent;border-color:#1e293b}}"
)


def _page(body: str, n_seeds: int) -> str:
    opts = []
    for h in TRAINED_HORIZONS:
        sel = " selected" if h == PAPER_HORIZON else ""
        note = (
            " (the only horizon the paper evaluates)"
            if h == PAPER_HORIZON
            else " &#8212; trained, unscored"
        )
        opts.append(f'<option value="{h}"{sel}>{h} min{note}</option>')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Trikaal &#183; CSV forecast</title>"
        f"<style>{_PAGE_CSS}</style></head><body><main>"
        "<h1>Trikaal &#8212; forecast from your own CSV</h1>"
        '<p class="l">Runs entirely on this machine. Your file is parsed in memory, never '
        "written to disk and never uploaded. Model: cell 1 (BSQ, OHLCV-only) &#8212; the "
        f"<b>baseline</b> arm, {n_seeds} seeds.</p>"
        '<form method="POST" enctype="multipart/form-data"><div class="row">'
        '<div style="flex:1;min-width:260px"><label>OHLCV CSV &#8212; needs timestamp, open, '
        "high, low, close (volume/amount optional)</label>"
        '<input type="file" name="csv" accept=".csv,text/csv" required></div>'
        f'<div><label>Horizon</label><select name="h">{"".join(opts)}</select></div>'
        '<div><button type="submit">Forecast</button></div></div>'
        f"<p><small>At least {MIN_ROWS:,} rows. The features self-normalize causally, so a "
        "fresh CSV starts cold &#8212; the first 720 bars are warm-up and the model reads a "
        "512-bar context.</small></p></form>"
        f"{body}</main></body></html>"
    )


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body: str, code: int = 200):
        out = _page(body, len(_UNITS)).encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        if self.path.startswith("/export.csv"):
            data = _LAST_EXPORT["csv"].encode()
            self.send_response(200 if data else 404)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=trikaal_forecast.csv")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._send("")

    def _read_form(self) -> tuple[str, int]:
        ctype = self.headers.get("Content-Type", "")
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        text, h = "", PAPER_HORIZON
        if "multipart/form-data" in ctype and "boundary=" in ctype:
            bnd = ("--" + ctype.split("boundary=")[1].strip().strip('"')).encode()
            for part in raw.split(bnd):
                if b"\r\n\r\n" not in part:
                    continue
                head, val = part.split(b"\r\n\r\n", 1)
                val = val.rstrip(b"\r\n-")
                if b'name="csv"' in head:
                    text = val.decode("utf-8", "replace")
                elif b'name="h"' in head:
                    try:
                        h = int(val.decode().strip())
                    except ValueError:
                        h = PAPER_HORIZON
        else:
            q = urllib.parse.parse_qs(raw.decode("utf-8", "replace"))
            text = q.get("csv", [""])[0]
            h = int(q.get("h", [str(PAPER_HORIZON)])[0])
        return text, h

    def do_POST(self):
        text, h = self._read_form()
        try:
            f = forecast_from_csv(text, units=_UNITS, h=h)
        except CsvRefused as e:
            msg = html.escape(str(e))
            self._send(f'<div class="err"><b>Refused.</b>\n{msg}</div>', 400)
            return
        except Exception as e:  # a browser must never receive a raw traceback
            msg = html.escape(str(e))
            self._send(f'<div class="err"><b>Could not process that file.</b>\n{msg}</div>', 400)
            return

        _LAST_EXPORT["csv"] = forecast_csv_export(f)
        warn = "".join(f'<div class="err">{html.escape(w)}</div>' for w in f.warnings)
        body = (
            warn
            + f'<div class="read">{html.escape(plain_english(f))}</div>'
            + _svg(f)
            + '<a class="dl" href="/export.csv">Download the forecast as CSV</a>'
        )
        self._send(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    seeds = available_seeds()
    if not seeds:
        print("no banked units found — nothing to serve", file=sys.stderr)
        return 1
    for s in seeds:
        _UNITS[s] = load_unit(s, device=args.device)
    print(f"[units] {seeds} loaded; identity verified against each run manifest")
    print(f"[serve] http://127.0.0.1:{args.port} — LOOPBACK ONLY, nothing leaves this machine")
    print(f"[input] OHLCV CSV, >= {MIN_ROWS:,} rows; horizons {TRAINED_HORIZONS}")
    with socketserver.TCPServer(("127.0.0.1", args.port), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[stop]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
