"""Local CSV forecast dashboard — stdlib http.server, binds to 127.0.0.1, fully offline.

    python scripts/m6_csv_dashboard.py        # then open http://127.0.0.1:8765

★ NOTHING LEAVES THE MACHINE, AND IT IS ENFORCED THREE WAYS RATHER THAN PROMISED ONCE.
(1) The socket binds to 127.0.0.1 only. (2) No module in this process makes an outbound request,
and the uploaded CSV is parsed in memory and dropped — never written to disk. (3) Every response
carries a Content-Security-Policy of ``default-src 'none' … connect-src 'self'``, so a future edit
that adds a CDN font, an analytics beacon or a remote image is BLOCKED BY THE BROWSER rather than
shipped quietly. The third one exists because the reference design this page is styled after loads
Poppins and JetBrains Mono from fonts.googleapis.com, and a webfont request would tell Google that
the user opened this tool — which would make the sentence we render on the page FALSE. The design
is worth taking; that line of it is not. A system font stack carries it.

★ NO NEW DEPENDENCY. ``http.server`` + hand-written SVG + hand-written CSS/JS. The reference is
written for a component framework (x-dc / DCLogic) that we do not ship: the DESIGN is extracted,
the CODE is not. Same reason as the demo — the disclosures must live INSIDE the figure, and a crop
that removes the out-of-distribution banner has to remove the chart with it.

★ WHAT THE REDESIGN FIXED, ALL THREE FOUND BY RENDERING THE OLD FIGURE AND LOOKING AT IT.
  1. THE SEED LEGEND WAS INVISIBLE. Legend marks sat at y=508 and the disclosure box spanned
     492-578 and was appended LATER, so it painted over them. The figure had shipped with no
     legend at all and the arithmetic says so deterministically — no font metrics involved.
  2. THE OVER-DISPERSION DISCLOSURE WAS CLIPPED OFF THE RIGHT EDGE. SVG ``<text>`` does not wrap;
     the 210-character line ran to x≈1178 in a 1020-wide viewBox and ended mid-word at
     "…vs the returns it forecast". The single most important caveat in the image was cut in half
     BY the image. It is now wrapped in a MONOSPACE face at a computed character budget, so the
     fit is arithmetic (advance is exactly 0.6em for any monospace) rather than eyeballed, and
     ``tests/demo/test_csv_dashboard.py`` asserts every line fits.
  3. THE FORECAST WAS 2.8% OF THE PLOT. 512 context bars against 15 forecast steps on one linear
     axis left the entire subject of the figure invisible in the last few pixels. The axis is now
     SPLIT — context compressed into 62% of the width, forecast expanded into 38% — with the break
     drawn, labelled, and named in the subtitle. A split axis is honest only if it is announced,
     so it is announced in the raster.

★ NO P&L, structurally. No profit, position, equity curve or Sharpe is computed here;
``tests/demo/test_no_pnl.py`` parses this file's AST — and the whole demo package's — and fails if
one appears.
"""

from __future__ import annotations

import argparse
import html
import http.server
import json
import secrets
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from trikaal.demo.csv_forecast import (  # noqa: E402
    MIN_ROWS,
    PAPER_HORIZON,
    SELECTABLE_HORIZONS,
    SEQ_LEN,
    WARMUP_ROWS,
    CsvRefused,
    forecast_csv_export,
    forecast_from_csv,
    plain_english,
)
from trikaal.demo.inference import (  # noqa: E402
    available_seeds,
    load_unit,
    reverify_identity,
)

# ─────────────────────────────────────────────────────────────────────────────────────────────
# FIGURE — dark-native, disclosures at full strength inside the raster
# ─────────────────────────────────────────────────────────────────────────────────────────────
# ★ THE SEED HUES ARE DELIBERATELY NOT RED/GREEN. Three seeds are three IDENTITIES, not three
# directions, and a green line rising is read as "up" by every person who has ever seen a trading
# screen. Blue / amber / violet cannot be misread as a direction, which leaves the page's single
# accent free to mean exactly one thing: live status.
SEED_COLOURS = {0: "#7aa2f7", 2: "#e0af68", 4: "#bb9af7"}

# Font stacks with NO quoted family names, so they drop straight into an SVG style attribute with
# nothing to escape — and, more to the point, with nothing to fetch. See the module docstring.
_SANS_CSS = "system-ui,-apple-system,Roboto,Helvetica,Arial,sans-serif"
_MONO_CSS = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"

FIG_BG = "#08090b"
FIG_FRAME = "#2a2f38"
FIG_GRID = "#171b21"
FIG_HIST = "#e6e9ee"
FIG_TITLE = "#ffffff"
FIG_SUB = "#aeb7c4"
FIG_AX = "#8b95a3"
DISC_BG = "#12161c"
DISC_BD = "#3d4653"
DISC_HEAD = "#ffffff"
DISC_BODY = "#dbe1e9"
OOD_BG = "#2b0f10"
OOD_BD = "#ef4444"
OOD_HEAD = "#ffe0e0"
OOD_BODY = "#fca5a5"

# ★ THE WRAP IS ARITHMETIC, NOT AESTHETIC. A monospace face advances a FIXED fraction of the em
# per character, so a character budget computed from the box width is a proof that the line fits —
# unlike a proportional face, where it is a guess that was wrong by 180px in the shipped figure.
#
# ★ 0.62, NOT 0.6, AND THE DIFFERENCE IS THE WHOLE GUARANTEE. "Monospace advances 0.6em" is true
# of SF Mono and Menlo and it is what this started at. Measuring the RENDERED figure in a browser
# — getBBox() on all 33 text nodes — showed the widest line at 983.2px inside a 984px box: 0.8px
# of headroom, i.e. the arithmetic was exactly right and exactly at the limit. But the stack ends
# in generic ``monospace``, which on Linux resolves to DejaVu Sans Mono at 0.6023em; that line
# becomes ~987px and clips by 3px on a machine none of us rendered on. A bound that holds only on
# the author's font is not a bound. 0.62 covers every face in the stack with room, and costs at
# most one extra wrapped line.
MONO_ADVANCE = 0.62
DISC_PX = 11.5
DISC_HEAD_PX = 12.5
DISC_LINE_H = 16.0
DISC_PAD = 14.0

# The history/forecast split. 512 context bars and 15 forecast steps do not share a linear axis
# usefully; this gives the forecast 38% of the width and the break is drawn and labelled.
HIST_FRAC = 0.62


def _wrap(text: str, max_chars: int) -> list[str]:
    """Greedy word wrap. Long unbreakable words are hard-split rather than allowed to overflow."""
    out: list[str] = []
    line = ""
    for word in text.split():
        while len(word) > max_chars:
            if line:
                out.append(line)
                line = ""
            out.append(word[:max_chars])
            word = word[max_chars:]
        if not line:
            line = word
        elif len(line) + 1 + len(word) <= max_chars:
            line += " " + word
        else:
            out.append(line)
            line = word
    if line:
        out.append(line)
    return out


def disclosure_lines(f) -> list[str]:
    """The qualifiers that must survive a crop. Kept at full length and full contrast.

    Every one of these is a MEASURED result of this project, not a legal hedge, which is why none
    of them is allowed to shrink into the elegant grey the palette has a slot for.
    """
    ood = f.inferred_bar_ms != 60_000
    tail = "  <- NOT 60s: OUT OF DISTRIBUTION" if ood else "  (matches training)"
    n_anchor = len(f.mtp_anchors.get(next(iter(f.mtp_anchors), 0), {}))
    return [
        "READ THE BAND, NOT THE WIGGLE. THIS IS NOT A PREDICTION OF THE PRICE.",
        "Bands are SAMPLE PERCENTILES of the model's own sampled paths, NOT confidence intervals. "
        "Our own measurement: mu-hat is 25-446x OVER-DISPERSED versus the returns it forecasts, "
        "so this fan is far WIDER than reality.",
        f"Lines are the MEDIAN of {f.n_mc_samples} sampled trajectories per seed, not 'the "
        f"prediction'. Squares mark the {n_anchor} trained MTP horizons IN RANGE (of 3 usable: "
        "5/15/60 - h=1 is structurally zero and is not offered); between them is interpolation.",
        "Measured edge ~0.005-0.02% per trade and NET NEGATIVE after a realistic ~0.10% round "
        "trip. Per-step shape is quantized at ~0.55z on the WORST-reconstructed channel.",
        f"TRAINED ON CRYPTO 1-MINUTE BARS. Inferred interval {f.inferred_bar_ms / 1000:g}s{tail}. "
        "Three seeds, NEVER averaged. No profit, position or P&L is computed anywhere in this "
        "app. Not investment advice.",
    ]


def _svg(f) -> str:
    """History solid, forecast expanded past a labelled break. Every qualifier is a text node."""
    import math as _m

    w, pl, pr, pt, plot_h = 1120, 78, 30, 122, 360
    pw = w - pl - pr

    # ── disclosure block sized FIRST, so the canvas grows to fit the text instead of clipping it
    max_chars = int((pw - 2 * DISC_PAD) / (DISC_PX * MONO_ADVANCE))
    head_chars = int((pw - 2 * DISC_PAD) / (DISC_HEAD_PX * MONO_ADVANCE))
    raw = disclosure_lines(f)
    wrapped: list[tuple[bool, str]] = [(True, ln) for ln in _wrap(raw[0], head_chars)]
    for para in raw[1:]:
        wrapped += [(False, ln) for ln in _wrap(para, max_chars)]

    # ★ THE INSET IS NOT A ZOOM FOR PRETTINESS, AND IT IS NOT ALLOWED TO REPLACE THE MAIN PANEL.
    # On a shared y-axis the whole 15-minute fan renders as a FLAT LINE, because it is ~0.1%
    # against context bars that move 2-3%. That flatness is the truest thing this figure says and
    # it stays. But it also makes the distribution — the entire point of sampling trajectories —
    # invisible, so the same numbers are drawn again below at their own scale, with their own axis
    # LABELLED IN PERCENT and a caption naming what changed. Both readings are present; neither is
    # substituted for the other. A cropped inset still carries its own axis values and its own
    # caption, which is what the context-stripping rule requires of a panel that can travel alone.
    inset_y = pt + plot_h + 40
    inset_h = 128.0
    inset_x = pl + 240.0
    inset_w = pw - 240.0
    box_y = inset_y + inset_h + 26
    box_h = 2 * DISC_PAD + len(wrapped) * DISC_LINE_H
    hgt = int(box_y + box_h + 26)

    close = np.asarray(f.context_close, float)
    anchor = float(close[-1])
    hist = close / anchor
    n = hist.size
    med = {s: [_m.exp(f.quantiles[s][50][k]) for k in range(f.h)] for s in f.quantiles}
    ymin, ymax = float(hist.min()), float(hist.max())
    for qd in f.quantiles.values():
        for q in (10, 90):
            for k in range(f.h):
                v = _m.exp(qd[q][k])
                ymin, ymax = min(ymin, v), max(ymax, v)
    for s in sorted(f.per_seed):
        v = f.per_seed[s]["price"] / anchor
        ymin, ymax = min(ymin, v), max(ymax, v)
    sp = max(ymax - ymin, 1e-9)
    ymin -= 0.10 * sp
    ymax += 0.10 * sp

    x_split = pl + pw * HIST_FRAC

    def xh(i: float) -> float:  # history bar index -> x
        return pl + (i / max(n - 1, 1)) * (pw * HIST_FRAC)

    def xf(k: float) -> float:  # forecast step (0 = now) -> x
        return x_split + (k / max(f.h, 1)) * (pw * (1 - HIST_FRAC))

    def yy(v: float) -> float:
        return pt + plot_h - (v - ymin) / (ymax - ymin) * plot_h

    p: list[str] = [
        f'<rect width="{w}" height="{hgt}" fill="{FIG_BG}"/>',
        f'<rect x="{pl}" y="{pt}" width="{pw}" height="{plot_h}" fill="none" '
        f'stroke="{FIG_FRAME}"/>',
    ]
    # forecast region gets a faint ground so the split reads as intentional, not as an artifact
    p.append(
        f'<rect x="{x_split:.1f}" y="{pt}" width="{pl + pw - x_split:.1f}" height="{plot_h}" '
        f'fill="#0d1015"/>'
    )
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = ymin + q * (ymax - ymin)
        y = yy(v)
        p.append(
            f'<line x1="{pl}" y1="{y:.1f}" x2="{pl + pw}" y2="{y:.1f}" stroke="{FIG_GRID}"/>'
            f'<text x="{pl - 10}" y="{y + 3.5:.1f}" text-anchor="end" '
            f'style="font:10.5px {_MONO_CSS};fill:{FIG_AX}">{v:.4f}</text>'
        )

    # ── bands: sample percentiles of each seed's own sampled paths, drawn in the expanded region
    for s in sorted(f.per_seed):
        c = SEED_COLOURS.get(s, "#7c7f8a")
        qd = f.quantiles.get(s)
        if not qd:
            continue
        for lo_q, hi_q, op in ((10, 90, 0.13), (25, 75, 0.20)):
            up = " ".join(f"{xf(k + 1):.1f},{yy(_m.exp(qd[hi_q][k])):.1f}" for k in range(f.h))
            dn = " ".join(
                f"{xf(k + 1):.1f},{yy(_m.exp(qd[lo_q][k])):.1f}" for k in range(f.h - 1, -1, -1)
            )
            p.append(
                f'<polygon points="{x_split:.1f},{yy(1.0):.1f} {up} {dn}" '
                f'fill="{c}" fill-opacity="{op}" stroke="none"/>'
            )

    hist_pts = " ".join(f"{xh(i):.1f},{yy(v):.1f}" for i, v in enumerate(hist))
    p.append(f'<polyline points="{hist_pts}" fill="none" stroke="{FIG_HIST}" stroke-width="1.5"/>')

    for s in sorted(f.per_seed):
        c = SEED_COLOURS.get(s, "#7c7f8a")
        if s in med:
            pts = " ".join(f"{xf(k + 1):.1f},{yy(med[s][k]):.1f}" for k in range(f.h))
            p.append(
                f'<polyline points="{x_split:.1f},{yy(1.0):.1f} {pts}" fill="none" '
                f'stroke="{c}" stroke-width="2.4" stroke-dasharray="5 4"/>'
            )
        y = yy(f.per_seed[s]["price"] / anchor)
        p.append(f'<circle cx="{xf(f.h):.1f}" cy="{y:.1f}" r="5" fill="{c}"/>')
        for ha, mu in sorted(f.mtp_anchors.get(s, {}).items()):
            if 1 <= ha <= f.h:
                p.append(
                    f'<rect x="{xf(ha) - 4:.1f}" y="{yy(_m.exp(mu)) - 4:.1f}" width="8" '
                    f'height="8" fill="none" stroke="{c}" stroke-width="1.8"/>'
                )

    # ── the break, drawn and named. A split axis is honest only when it is announced.
    p.append(
        f'<line x1="{x_split:.1f}" y1="{pt}" x2="{x_split:.1f}" y2="{pt + plot_h}" '
        f'stroke="#5b6673" stroke-width="1.4"/>'
    )
    p.append(
        f'<text x="{x_split - 8:.1f}" y="{pt + plot_h + 20}" text-anchor="end" '
        f'style="font:10.5px {_MONO_CSS};fill:{FIG_AX}">'
        f"&#9664; {n} BARS OF CONTEXT, COMPRESSED</text>"
        f'<text x="{x_split + 8:.1f}" y="{pt + plot_h + 20}" '
        f'style="font:10.5px {_MONO_CSS};fill:{FIG_AX}">'
        f"NEXT {f.h} MIN, EXPANDED &#9654;</text>"
    )
    p.append(
        f'<text x="{x_split:.1f}" y="{pt - 8}" text-anchor="middle" '
        f'style="font:600 10.5px {_MONO_CSS};fill:{FIG_SUB}">NOW</text>'
    )

    # ── legend, in its OWN band. It used to be drawn under the disclosure box and never seen.
    for k, s in enumerate(sorted(f.per_seed)):
        c = SEED_COLOURS.get(s, "#7c7f8a")
        ly = inset_y + 20 + k * 26
        p.append(
            f'<circle cx="{pl + 6}" cy="{ly - 4}" r="5" fill="{c}"/>'
            f'<text x="{pl + 20}" y="{ly}" style="font:600 13px {_SANS_CSS};fill:{c}">'
            f"seed {s}</text>"
            f'<text x="{pl + 86}" y="{ly}" style="font:13px {_MONO_CSS};fill:{FIG_SUB}">'
            f"{f.per_seed[s]['pct']:+.3f}%</text>"
        )

    # ── THE OWN-SCALE INSET. Same numbers, own y-axis, labelled in percent.
    pcts: list[float] = []
    for qd in f.quantiles.values():
        for q in (10, 90):
            pcts += [(_m.exp(qd[q][k]) - 1.0) * 100.0 for k in range(f.h)]
    pcts += [f.per_seed[s]["pct"] for s in f.per_seed] + [0.0]
    imin, imax = (min(pcts), max(pcts)) if pcts else (-0.01, 0.01)
    isp = max(imax - imin, 1e-9)
    imin, imax = imin - 0.12 * isp, imax + 0.12 * isp

    def ix(k: float) -> float:
        return inset_x + 62 + (k / max(f.h, 1)) * (inset_w - 74)

    def iy(v: float) -> float:
        # top of the data band sits BELOW the caption: the first version put the highest axis
        # label straight through the caption text, which is a legibility bug in the one panel
        # whose whole job is to be legible on its own.
        return inset_y + inset_h - 26 - (v - imin) / (imax - imin) * (inset_h - 62)

    p.append(
        f'<rect id="inset" x="{inset_x:.1f}" y="{inset_y:.1f}" width="{inset_w:.1f}" '
        f'height="{inset_h:.1f}" rx="6" fill="#0d1015" stroke="{FIG_FRAME}"/>'
        f'<text x="{inset_x + 12:.1f}" y="{inset_y + 15:.1f}" '
        f'style="font:700 9.5px {_MONO_CSS};fill:{FIG_SUB}">'
        "THE SAME FORECAST AT ITS OWN Y-SCALE (%). THE PANEL ABOVE SHOWS ITS TRUE SIZE AGAINST "
        "YOUR BARS.</text>"
    )
    for v in (imin, (imin + imax) / 2, imax):
        p.append(
            f'<line x1="{inset_x + 62:.1f}" y1="{iy(v):.1f}" x2="{inset_x + inset_w - 12:.1f}" '
            f'y2="{iy(v):.1f}" stroke="{FIG_GRID}"/>'
            f'<text x="{inset_x + 56:.1f}" y="{iy(v) + 3.5:.1f}" text-anchor="end" '
            f'style="font:9.5px {_MONO_CSS};fill:{FIG_AX}">{v:+.3f}%</text>'
        )
    p.append(
        f'<line x1="{ix(0):.1f}" y1="{iy(imax):.1f}" x2="{ix(0):.1f}" y2="{iy(imin):.1f}" '
        f'stroke="#5b6673"/>'
    )
    for s in sorted(f.per_seed):
        c = SEED_COLOURS.get(s, "#7c7f8a")
        qd = f.quantiles.get(s)
        if qd:
            for lo_q, hi_q, op in ((10, 90, 0.15), (25, 75, 0.24)):
                up = " ".join(
                    f"{ix(k + 1):.1f},{iy((_m.exp(qd[hi_q][k]) - 1) * 100):.1f}" for k in range(f.h)
                )
                dn = " ".join(
                    f"{ix(k + 1):.1f},{iy((_m.exp(qd[lo_q][k]) - 1) * 100):.1f}"
                    for k in range(f.h - 1, -1, -1)
                )
                p.append(
                    f'<polygon points="{ix(0):.1f},{iy(0.0):.1f} {up} {dn}" fill="{c}" '
                    f'fill-opacity="{op}" stroke="none"/>'
                )
            mid = " ".join(
                f"{ix(k + 1):.1f},{iy((_m.exp(qd[50][k]) - 1) * 100):.1f}" for k in range(f.h)
            )
            p.append(
                f'<polyline points="{ix(0):.1f},{iy(0.0):.1f} {mid}" fill="none" stroke="{c}" '
                f'stroke-width="2" stroke-dasharray="5 4"/>'
            )
        p.append(
            f'<circle cx="{ix(f.h):.1f}" cy="{iy(f.per_seed[s]["pct"]):.1f}" r="4" fill="{c}"/>'
        )
    p.append(
        f'<text x="{ix(0):.1f}" y="{inset_y + inset_h - 8:.1f}" '
        f'style="font:9.5px {_MONO_CSS};fill:{FIG_AX}">now</text>'
        f'<text x="{ix(f.h):.1f}" y="{inset_y + inset_h - 8:.1f}" text-anchor="end" '
        f'style="font:9.5px {_MONO_CSS};fill:{FIG_AX}">+{f.h} min</text>'
    )

    ood = f.inferred_bar_ms != 60_000
    p.append(
        f'<rect id="disclosure" x="{pl}" y="{box_y:.1f}" width="{pw}" height="{box_h:.1f}" rx="6" '
        f'fill="{OOD_BG if ood else DISC_BG}" stroke="{OOD_BD if ood else DISC_BD}" '
        f'stroke-width="{2 if ood else 1}"/>'
    )
    for i, (is_head, txt) in enumerate(wrapped):
        y = box_y + DISC_PAD + DISC_LINE_H * (i + 0.78)
        style = (
            f"font:700 {DISC_HEAD_PX}px {_MONO_CSS};fill:{OOD_HEAD if ood else DISC_HEAD}"
            if is_head
            else f"font:{DISC_PX}px {_MONO_CSS};fill:{OOD_BODY if ood else DISC_BODY}"
        )
        p.append(
            f'<text x="{pl + DISC_PAD:.1f}" y="{y:.1f}" style="{style}">{html.escape(txt)}</text>'
        )

    dt = datetime.fromtimestamp(f.decision_ts_ms / 1000, tz=UTC)
    stamp = dt.strftime("%Y-%m-%d %H:%M UTC")
    p.append(
        # ★ THESE LINES ARE KEPT SHORT ON PURPOSE. SVG text does not wrap, and the shipped figure
        # lost half of its most important caveat to exactly this. Proportional advance cannot be
        # computed the way monospace can, so the sans lines are budgeted at a conservative 0.58em
        # and the fit is ALSO checked visually — the arithmetic alone is not a proof here.
        f'<text x="{pl}" y="38" style="font:600 19px {_SANS_CSS};fill:{FIG_TITLE}">'
        f"Trikaal &#183; {f.h}-minute forecast &#183; cell 1 (BSQ, OHLCV-only &#8212; the "
        "BASELINE arm)</text>"
        f'<text x="{pl}" y="61" style="font:13px {_SANS_CSS};fill:{FIG_SUB}">'
        f"Your CSV, {f.n_rows:,} rows. Last bar {html.escape(stamp)}.</text>"
        f'<text x="{pl}" y="81" style="font:13px {_SANS_CSS};fill:{FIG_SUB}">'
        "Solid = your data; dashed = each seed&#8217;s median path; "
        "shaded = sample percentiles (10-90, 25-75).</text>"
        f'<text x="{pl}" y="101" style="font:13px {_SANS_CSS};fill:{FIG_SUB}">'
        "Three seeds, never averaged. y-axis = price &#247; last close. "
        "THE X-AXIS IS SPLIT AT &#8220;NOW&#8221; &#8212; the two sides are NOT one scale.</text>"
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {hgt}" '
        f'width="{w}" height="{hgt}" role="img" '
        f'aria-label="Trikaal {f.h}-minute forecast, three seeds, not a trading result">'
        + "".join(p)
        + "</svg>"
    )


# ─────────────────────────────────────────────────────────────────────────────────────────────
# JOB STORE — in memory, session only, RESULTS ONLY
# ─────────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Job:
    """One run. ★ THERE IS DELIBERATELY NO FIELD THAT COULD HOLD THE UPLOADED BARS.

    The page states that the file is parsed in memory and never written to disk, and history must
    not quietly break that. The uploaded text is a local variable in ``_execute`` and is unreachable
    the moment it returns; what survives is the FORECAST — numbers, the rendered figure, the
    filename and the shape of the input. ``tests/demo/test_csv_dashboard.py`` pins this field set,
    so adding a place to keep the bars fails the suite rather than passing review.
    """

    id: str
    filename: str
    h: int
    started: float
    stages: list[dict] = field(default_factory=list)
    done: bool = False
    error: str | None = None
    result: dict | None = None
    export: str = ""


BG_IMAGE = REPO / "assets" / "trikaal-bg-image.png"

_UNITS: dict = {}
_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_RUN_LOCK = threading.Lock()  # one forecast at a time; a local tool has one operator
MAX_UPLOAD = 64 * 1024 * 1024


def _emit(job: Job, stage: str, detail: str, state: str) -> None:
    """Append a stage, or complete the one that is running. Carries its own real elapsed time."""
    entry = {
        "stage": stage,
        "detail": detail,
        "state": state,
        "t": round(time.monotonic() - job.started, 2),
    }
    with _LOCK:
        last = job.stages[-1] if job.stages else None
        if last is not None and last["stage"] == stage and last["state"] == "run":
            job.stages[-1] = entry
        else:
            job.stages.append(entry)


def _execute(job: Job, text: str, device: str) -> None:
    """Run one forecast. ``text`` is local and is never stored anywhere."""
    try:
        # ★ WAITING FOR THE LOCK IS ITSELF A STAGE. One forecast runs at a time, so a second tab
        # would otherwise sit on a loading screen with a ticking clock and NO LINES — a blank
        # screen that is technically accurate and reads as a hang. A trace that goes silent while
        # something real is happening is a trace with a hole in it.
        if not _RUN_LOCK.acquire(blocking=False):
            _emit(
                job, "QUEUED", "another forecast holds the model; this one starts after it", "run"
            )
            _RUN_LOCK.acquire()
            _emit(job, "QUEUED", "the model is free; starting now", "ok")
        try:
            # ★ IDENTITY BEFORE INPUT. The instrument is verified before the file is even read:
            # both content hashes are RECOMPUTED from the live modules about to run and must agree
            # with the stored value AND the run manifest. load_unit checks the bytes it opened;
            # this checks the model that is serving. If it fails the run STOPS here.
            _emit(job, "VERIFYING MODEL IDENTITY", "", "run")
            checked = []
            for seed in sorted(_UNITS):
                v = reverify_identity(_UNITS[seed])
                if not v["ok"]:
                    raise RuntimeError(
                        f"seed {seed} IDENTITY FAILED — recomputed content hash does not match "
                        f"the stored value and the run manifest. REFUSING to forecast with a "
                        f"model that is not the scored one. {json.dumps(v['checks'])}"
                    )
                checked.append(str(seed))
            _emit(
                job,
                "VERIFYING MODEL IDENTITY",
                f"seeds {', '.join(checked)}: tokenizer+predictor content hash "
                "recomputed == stored == manifest",
                "ok",
            )

            f = forecast_from_csv(
                text,
                units=_UNITS,
                h=job.h,
                device=device,
                progress=lambda s, d, st: _emit(job, s, d, st),
            )

            _emit(job, "RENDERING", "", "run")
            svg = _svg(f)
            job.export = forecast_csv_export(f)
            _emit(
                job,
                "RENDERING",
                f"{len(svg):,} bytes of SVG, {len(disclosure_lines(f))} disclosure paragraphs "
                "inside the raster",
                "ok",
            )
            with _LOCK:
                job.result = {
                    "id": job.id,
                    "file": job.filename,
                    "h": f.h,
                    "rows": f.n_rows,
                    "interval_s": f.inferred_bar_ms / 1000,
                    "stamp": datetime.fromtimestamp(f.decision_ts_ms / 1000, tz=UTC).strftime(
                        "%Y-%m-%d %H:%M UTC"
                    ),
                    "read": plain_english(f),
                    "warnings": list(f.warnings),
                    "svg": svg,
                    "seeds": [{"seed": s, "pct": f.per_seed[s]["pct"]} for s in sorted(f.per_seed)],
                    "elapsed": round(time.monotonic() - job.started, 1),
                }
                job.done = True
        finally:
            _RUN_LOCK.release()
    except CsvRefused as e:
        _fail(job, "REFUSED", str(e))
    except Exception as e:  # a browser must never receive a raw traceback
        _fail(job, "FAILED", f"{type(e).__name__}: {e}")


def _fail(job: Job, kind: str, msg: str) -> None:
    with _LOCK:
        if job.stages and job.stages[-1]["state"] == "run":
            job.stages[-1] = {**job.stages[-1], "state": "fail", "detail": kind}
        else:
            job.stages.append(
                {
                    "stage": kind,
                    "detail": "",
                    "state": "fail",
                    "t": round(time.monotonic() - job.started, 2),
                }
            )
        job.error = msg
        job.done = True


# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE PAGE
# ─────────────────────────────────────────────────────────────────────────────────────────────
# ★ CSP: a promise the browser enforces. Nothing external can load even if someone adds it.
CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src 'self' data:; connect-src 'self'; form-action 'self'; base-uri 'none'; "
    "frame-ancestors 'none'"
)

_APP_CSS = """
:root{
  --bg:#000; --pane:rgba(9,9,9,0.86); --line:rgba(255,255,255,0.09);
  --t1:rgba(255,255,255,0.86); --t2:rgba(255,255,255,0.5);
  --t3:rgba(255,255,255,0.3); --t4:rgba(255,255,255,0.22);
  --ac:#4fe1a1;
  /* ★ CLASH DISPLAY IS REFERENCED BY FAMILY NAME AND IS NOT SHIPPED. The Fontshare EULA §02
     forbids "uploading them in a public server", and this repository is public — so the woff2
     files are deliberately NOT committed. The font renders from the operator's own installed
     copy (~/Library/Fonts, family "Clash Display"), which is a local lookup with no network
     request and nothing distributed. Anywhere it is absent the stack falls through to Helvetica,
     which is what the body text uses anyway. This is the same reasoning that kept the reference
     design's Google Fonts link out: the page must not fetch, and now also must not re-publish. */
  --d:'Clash Display','Clash Display Medium','Helvetica Neue',Helvetica,Arial,sans-serif;
  --f:'Helvetica Neue',Helvetica,Arial,sans-serif;
  /* ★ THE UI LABEL FACE IS HELVETICA, NOT MONOSPACE — and the variable is named --l, not
     --m, because a variable called "mono" holding a proportional face is the prose-vs-
     artifact drift this project keeps paying for. THE FIGURE'S MONO IS A DIFFERENT
     CONSTANT (_MONO_CSS) AND MUST STAY MONOSPACE: the disclosure wrap's fit is a PROOF
     that depends on a fixed 0.62em advance, and a proportional face turns that proof back
     into the guess that clipped the over-dispersion caveat off the raster. */
  --l:'Helvetica Neue',Helvetica,Arial,sans-serif;
  --sh:0 30px 80px rgba(0,0,0,0.6);
}
@supports (color: oklch(0.8 0.14 150)){ :root{ --ac: oklch(0.8 0.14 150); } }
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:#fff;font-family:var(--f)}
body{min-height:100vh;overflow-x:hidden}
::selection{background:rgba(255,255,255,0.22)}
::placeholder{color:rgba(255,255,255,0.28)}
.scr::-webkit-scrollbar{width:3px;height:3px}
.scr::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.16);border-radius:3px}
@keyframes rise{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0.15}}
.rise{animation:rise 260ms ease both}
.blink{animation:blink 1.4s ease-in-out infinite}

/* ── THE HERO IMAGE, AND ITS ONE JOB: GET OUT OF THE WAY WHEN THERE IS A RESULT ──────────────
   Served from our own handler at /bg.png, never from anywhere else. It carries the empty state
   and then fades to a trace, because a full-bleed art plate behind a forecast chart is exactly
   the "trading product" register the disclosures exist to resist. */
.bgwrap{position:fixed;inset:0;z-index:0;pointer-events:none;overflow:hidden}
.bgwrap img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;
  object-position:62% 50%;opacity:0.95;transition:opacity 900ms cubic-bezier(.4,0,.2,1)}
body[data-state="result"] .bgwrap img{opacity:0.05}
.bgwrap:after{content:'';position:absolute;inset:0;background:
  radial-gradient(120% 90% at 8% 92%, rgba(0,0,0,0.94) 0%, rgba(0,0,0,0.58) 38%,
    rgba(0,0,0,0) 68%),
  linear-gradient(180deg, rgba(0,0,0,0.74) 0%, rgba(0,0,0,0) 24%, rgba(0,0,0,0) 60%,
    rgba(0,0,0,0.7) 100%);}

/* ★ ONE FLAG REVEALS THE FIGURE AND THE DISCLOSURES TOGETHER, AND THAT IS DELIBERATE.
   The empty state is bare on purpose — there is no forecast yet, so there is nothing to qualify.
   The moment there IS one, `data-state="result"` shows `.right`, which contains the chart AND
   the "what this is not" panel. They are siblings under a single selector, so no edit can reveal
   a forecast while leaving its qualifiers hidden: to break that you would have to move one of
   them out of `.right`, which `tests/demo/test_csv_dashboard.py` refuses. */
body[data-state="empty"] .right{display:none}
body[data-state="empty"] .thread{display:none}

header{position:relative;z-index:6;height:76px;display:flex;align-items:center;
  justify-content:space-between;padding:0 26px 0 24px;gap:16px}
.mark{width:30px;height:30px;border-radius:9px;background:#fff;display:grid;place-items:center;
  flex:0 0 auto}
.mark i{position:relative;width:12px;height:12px;display:block}
.mark i:before,.mark i:after{content:'';position:absolute;background:#000}
.mark i:before{top:5px;left:0;width:12px;height:2px}
.mark i:after{left:5px;top:0;width:2px;height:12px}
.mono{font-family:var(--l)}
.lbl{font-family:var(--d);font-size:16px;font-weight:500;letter-spacing:0.14em;
  text-transform:uppercase;color:rgba(255,255,255,0.94)}
.sep{width:1px;height:14px;background:rgba(255,255,255,0.18)}
.sub{font-size:13px;color:var(--t2)}
.pill{display:flex;align-items:center;gap:9px;height:38px;padding:0 18px;border-radius:999px;
  border:1px solid rgba(255,255,255,0.12);background:rgba(10,10,10,0.6);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  font-family:var(--l);font-size:11px;letter-spacing:0.1em;color:rgba(255,255,255,0.7);
  white-space:nowrap}
.dot{width:6px;height:6px;border-radius:50%;background:var(--ac);
  box-shadow:0 0 10px var(--ac);flex:0 0 auto}

.wrap{position:relative;z-index:5;display:flex;gap:18px;padding:0 26px 26px 24px;
  align-items:stretch;min-height:calc(100vh - 92px)}
nav.rail{position:sticky;top:16px;align-self:center;display:flex;flex-direction:column;gap:6px;
  padding:8px;
  border-radius:999px;border:1px solid rgba(255,255,255,0.08);background:rgba(8,8,8,0.55);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);flex:0 0 auto}
nav.rail button{width:38px;height:38px;border:0;border-radius:12px;background:transparent;
  display:grid;place-items:center;cursor:pointer;transition:background 160ms ease;
  font-family:var(--l);font-size:14px;color:var(--t2)}
nav.rail button:hover{background:rgba(255,255,255,0.1);color:#fff}
nav.rail button.on{background:rgba(255,255,255,0.14);color:#fff}

.col{display:flex;flex-direction:column;gap:14px;min-width:0}
/* thread + composer sit at the BOTTOM LEFT, as in the reference; the figure fills the right */
.left{flex:0 0 600px;max-width:600px;justify-content:flex-end}
.right{flex:1 1 auto;min-width:0;justify-content:flex-start}
.glass{border-radius:26px;border:1px solid var(--line);background:var(--pane);
  backdrop-filter:blur(26px);-webkit-backdrop-filter:blur(26px);box-shadow:var(--sh)}
.k{font-family:var(--l);font-size:10px;letter-spacing:0.2em;color:var(--t3);
  text-transform:uppercase}

.thread{display:flex;flex-direction:column;gap:10px;max-height:52vh;overflow-y:auto;
  padding-right:6px}
.msg{max-width:520px;padding:14px 16px;border-radius:16px;backdrop-filter:blur(22px);
  -webkit-backdrop-filter:blur(22px);box-shadow:0 18px 50px rgba(0,0,0,0.45)}
.msg.you{align-self:flex-end;border:1px solid rgba(255,255,255,0.16);
  background:rgba(255,255,255,0.07)}
.msg.sys{align-self:flex-start;border:1px solid rgba(255,255,255,0.08);
  background:rgba(9,9,9,0.84);max-width:100%;width:100%;cursor:pointer}
.msg.sys.sel{border-color:rgba(255,255,255,0.28)}
.msg.bad{align-self:flex-start;width:100%;border:1.5px solid #ef4444;
  background:rgba(43,15,16,0.9)}
.mhead{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;gap:12px}
.mhead .who{font-family:var(--l);font-size:10px;letter-spacing:0.18em}
.mhead .st{font-family:var(--l);font-size:10px;letter-spacing:0.1em;color:var(--t4)}
.msg p{margin:0;font-size:14px;line-height:1.6;color:var(--t1)}
.msg pre{margin:0;font-family:var(--l);font-size:12px;line-height:1.65;color:#fecaca;
  white-space:pre-wrap;word-break:break-word}

.composer{position:relative;min-height:150px;padding:20px 24px 16px;display:flex;
  flex-direction:column;justify-content:space-between;transition:border-color 160ms ease}
.composer.drag{border-color:var(--ac)}
.drop{margin:12px 0 0;font-family:var(--d);font-size:34px;font-weight:500;line-height:1.1;
  letter-spacing:-0.01em;color:#fff}
.drop small{display:block;margin-top:9px;font-family:var(--f);font-size:12.5px;font-weight:400;
  color:var(--t2);letter-spacing:0}
.chips{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.chip{height:30px;padding:0 13px;border-radius:999px;border:1px solid rgba(255,255,255,0.1);
  background:rgba(255,255,255,0.03);cursor:pointer;font-family:var(--l);font-size:10px;
  letter-spacing:0.1em;color:var(--t2);transition:all 160ms ease}
.chip:hover{color:#fff;border-color:rgba(255,255,255,0.28)}
.chip.on{color:#000;background:var(--ac);border-color:var(--ac)}
.file{display:flex;align-items:center;gap:10px;height:34px;padding:0 12px;border-radius:11px;
  border:1px solid rgba(255,255,255,0.12);background:rgba(14,14,14,0.78);
  font-size:12px;color:rgba(255,255,255,0.72)}
.file span.tag{width:auto;padding:0 6px;height:18px;border-radius:5px;
  background:rgba(255,255,255,0.14);display:grid;place-items:center;font-family:var(--l);
  font-size:9px;color:rgba(255,255,255,0.7)}
.file button{border:0;background:transparent;cursor:pointer;font-family:var(--l);font-size:13px;
  color:var(--t3);padding:0}
.file button:hover{color:#fff}
/* the reference puts the action at the far bottom-right of the screen, not beside the composer */
.send{position:fixed;right:30px;bottom:30px;z-index:9;
  width:58px;height:58px;border-radius:50%;border:0;background:rgba(255,255,255,0.9);
  cursor:pointer;display:grid;place-items:center;box-shadow:0 18px 44px rgba(0,0,0,0.55);
  transition:transform 200ms ease,background 200ms ease,opacity 200ms ease;flex:0 0 auto}
.send:hover:not(:disabled){transform:translateY(-2px);background:#fff}
.send:disabled{opacity:0.28;cursor:not-allowed}
.foot{display:flex;align-items:flex-end;gap:10px}

.disc{padding:18px 20px}
.disc h3{margin:0 0 12px;font-family:var(--l);font-size:10px;letter-spacing:0.2em;
  color:var(--t3);text-transform:uppercase}
.disc ul{margin:0;padding:0;list-style:none;display:flex;flex-direction:column;gap:9px}
.disc li{font-size:12.5px;line-height:1.5;color:var(--t1);padding-left:14px;position:relative}
.disc li:before{content:'';position:absolute;left:0;top:7px;width:5px;height:5px;
  border-radius:1px;background:rgba(255,255,255,0.42)}
.disc b{color:#fff;font-weight:600}
.canvas{padding:14px}
.canvas svg{width:100%;height:auto;display:block;border-radius:12px}
.empty{padding:56px 24px;text-align:center;color:var(--t2);font-size:13px}
.dl{display:inline-flex;align-items:center;gap:8px;margin-top:12px;height:32px;padding:0 14px;
  border-radius:10px;border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.04);
  font-family:var(--l);font-size:10.5px;letter-spacing:0.1em;color:var(--t1);
  text-decoration:none}
.dl:hover{border-color:rgba(255,255,255,0.34)}
.stagebtn{margin-top:10px;height:26px;padding:0 10px;border-radius:8px;
  border:1px solid rgba(255,255,255,0.12);background:transparent;cursor:pointer;
  font-family:var(--l);font-size:9.5px;letter-spacing:0.12em;color:var(--t3)}
.stagebtn:hover{color:#fff}

.overlay{position:fixed;inset:0;z-index:40;display:none;place-items:center;
  background:rgba(0,0,0,0.82);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px)}
.overlay.on{display:grid}
.load{width:min(760px,92vw);max-height:86vh;overflow-y:auto;padding:26px 28px}
.load .hd{display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;
  gap:16px}
.load h2{margin:0;font-family:var(--d);font-size:26px;font-weight:500;letter-spacing:-0.01em}
.stages{display:flex;flex-direction:column;gap:2px}
.row{display:grid;grid-template-columns:12px 1fr auto;gap:12px;align-items:baseline;
  padding:7px 0;border-bottom:1px solid rgba(255,255,255,0.05);animation:rise 240ms ease both}
.row i{width:6px;height:6px;border-radius:50%;background:var(--t3);display:block;
  transform:translateY(-2px)}
.row.ok i{background:var(--ac)}
.row.run i{background:#fff;animation:blink 1.4s ease-in-out infinite}
.row.warn i{background:#f59e0b}
.row.fail i{background:#ef4444}
.row .n{font-family:var(--l);font-size:10.5px;letter-spacing:0.14em;color:rgba(255,255,255,0.7);
  text-transform:uppercase}
/* the measured values are the substance of this screen; Helvetica sits lighter than the
   monospace it replaced, so they get back the presence the face change cost them */
.row .v{font-family:var(--l);font-size:11.5px;color:rgba(255,255,255,0.62);display:block;
  margin-top:3px;letter-spacing:0;text-transform:none;word-break:break-word}
.row.warn .v{color:#fbbf24}
.row.fail .v{color:#fca5a5}
.row .tm{font-family:var(--l);font-size:10px;color:var(--t4)}
.note{margin-top:16px;font-size:12px;line-height:1.55;color:var(--t2)}

.panel{padding:18px 20px;font-size:13px;line-height:1.6;color:var(--t1)}
.panel h3{margin:0 0 10px;font-family:var(--l);font-size:10px;letter-spacing:0.2em;
  color:var(--t3);text-transform:uppercase}
.panel p{margin:0 0 10px}
.panel .hist{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
.panel .hist button{text-align:left;border:1px solid rgba(255,255,255,0.1);border-radius:12px;
  background:rgba(255,255,255,0.03);padding:9px 12px;cursor:pointer;color:var(--t1);
  font-family:var(--l);font-size:11px}
.panel .hist button:hover{border-color:rgba(255,255,255,0.3)}
.clear{height:30px;padding:0 12px;border-radius:9px;border:1px solid rgba(239,68,68,0.5);
  background:transparent;color:#fca5a5;font-family:var(--l);font-size:10px;letter-spacing:0.1em;
  cursor:pointer}
.clear:hover{background:rgba(239,68,68,0.14)}

@media (max-width:1180px){
  .wrap{flex-wrap:wrap}
  .left{flex:1 1 100%;max-width:100%;order:2}
  .right{flex:1 1 100%;order:1}
  .thread{max-height:none}
}
@media (max-width:720px){
  header{height:auto;padding:14px;flex-wrap:wrap;gap:10px}
  .wrap{padding:0 12px 20px;gap:12px;min-height:0}
  nav.rail{flex-direction:row;position:static;flex:1 1 100%;justify-content:center;
    align-self:auto}
  /* touch targets: 38px reads fine with a mouse and misses with a thumb */
  nav.rail button{width:44px;height:44px}
  .chip{height:34px;padding:0 14px}
  .msg{max-width:100%}
  .drop{font-size:20px}
  .load{padding:18px}
}
"""

_APP_JS = r"""
const CFG = window.__CFG__;
const S = {runs: [], sel: -1, file: null, h: CFG.paper_h, busy: false, panel: null};
const $ = (id) => document.getElementById(id);
const esc = (s) => {
  const d = document.createElement('div'); d.textContent = s; return d.innerHTML;
};

function chips(){
  const box = $('chips'); box.innerHTML = '';
  CFG.horizons.forEach(h => {
    const b = document.createElement('button');
    b.className = 'chip' + (h === S.h ? ' on' : '');
    b.textContent = h + ' MIN' + (h === CFG.paper_h ? ' · PAPER' : '');
    b.title = h === CFG.paper_h ? 'the only horizon the paper evaluates'
                                : 'trained, but never scored in the paper';
    b.onclick = () => { S.h = h; chips(); };
    box.appendChild(b);
  });
}

function setFile(f){
  S.file = f;
  const box = $('filebox');
  box.innerHTML = '';
  if (f){
    const d = document.createElement('div'); d.className = 'file rise';
    d.innerHTML = '<span class="tag">CSV</span><span>' + esc(f.name) + '</span>'
      + '<span style="color:rgba(255,255,255,0.3)">'
      + (f.size/1024).toFixed(0) + ' KB</span>';
    const x = document.createElement('button'); x.textContent = '\u00d7';
    x.onclick = () => setFile(null); d.appendChild(x); box.appendChild(d);
  }
  $('sendbtn').disabled = !f || S.busy;
  $('dropmsg').innerHTML = f
    ? 'Ready to forecast<small>' + esc(f.name) + ' · horizon ' + S.h + ' min</small>'
    : 'Drop an OHLCV CSV<small>timestamp, open, high, low, close (volume optional) · at least '
      + CFG.min_rows.toLocaleString() + ' rows</small>';
}

// The running row's clock comes from the SERVER's own elapsed, not a client-side interval. The
// first version ticked a setInterval into the DOM and the 400ms poll re-render wiped it every
// time, so a stage that takes a minute showed nothing at all. A timer the page keeps for itself
// is also a timer that can drift away from the run it claims to be timing.
function stageRow(s, elapsed){
  const d = document.createElement('div');
  d.className = 'row ' + s.state;
  const t = s.state === 'run' ? '+' + Math.max(0, elapsed - s.t).toFixed(0) + 's'
                              : s.t.toFixed(2) + 's';
  d.innerHTML = '<i></i><div><span class="n">' + esc(s.stage) + '</span>'
    + (s.detail ? '<span class="v">' + esc(s.detail) + '</span>' : '')
    + '</div><span class="tm">' + t + '</span>';
  return d;
}

function renderStages(list, elapsed){
  const box = $('stages');
  while (box.children.length > list.length) box.removeChild(box.lastChild);
  list.forEach((s, i) => {
    const fresh = stageRow(s, elapsed === undefined ? s.t : elapsed);
    const cur = box.children[i];
    if (!cur) box.appendChild(fresh);
    else if (cur.outerHTML !== fresh.outerHTML) box.replaceChild(fresh, cur);
  });
  box.scrollIntoView({block: 'end'});
}

function typeIn(el, text){
  let i = 0; el.textContent = '';
  const step = () => {
    i += 2; el.textContent = text.slice(0, i);
    if (i < text.length) setTimeout(step, 12);
  };
  step();
}

function thread(){
  const t = $('thread'); t.innerHTML = '';
  S.runs.forEach((r, i) => {
    const u = document.createElement('div'); u.className = 'msg you rise';
    u.innerHTML = '<div class="mhead">'
      + '<span class="who" style="color:rgba(255,255,255,0.55)">YOU</span>'
      + '<span class="st">' + esc(r.clock) + '</span></div><p>' + esc(r.file)
      + (r.rows ? ' · ' + r.rows.toLocaleString() + ' rows · ' + r.interval_s + 's bars' : '')
      + ' · horizon ' + r.h + ' min</p>';
    t.appendChild(u);

    const s = document.createElement('div');
    s.className = 'msg ' + (r.error ? 'bad' : 'sys rise') + (i === S.sel && !r.error ? ' sel' : '');
    const who = r.error ? '<span class="who" style="color:#fca5a5">REFUSED</span>'
                        : '<span class="who" style="color:var(--ac)">TRIKAAL · CELL 1</span>';
    s.innerHTML = '<div class="mhead">' + who
      + '<span class="st">' + esc(r.clock) + '</span></div>';
    if (r.error){
      const pre = document.createElement('pre'); pre.textContent = r.error; s.appendChild(pre);
    } else {
      const p = document.createElement('p'); p.id = 'read' + i; s.appendChild(p);
      if (r.typed) p.textContent = r.read; else { typeIn(p, r.read); r.typed = true; }
      (r.warnings || []).forEach(w => {
        const wd = document.createElement('p');
        wd.style.cssText = 'margin-top:10px;color:#fbbf24;font-size:12.5px';
        wd.textContent = w; s.appendChild(wd);
      });
      s.onclick = () => { S.sel = i; thread(); canvas(); };
      const b = document.createElement('button'); b.className = 'stagebtn';
      b.textContent = 'STAGE TRACE (' + r.stages.length + ')';
      b.onclick = (e) => { e.stopPropagation(); showStages(r); };
      s.appendChild(b);
    }
    t.appendChild(s);
  });
  t.scrollTop = t.scrollHeight;
}

function canvas(){
  const c = $('canvas');
  const r = S.runs[S.sel];
  // ★ ONE PLACE SETS THE STATE, AND IT IS THE PLACE THAT DRAWS THE FIGURE. `.right` holds the
  // chart AND the disclosures, so they can only ever appear together.
  document.body.dataset.state = (r && !r.error) ? 'result' : 'empty';
  if (!r || r.error){ c.innerHTML = ''; return; }
  c.innerHTML = r.svg + '<a class="dl" href="/export.csv?job=' + encodeURIComponent(r.id)
    + '" download>DOWNLOAD FORECAST CSV</a>';
}

function showStages(r){
  $('loadtitle').textContent = 'Stage trace · ' + r.file;
  $('loadsub').textContent = r.elapsed ? r.elapsed + 's total' : '';
  $('closeload').style.display = 'inline-flex';
  renderStages(r.stages);
  $('overlay').classList.add('on');
}

async function run(){
  if (!S.file || S.busy) return;
  S.busy = true; $('sendbtn').disabled = true;
  $('loadtitle').textContent = 'Forecasting · ' + S.file.name;
  $('loadsub').textContent = '';
  $('closeload').style.display = 'none';
  $('stages').innerHTML = '';
  $('overlay').classList.add('on');

  const fd = new FormData();
  fd.append('csv', S.file); fd.append('h', String(S.h));
  const clock = new Date().toTimeString().slice(0,5);
  let job;
  try {
    const res = await fetch('/run', {method: 'POST', body: fd});
    job = (await res.json()).job;
  } catch (e) {
    S.busy = false; $('sendbtn').disabled = false; $('overlay').classList.remove('on');
    alert('could not start the run: ' + e); return;
  }
  while (true){
    await new Promise(r => setTimeout(r, 400));
    const st = await (await fetch('/stages?job=' + encodeURIComponent(job))).json();
    renderStages(st.stages, st.elapsed);
    $('loadsub').textContent = st.elapsed.toFixed(1) + 's elapsed';
    if (st.done){
      const out = await (await fetch('/result?job=' + encodeURIComponent(job))).json();
      const rec = out.error
        ? {id: job, file: S.file.name, h: S.h, clock, error: out.error, stages: st.stages}
        : Object.assign({}, out.result, {clock, stages: st.stages});
      S.runs.push(rec);
      if (!out.error) S.sel = S.runs.length - 1;
      S.busy = false;
      $('overlay').classList.remove('on');
      setFile(null);
      thread(); canvas(); panel();
      break;
    }
  }
}

function panel(){
  const p = $('panel');
  if (!S.panel){ p.style.display = 'none'; return; }
  p.style.display = 'block';
  if (S.panel === 'history'){
    let h = '<h3>History · this session only</h3>';
    h += '<p>Held <b>in memory only</b>. Nothing is written to disk, so quitting the '
       + 'server erases it. What is kept: the forecast numbers, the rendered figure '
       + '(which contains the ' + CFG.seq_len + '-bar context path drawn from your file), '
       + 'the filename, row count, inferred interval, horizon and time. '
       + 'What is never kept, here or anywhere: <b>your uploaded bars</b>.</p>';
    h += '<div class="hist">';
    if (!S.runs.length) h += '<span style="color:var(--t3);font-size:12px">nothing yet</span>';
    S.runs.forEach((r, i) => {
      h += '<button data-i="' + i + '">' + esc(r.clock) + ' · ' + esc(r.file) + ' · h=' + r.h
        + (r.error ? ' · REFUSED' : ' · ' + r.seeds.map(s => s.pct.toFixed(3) + '%').join(' / '))
        + '</button>';
    });
    h += '</div><button class="clear" id="clr">CLEAR HISTORY</button>';
    p.innerHTML = h;
    p.querySelectorAll('.hist button').forEach(b => b.onclick = () => {
      const i = +b.dataset.i; if (!S.runs[i].error){ S.sel = i; thread(); canvas(); }
    });
    $('clr').onclick = async () => {
      await fetch('/forget', {method: 'POST'});
      S.runs = []; S.sel = -1; thread(); canvas(); panel();
    };
  } else {
    p.innerHTML = CFG.about;
  }
}

function boot(){
  chips(); setFile(null); thread(); canvas(); panel();
  $('sendbtn').onclick = run;
  $('pick').onclick = () => $('fileinput').click();
  $('fileinput').onchange = (e) => { if (e.target.files[0]) setFile(e.target.files[0]); };
  const comp = $('composer');
  ['dragenter','dragover'].forEach(ev => comp.addEventListener(ev, e => {
    e.preventDefault(); comp.classList.add('drag'); }));
  ['dragleave','drop'].forEach(ev => comp.addEventListener(ev, e => {
    e.preventDefault(); comp.classList.remove('drag'); }));
  comp.addEventListener('drop', e => {
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey && S.file && !S.busy){ e.preventDefault(); run(); }
    if (e.key === 'Escape' && !S.busy) $('overlay').classList.remove('on');
  });
  $('closeload').onclick = () => $('overlay').classList.remove('on');
  document.querySelectorAll('nav.rail button').forEach(b => b.onclick = () => {
    const k = b.dataset.k;
    if (k === 'new'){ S.panel = null; $('composer').scrollIntoView({behavior:'smooth'}); }
    else S.panel = (S.panel === k) ? null : k;
    document.querySelectorAll('nav.rail button').forEach(x =>
      x.classList.toggle('on', x.dataset.k === S.panel));
    panel();
  });
}
document.addEventListener('DOMContentLoaded', boot);
"""


def _about_html(n_seeds: int, params: int, mtp: int) -> str:
    return (
        "<h3>Method &amp; provenance</h3>"
        "<p><b>What this serves.</b> Cell 1 &#8212; BSQ quantizer, OHLCV-only inputs. That is the "
        "<b>baseline</b> arm of the ablation, not the microstructure arm. The microstructure cells "
        "have no scored artifacts at all: the legibility gate refused them before Stage-2, and "
        "that refusal is the project&#8217;s primary finding.</p>"
        f"<p><b>The model.</b> {params:,} realized parameters per seed, of which {mtp:,} "
        "are the MTP heads. A two-stage design: a Transformer autoencoder compresses each bar "
        "through a quantizer into ~20 bits, and a decoder-only causal Transformer "
        f"(8 layers, d_model 512, {SEQ_LEN}-bar context) predicts the next tokens.</p>"
        f"<p><b>Why {MIN_ROWS:,} rows.</b> The features are causally self-normalized by streaming "
        f"EWMA, so a fresh CSV starts <i>cold</i>: the first {WARMUP_ROWS} rows are warm-up and "
        f"the model reads a {SEQ_LEN}-bar context. {WARMUP_ROWS} + {SEQ_LEN} = {MIN_ROWS:,}. "
        "Below that, every bar the model would see sits inside warm-up, so a forecast is "
        "<b>refused</b> rather than returned with a caveat.</p>"
        f"<p><b>Three seeds, never averaged.</b> {n_seeds} identically configured runs on "
        "identical data produced headline information ratios spanning 118.03. The spread is a "
        "result, so it is drawn and never collapsed into a mean.</p>"
        "<p><b>Runs entirely on this machine.</b> The server binds to 127.0.0.1, no code path in "
        "this process makes an outbound request, your file is parsed in memory and never written "
        "to disk, and the page is served under a Content-Security-Policy that forbids loading "
        "anything from anywhere else. The reference design this is styled after pulls its fonts "
        "from a CDN; that request would have told a third party you opened this tool, so the page "
        "uses your system fonts instead.</p>"
    )


def _shell(n_seeds: int, params: int, mtp: int) -> str:
    cfg = json.dumps(
        {
            "horizons": list(SELECTABLE_HORIZONS),
            "paper_h": PAPER_HORIZON,
            "min_rows": MIN_ROWS,
            "seq_len": SEQ_LEN,
            "about": _about_html(n_seeds, params, mtp),
        }
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Trikaal &#183; CSV forecast</title>"
        f"<style>{_APP_CSS}</style></head>"
        # ★ data-state STARTS AT "empty". Everything that qualifies a forecast lives behind
        # data-state="result", and so does the forecast. See the CSS note above `.right`.
        '<body data-state="empty">'
        '<div class="bgwrap"><img src="/bg.png" alt=""></div>'
        "<header>"
        '<div style="display:flex;align-items:center;gap:18px;min-width:0">'
        '<div class="mark"><i></i></div>'
        '<div style="display:flex;align-items:center;gap:14px;min-width:0">'
        '<span class="lbl">Trikaal</span><span class="sep"></span>'
        '<span class="sub">Cell 1 &#183; BSQ, OHLCV-only &#183; the baseline arm</span>'
        "</div></div>"
        '<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
        f'<div class="pill"><span class="dot"></span>{n_seeds} SEEDS &#183; '
        "IDENTITY VERIFIED</div>"
        '<div class="pill">LOCAL &#183; NOTHING UPLOADED</div>'
        "</div></header>"
        '<div class="wrap">'
        '<nav class="rail">'
        '<button data-k="new" title="New forecast">&#9673;</button>'
        '<button data-k="history" title="History">&#9636;</button>'
        '<button data-k="about" title="About &amp; method">&#9671;</button>'
        "</nav>"
        '<div class="col left">'
        '<div class="thread scr" id="thread"></div>'
        '<div class="foot">'
        '<div class="glass composer" id="composer" style="flex:1;min-width:0">'
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:12px">'
        '<span class="k">Composer</span>'
        '<span class="k" style="letter-spacing:0.14em;color:var(--t4)">&#9166; to forecast</span>'
        "</div>"
        '<div class="drop" id="dropmsg"></div>'
        '<div id="filebox" style="margin-top:10px"></div>'
        '<div style="display:flex;align-items:center;justify-content:space-between;gap:12px;'
        'margin-top:14px;flex-wrap:wrap">'
        '<div class="chips" id="chips"></div>'
        '<button class="chip" id="pick">BROWSE&#8230;</button>'
        "</div>"
        '<input type="file" id="fileinput" accept=".csv,text/csv" style="display:none">'
        "</div></div></div>"
        '<button class="send" id="sendbtn" title="Forecast" disabled>'
        '<svg width="22" height="22" viewBox="0 0 22 22" fill="none" aria-hidden="true">'
        '<path d="M11 18V4M11 4L5 10M11 4l6 6" stroke="#0a0a0a" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg></button>'
        '<div class="col right">'
        '<div class="glass canvas" id="canvas"></div>'
        '<div class="glass disc">'
        "<h3>What this is not</h3><ul>"
        "<li><b>Forecast only.</b> No profit, position, equity curve or Sharpe is computed "
        "anywhere in this app &#8212; enforced by a test that parses the source, not by a "
        "promise.</li>"
        "<li><b>The measured edge is ~0.005&#8211;0.02% per trade, and NET NEGATIVE after a "
        "realistic ~0.10% round trip.</b> Break-even cost is 4.8&#8211;43&#215; below what "
        "execution actually costs.</li>"
        "<li><b>&#956;&#770; is 25&#8211;446&#215; over-dispersed</b> versus the returns it "
        "forecasts. The shaded fan is far wider than reality and is <b>not</b> a confidence "
        "interval &#8212; it is the spread of the model&#8217;s own samples.</li>"
        "<li><b>Three seeds, never averaged.</b> Identical data and configuration produced "
        "headline information ratios spanning 118.03.</li>"
        "<li><b>Trained on crypto 1-minute bars.</b> Any other asset class or bar interval is "
        "extrapolation, and only the interval can be detected.</li>"
        "<li><b>Not investment advice.</b></li>"
        "</ul></div>"
        '<div class="glass panel" id="panel" style="display:none"></div>'
        "</div></div>"
        '<div class="overlay" id="overlay"><div class="glass load scr">'
        '<div class="hd"><div><h2 id="loadtitle"></h2>'
        '<span class="k" id="loadsub" style="letter-spacing:0.12em"></span></div>'
        '<button class="chip" id="closeload" style="display:none">CLOSE</button></div>'
        '<div class="stages" id="stages"></div>'
        '<p class="note">Every line above is a real event in the pipeline, emitted by the code '
        "that does the work and carrying the value it actually computed &#8212; there is no "
        "scripted progress here. On CPU a full run takes minutes: most of it is the "
        "Monte-Carlo rollout, which re-fills a 512-bar context once per sampled path.</p>"
        "</div></div>"
        f"<script>window.__CFG__ = {cfg};</script>"
        f"<script>{_APP_JS}</script>"
        "</body></html>"
    )


# ─────────────────────────────────────────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────────────────────────────────────────
class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "trikaal-local"
    device = "cpu"

    def log_message(self, *a):
        pass

    def _head(self, code: int, ctype: str, n: int, extra: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(n))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()

    def _json(self, obj, code: int = 200):
        out = json.dumps(obj).encode()
        self._head(code, "application/json; charset=utf-8", len(out))
        self.wfile.write(out)

    def _job(self) -> Job | None:
        q = urllib.parse.urlparse(self.path).query
        jid = urllib.parse.parse_qs(q).get("job", [""])[0]
        with _LOCK:
            return _JOBS.get(jid)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/stages":
            job = self._job()
            if job is None:
                self._json({"error": "unknown job"}, 404)
                return
            with _LOCK:
                self._json(
                    {
                        "stages": list(job.stages),
                        "done": job.done,
                        "elapsed": round(time.monotonic() - job.started, 1),
                    }
                )
            return
        if path == "/result":
            job = self._job()
            if job is None:
                self._json({"error": "unknown job"}, 404)
                return
            with _LOCK:
                self._json({"error": job.error, "result": job.result})
            return
        if path == "/bg.png":
            # ★ SERVED FROM OUR OWN HANDLER, off a committed asset. The reference design pulls its
            # hero from a CDN; this one cannot, for the same reason the fonts cannot.
            data = BG_IMAGE.read_bytes() if BG_IMAGE.exists() else b""
            self._head(200 if data else 404, "image/png", len(data))
            self.wfile.write(data)
            return
        if path == "/export.csv":
            job = self._job()
            data = (job.export if job else "").encode()
            self._head(
                200 if data else 404,
                "text/csv; charset=utf-8",
                len(data),
                {"Content-Disposition": "attachment; filename=trikaal_forecast.csv"},
            )
            self.wfile.write(data)
            return
        with _LOCK:
            n_seeds = len(_UNITS)
            any_u = _UNITS[sorted(_UNITS)[0]] if _UNITS else None
        out = _shell(
            n_seeds,
            any_u.n_params_realized if any_u else 0,
            any_u.n_params_mtp if any_u else 0,
        ).encode()
        self._head(200, "text/html; charset=utf-8", len(out))
        self.wfile.write(out)

    def _multipart(self) -> tuple[str, str, int]:
        """Return (filename, csv text, horizon). Hand-rolled: no cgi module in 3.13+."""
        ctype = self.headers.get("Content-Type", "")
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_UPLOAD:
            raise CsvRefused(f"upload is {n / 1e6:.0f} MB; the limit is {MAX_UPLOAD / 1e6:.0f} MB")
        raw = self.rfile.read(n)
        name, text, h = "upload.csv", "", PAPER_HORIZON
        if "multipart/form-data" not in ctype or "boundary=" not in ctype:
            raise CsvRefused("expected a multipart form upload")
        bnd = ("--" + ctype.split("boundary=")[1].strip().strip('"')).encode()
        for part in raw.split(bnd):
            if b"\r\n\r\n" not in part:
                continue
            head, val = part.split(b"\r\n\r\n", 1)
            # ★ EXACT trailing-CRLF strip, NOT rstrip(b"\r\n-"): a CSV whose last line ends in
            # dashes would have had them eaten by the old form. One byte of the user's file is
            # still the user's file.
            if val.endswith(b"\r\n"):
                val = val[:-2]
            # ★ PARSE THE Content-Disposition LINE, NOT THE WHOLE HEAD BLOCK. A real browser sends
            # a Content-Type sub-header after it, so splitting the block on ";" made the LAST
            # filename token run to the end of the block and the app displayed
            # `x.csv"\r\nContent-Type: text/csv` as the user's filename. The unit test passed
            # throughout, because the multipart body it hand-built had no Content-Type line —
            # a fixture narrower than reality, which is a fixture that cannot discriminate.
            disp = b""
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-disposition:"):
                    disp = line
                    break
            if b'name="csv"' in disp:
                text = val.decode("utf-8", "replace")
                for token in disp.split(b";"):
                    token = token.strip()
                    if token.lower().startswith(b"filename="):
                        got = token.split(b"=", 1)[1].strip().strip(b'"').decode("utf-8", "replace")
                        name = got or name
            elif b'name="h"' in disp:
                try:
                    h = int(val.decode().strip())
                except ValueError:
                    h = PAPER_HORIZON
        return name, text, h

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/forget":
            with _LOCK:
                _JOBS.clear()
            self._json({"cleared": True})
            return
        if path != "/run":
            self._json({"error": "not found"}, 404)
            return
        try:
            name, text, h = self._multipart()
        except CsvRefused as e:
            self._json({"error": str(e)}, 400)
            return
        job = Job(id=secrets.token_hex(8), filename=name, h=h, started=time.monotonic())
        with _LOCK:
            _JOBS[job.id] = job
        threading.Thread(target=_execute, args=(job, text, self.device), daemon=True).start()
        self._json({"job": job.id})


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
    Handler.device = args.device
    print(f"[units] {seeds} loaded; identity verified at load AND re-verified on every forecast")
    print(f"[serve] http://127.0.0.1:{args.port} — LOOPBACK ONLY, nothing leaves this machine")
    print(f"[input] OHLCV CSV, >= {MIN_ROWS:,} rows; horizons {SELECTABLE_HORIZONS}")
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n[stop]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
