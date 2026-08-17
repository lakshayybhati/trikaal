"""The dashboard's page, figure and job store — the claims it makes, checked rather than trusted.

Three of these exist because the SHIPPED figure was wrong in ways no test could see, and all three
were found by rendering it and looking at it rather than by re-reading the code:

  * the seed legend was drawn at y=508 and the disclosure box (492-578) was appended afterwards and
    painted over it, so the figure had no legend at all;
  * the 25-446x over-dispersion caveat was a 210-character ``<text>`` node in a 1020-wide viewBox
    and ran off the right edge mid-word — the caveat was cut in half by the image it protects;
  * the forecast occupied 2.8% of the plot width, so the subject of the figure was invisible.

The lesson is the execution rule in miniature: the code EXISTED and was REVIEWED, and none of that
distinguishes "correct" from "never actually looked at". So the geometry is now asserted.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from trikaal.demo.csv_forecast import CsvForecast

REPO = Path(__file__).resolve().parents[2]
DASH_PATH = REPO / "scripts" / "m6_csv_dashboard.py"


def _load_dash():
    """Import the script without running it. It defines no side effects at import time.

    The module is registered in ``sys.modules`` BEFORE execution because ``@dataclass`` resolves
    annotations through ``sys.modules[cls.__module__]`` and raises on a module that is not there.
    """
    spec = importlib.util.spec_from_file_location("m6_csv_dashboard", DASH_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


dash = _load_dash()


def _fixture_forecast(*, h: int = 15, bar_ms: int = 60_000, n_ctx: int = 512) -> CsvForecast:
    """A REAL ``CsvForecast`` with synthetic numbers — the dataclass, not a duck-typed stub.

    The subject under test is ``_svg``; this is its INPUT, not a stand-in for it. Using the real
    dataclass means a field added to the forecast cannot silently bypass the figure's assertions.
    """
    rng = np.random.default_rng(7)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 2e-4, n_ctx)))
    seeds = (0, 2, 4)
    per_seed, quantiles, anchors = {}, {}, {}
    for k, s in enumerate(seeds):
        mu = (k - 1) * 3e-4
        per_seed[s] = {
            "mu": mu,
            "pct": (np.exp(mu) - 1.0) * 100.0,
            "price": float(close[-1]) * float(np.exp(mu)),
        }
        step = np.linspace(0.0, mu, h)
        quantiles[s] = {q: step + (q - 50) / 50.0 * 1.5e-3 for q in (10, 25, 50, 75, 90)}
        anchors[s] = {a: float(step[min(a, h) - 1]) for a in (5, 15) if a <= h}
    return CsvForecast(
        h=h,
        decision_ts_ms=1_700_000_000_000,
        last_close=float(close[-1]),
        per_seed=per_seed,
        band={"lo_pct": -0.03, "hi_pct": 0.03, "lo_price": 99.0, "hi_price": 101.0},
        context_ts=np.arange(n_ctx, dtype=np.int64) * bar_ms,
        context_close=close,
        inferred_bar_ms=bar_ms,
        warnings=[],
        n_rows=n_ctx + 720,
        quantiles=quantiles,
        n_mc_samples=48,
        mtp_anchors=anchors,
    )


# ─────────────────────────────────────────────────────────────────────────────────────────────
# "NOTHING LEAVES THIS MACHINE" — the page's own sentence, made checkable
# ─────────────────────────────────────────────────────────────────────────────────────────────
# Attributes a browser will actually FETCH. ``xmlns`` is deliberately absent: it is an XML
# namespace IDENTIFIER, never resolved over the network, and a scanner that flagged it would be
# either permanently red or (worse) whitelisted into uselessness. The distinction is the whole
# point of scanning by attribute rather than grepping for "http".
FETCHING_ATTRS = ("src", "href", "action", "srcset", "poster", "formaction", "data")
_ATTR_RE = re.compile(
    r"\b(" + "|".join(FETCHING_ATTRS) + r")\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE
)
_CSS_URL_RE = re.compile(r"(?:url\(|@import\s+)[\"']?([^\"')]+)", re.IGNORECASE)
_EXTERNAL = re.compile(r"^\s*(?:[a-z][a-z0-9+.-]*:)?//", re.IGNORECASE)


def external_references(markup: str) -> list[str]:
    """Every reference in ``markup`` that a browser would fetch from another host."""
    out = []
    for _attr, val in _ATTR_RE.findall(markup):
        if _EXTERNAL.match(val):
            out.append(val)
    for val in _CSS_URL_RE.findall(markup):
        val = val.strip()
        # ``@import url(x)`` matches on the ``@import`` branch and captures the ``url(`` with it;
        # unwrap before testing, or the scanner reports clean on the most common CSS import form.
        if val.lower().startswith("url("):
            val = val[4:].lstrip("\"'")
        if _EXTERNAL.match(val):
            out.append(val)
    return out


def test_the_external_reference_scanner_can_actually_fail():
    """FIXTURE DISCRIMINATION, against the exact line we refused to copy.

    The reference UI's line 13 is reproduced verbatim here. If the scanner cannot see THAT, it
    cannot protect the sentence we render on the page, and every green result below is decoration.
    """
    offending = (
        '<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400" '
        'rel="stylesheet" />'
    )
    assert external_references(offending) == [
        "https://fonts.googleapis.com/css2?family=Poppins:wght@300;400"
    ]
    assert external_references("<style>@import url(//evil.example/x.css);</style>")
    assert external_references('<img src="//cdn.example/a.png">')
    # ...and it must NOT fire on an XML namespace, or it would be whitelisted into uselessness
    assert (
        external_references('<svg xmlns="http://www.w3.org/2000/svg"><text>hi</text></svg>') == []
    )
    assert external_references('<a href="/export.csv?job=1">x</a>') == []


def test_served_page_fetches_nothing_from_anywhere_else():
    page = dash._shell(3, 31_725_568, 10_493_952)
    assert external_references(page) == [], (
        "the page promises the file never leaves this machine; a single external fetch tells a "
        "third party the user opened the tool and makes that sentence false"
    )


def test_figure_fetches_nothing_from_anywhere_else():
    assert external_references(dash._svg(_fixture_forecast())) == []


def test_csp_forbids_every_external_origin():
    """A promise the BROWSER enforces, so a future edit that adds a CDN is blocked, not shipped."""
    csp = dash.CSP
    assert "default-src 'none'" in csp
    assert "connect-src 'self'" in csp
    assert external_references(csp) == []
    for directive in ("script-src", "style-src", "img-src", "form-action"):
        assert directive in csp
    assert "http://" not in csp and "https://" not in csp


def test_no_font_is_fetched_and_none_is_redistributed():
    """★ TWO SEPARATE OBLIGATIONS, AND THE SECOND ONE IS A LICENCE, NOT A PREFERENCE.

    (1) NOTHING IS FETCHED. No @font-face src, no CDN, no Poppins/JetBrains link — the reason the
        reference design's Google Fonts line was refused in the first place.
    (2) NOTHING IS REDISTRIBUTED. Clash Display is referenced BY FAMILY NAME and resolves from the
        operator's own installed copy. The Fontshare EULA §02 forbids "uploading them in a public
        server", and this repository is public, so committing the woff2 would breach it even
        though self-hosting from 127.0.0.1 is otherwise exactly what the supervisor sanctioned.
        Wherever the font is absent the stack falls through to Helvetica.
    """
    for stack in (dash._SANS_CSS, dash._MONO_CSS):
        assert "googleapis" not in stack and "//" not in stack
    css = dash._APP_CSS
    assert "Poppins" not in css and "JetBrains" not in css
    assert "@font-face" not in css, "no font may be embedded or served; family names only"
    assert "Clash Display" in css and "Helvetica" in css
    shipped = [
        p.name
        for p in REPO.rglob("*")
        if p.suffix.lower() in {".woff", ".woff2", ".otf", ".ttf", ".eot"}
        and ".venv" not in p.parts
        and ".git" not in p.parts
    ]
    assert not shipped, (
        f"font files in the repository: {shipped}. The Fontshare EULA forbids redistribution, "
        "and this repository is public — reference the family name instead."
    )


def test_the_hero_image_is_ours_and_served_from_our_own_handler():
    page = dash._shell(3, 1, 1)
    assert 'src="/bg.png"' in page, "the hero must be a same-origin path, never a remote URL"
    assert external_references(page) == []
    assert dash.BG_IMAGE.exists(), f"{dash.BG_IMAGE} is referenced by the page but not committed"


def test_the_empty_state_is_bare_and_ONE_FLAG_reveals_figure_and_disclosures_TOGETHER():
    """★ THE STRUCTURAL GUARANTEE BEHIND HIDING THE DISCLOSURES ON THE LANDING SCREEN.

    The operator asked for a clean empty state, and that is defensible: there is no forecast yet,
    so there is nothing to qualify. What is NOT defensible is a build where a later edit reveals a
    forecast while its qualifiers stay hidden. So the chart and the "what this is not" panel are
    siblings inside `.right`, and a SINGLE selector controls the pair. To show one without the
    other you would have to move it out of `.right`, which this test refuses.
    """
    page = dash._shell(3, 1, 1)
    assert 'data-state="empty"' in page, "the page must start with nothing revealed"
    assert 'body[data-state="empty"] .right{display:none}' in dash._APP_CSS

    right = page.split('<div class="col right">', 1)[1]
    # everything up to the close of the right column
    assert 'id="canvas"' in right, "the figure must live inside .right"
    disc = right.split('<div class="glass panel"', 1)[0]
    assert 'class="glass disc"' in disc, "the disclosure panel must live inside .right too"
    assert "NET NEGATIVE" in disc and "over-dispersed" in disc

    # ...and only ONE place in the JS may set the flag, so it cannot be flipped from a path that
    # does not draw a figure.
    js = dash._APP_JS
    assert js.count("dataset.state") == 1, (
        f"exactly one assignment may control the revealed state; found {js.count('dataset.state')}"
    )


# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE FIGURE'S GEOMETRY — the three defects that shipped
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _rect(svg: str, el_id: str):
    """Select a rect BY ID, not by a geometry heuristic.

    The first version of these tests picked "the rect with rx=6". When the own-scale inset was
    added it also had rx=6, came first in document order, and one test silently began measuring
    THAT box instead of the disclosure box — and still passed, because enough mono text happened
    to fall inside the inset to satisfy its line count. A selector that can quietly start matching
    a different element is a check that has stopped checking what it names.
    """
    root = ET.fromstring(svg)
    hits = [el for el in root.iter("{http://www.w3.org/2000/svg}rect") if el.get("id") == el_id]
    assert len(hits) == 1, f"expected exactly one rect id={el_id!r}, found {len(hits)}"
    return hits[0]


def _text_nodes(svg: str):
    """(x, y, font_px, is_mono, text) for every text node, read from the rendered artifact."""
    root = ET.fromstring(svg)
    out = []
    for el in root.iter("{http://www.w3.org/2000/svg}text"):
        style = el.get("style", "")
        m = re.search(r"font:(?:\s*\d+\s+)?([\d.]+)px\s+([^;]+)", style)
        if not m:
            continue
        out.append(
            (
                float(el.get("x", 0)),
                float(el.get("y", 0)),
                float(m.group(1)),
                "mono" in m.group(2),
                el.text or "",
                el.get("text-anchor", "start"),
            )
        )
    return out


def test_no_text_node_runs_off_the_right_edge():
    """DEFECT 2. The mono bound is ARITHMETIC (0.6em advance, exact for any monospace face); the
    sans bound is a conservative 0.58em and is additionally confirmed by looking at the render.
    """
    svg = dash._svg(_fixture_forecast())
    w = int(re.search(r'viewBox="0 0 (\d+)', svg).group(1))
    right = w - 30  # pr
    over = []
    for x, _y, px, is_mono, text, anchor in _text_nodes(svg):
        width = len(text) * px * (dash.MONO_ADVANCE if is_mono else 0.58)
        end = x + width if anchor == "start" else (x if anchor == "end" else x + width / 2)
        if end > right:
            over.append((text[:60], round(end), right))
    assert not over, f"text runs past x={right}: {over}"


def test_every_disclosure_line_fits_inside_the_box_it_is_drawn_in():
    """Each line is measured at ITS OWN rendered font size, read back off the artifact.

    Measuring every line against one hand-picked constant is how a check ends up asserting
    something the code never claimed — the first version of this test used the HEADER size for
    BODY lines and failed a figure that was correct.
    """
    svg = dash._svg(_fixture_forecast())
    box = _rect(svg, "disclosure")
    top, bot = float(box.get("y")), float(box.get("y")) + float(box.get("height"))
    inner = float(box.get("width")) - 2 * dash.DISC_PAD
    lines = [
        (px, t) for _x, y, px, is_mono, t, _a in _text_nodes(svg) if is_mono and top <= y <= bot
    ]
    assert len(lines) >= 8, f"expected the wrapped disclosure block, got {len(lines)} lines"
    for px, line in lines:
        assert len(line) * px * dash.MONO_ADVANCE <= inner + 1e-6, (px, line)


def test_wrap_handles_a_word_longer_than_the_budget():
    assert dash._wrap("a bb ccc", 3) == ["a", "bb", "ccc"]
    assert dash._wrap("x" * 10, 4) == ["xxxx", "xxxx", "xx"]
    assert dash._wrap("hello enormous", 6) == ["hello", "enormo", "us"]


def test_the_seed_legend_is_visible_and_not_under_the_disclosure_box():
    """DEFECT 1. The legend used to be inside the box's y-range AND drawn before it."""
    svg = dash._svg(_fixture_forecast())
    box = _rect(svg, "disclosure")
    box_top = float(box.get("y"))
    box_bot = box_top + float(box.get("height"))

    legend = [(x, y, t) for x, y, _px, _m, t, _a in _text_nodes(svg) if t.startswith("seed ")]
    assert len(legend) == 3, f"three seeds must be labelled, got {legend}"
    for _x, y, t in legend:
        assert not (box_top <= y <= box_bot), f"{t!r} at y={y} sits under the disclosure box"


def test_the_forecast_region_is_a_readable_share_of_the_plot():
    """DEFECT 3. 15 steps against 512 bars on one linear axis gave the forecast 2.8% of the width.

    The split must be announced in the raster too: a compressed axis that is not labelled is a
    misleading axis, whatever it buys in legibility.
    """
    assert 0.25 <= 1 - dash.HIST_FRAC <= 0.5
    svg = dash._svg(_fixture_forecast())
    text = " ".join(t for *_, t, _a in _text_nodes(svg))
    assert "COMPRESSED" in text and "EXPANDED" in text
    # matched case-insensitively on CONTENT: the qualifier moved into the disclosure block (where
    # it wraps instead of clipping) and is rendered in caps there. A test that pins the casing of
    # a sentence pins the wrong thing — what must survive is that the split is ANNOUNCED.
    low = text.lower()
    assert "not one scale" in low or "not the same scale" in low


# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE DISCLOSURES SURVIVE THE REDESIGN AT FULL STRENGTH
# ─────────────────────────────────────────────────────────────────────────────────────────────
REQUIRED_IN_RASTER = [
    "READ THE BAND, NOT THE WIGGLE",
    "25-446x OVER-DISPERSED",
    "NOT confidence intervals",
    "NET NEGATIVE",
    "0.005-0.02%",
    "quantized at ~0.55z",
    "TRAINED ON CRYPTO 1-MINUTE BARS",
    "No profit, position or P&L is computed",
    "Not investment advice",
    "NEVER averaged",
]


@pytest.mark.parametrize("needle", REQUIRED_IN_RASTER)
def test_every_disclosure_is_inside_the_raster(needle):
    """CONTEXT-STRIPPING: the qualifier must travel with the image, not beside it."""
    body = " ".join(dash.disclosure_lines(_fixture_forecast()))
    assert needle in body


def test_out_of_distribution_input_makes_the_box_loud_not_quiet():
    ood = dash._svg(_fixture_forecast(bar_ms=300_000))
    ok = dash._svg(_fixture_forecast(bar_ms=60_000))
    assert "OUT OF DISTRIBUTION" in ood and "OUT OF DISTRIBUTION" not in ok
    assert dash.OOD_BD in ood and dash.OOD_BD not in ok
    assert 'stroke-width="2"' in ood


def test_page_disclosures_are_not_shrunk_into_the_faintest_ramp_slot():
    """The palette has a 0.22-opacity slot; the disclosures are not allowed to live in it."""
    page = dash._shell(3, 31_725_568, 10_493_952)
    block = page.split('class="glass disc"')[1].split("</div>")[0]
    assert "--t4" not in block and "0.22" not in block
    for needle in ("NET NEGATIVE", "over-dispersed", "never averaged", "Not investment advice"):
        assert needle in block


def test_the_credit_pill_became_a_true_statement():
    page = dash._shell(3, 1, 1)
    assert "LOCAL &#183; NOTHING UPLOADED" in page
    assert "CREDIT" not in page.upper().replace("CREDITS", "")


# ─────────────────────────────────────────────────────────────────────────────────────────────
# HISTORY STORES RESULTS, NEVER INPUTS
# ─────────────────────────────────────────────────────────────────────────────────────────────
def test_job_record_has_nowhere_to_keep_the_uploaded_bars():
    """Pinned like ``ForecastBundle``'s missing mean: adding a place to keep the file FAILS."""
    fields = set(dash.Job.__dataclass_fields__)
    assert fields == {
        "id",
        "filename",
        "h",
        "started",
        "stages",
        "done",
        "error",
        "result",
        "export",
    }
    banned = {"text", "csv", "raw", "bars", "upload", "body", "content", "source"}
    assert not (fields & banned)


def test_result_payload_carries_no_input_series():
    """The keys the page receives are named and bounded — no raw OHLC among them."""
    src = ast.parse(DASH_PATH.read_text())
    keys = set()
    for node in ast.walk(src):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    keys.add(k.value)
    for banned in ("open", "high", "low", "bars", "ohlc", "raw_rows", "csv_text"):
        assert banned not in keys, f"the result payload must not carry {banned!r}"


def test_nothing_is_written_to_disk_by_the_request_path():
    """No open()/write/mkdir anywhere in the module — the page says the file is never stored."""
    tree = ast.parse(DASH_PATH.read_text())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in {"open", "write_text", "write_bytes", "mkdir", "touch", "save"}:
                bad.append(f"{name} at line {node.lineno}")
    assert not bad, f"the dashboard must not write anything to disk: {bad}"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE MULTIPART READER
# ─────────────────────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────────────────────
# IDENTITY, RE-VERIFIED AT USE — and WATCHED FAILING, which is the only thing that closes a gate
# ─────────────────────────────────────────────────────────────────────────────────────────────
def test_reverify_identity_passes_on_a_real_unit():
    from trikaal.demo.inference import available_seeds, load_unit, reverify_identity

    seeds = available_seeds()
    if not seeds:
        pytest.skip("banked units not present")
    v = reverify_identity(load_unit(seeds[0]))
    assert v["ok"] is True
    assert set(v["checks"]) == {"tokenizer", "predictor"}
    for c in v["checks"].values():
        assert c["recomputed"] == c["stored"] == c["manifest"]


def test_reverify_identity_FAILS_on_a_weight_mutated_after_loading():
    """★ THE GATE, WATCHED FAILING — and shown to catch what the load-time check CANNOT.

    ``load_unit`` hashes the state dict it read off disk. Corrupt a parameter afterwards and that
    check is untouched: the stored hash field still equals the manifest, so anything reading the
    FIELD reports a clean bill of health on a model that is no longer the scored one. The second
    assertion below is the discriminating one — it proves the recomputation is load-bearing rather
    than a more expensive way of re-reading the same string.
    """
    import torch

    from trikaal.demo.inference import available_seeds, load_unit, reverify_identity

    seeds = available_seeds()
    if not seeds:
        pytest.skip("banked units not present")
    u = load_unit(seeds[0])
    assert reverify_identity(u)["ok"] is True, "must be clean BEFORE the mutation"

    stored_before = u.predictor_hash
    with torch.no_grad():
        p = next(iter(u.model.parameters()))
        p.view(-1)[0] += 1.0

    v = reverify_identity(u)
    assert v["ok"] is False, "a mutated live weight must fail re-verification"
    assert v["checks"]["predictor"]["ok"] is False
    assert v["checks"]["tokenizer"]["ok"] is True, "only the mutated module may fail"
    # ...and the stored/manifest strings are IDENTICAL to before, so a check that read them
    # instead of recomputing would still be green on this exact object.
    assert u.predictor_hash == stored_before
    assert v["checks"]["predictor"]["stored"] == v["checks"]["predictor"]["manifest"]
    assert v["checks"]["predictor"]["recomputed"] != v["checks"]["predictor"]["stored"]


def _read_multipart(body: str, *, filename: str, content_type: bool):
    """Drive ``Handler._multipart`` over a hand-built body, optionally with the ``Content-Type``
    sub-header a REAL browser emits after ``Content-Disposition``."""
    bnd = "----X"
    ct = "Content-Type: text/csv\r\n" if content_type else ""
    raw = (
        f"--{bnd}\r\n"
        f'Content-Disposition: form-data; name="csv"; filename="{filename}"\r\n'
        f"{ct}"
        "\r\n"
        f"{body}\r\n"
        f"--{bnd}\r\n"
        'Content-Disposition: form-data; name="h"\r\n\r\n'
        "15\r\n"
        f"--{bnd}--\r\n"
    ).encode()

    class _H(dash.Handler):
        def __init__(self):  # bypass BaseHTTPRequestHandler's socket setup
            import io as _io

            self.headers = {
                "Content-Type": f"multipart/form-data; boundary={bnd}",
                "Content-Length": str(len(raw)),
            }
            self.rfile = _io.BytesIO(raw)

    return _H()._multipart()


def test_trailing_dashes_in_the_file_survive_the_upload():
    """The old reader did ``rstrip(b"\\r\\n-")`` and ate real bytes off the end of the user's file.

    One byte of their data is still their data, and a CSV whose final field is a dash is a
    perfectly ordinary CSV.
    """
    body = "timestamp,open,high,low,close\n1,1,1,1,---\n"
    name, text, h = _read_multipart(body, filename="t.csv", content_type=True)
    assert name == "t.csv"
    assert h == 15
    assert text == body, f"file content was altered: {text!r}"


@pytest.mark.parametrize("content_type", [True, False], ids=["browser-form", "bare-form"])
def test_filename_is_read_from_the_content_disposition_line_only(content_type):
    """★ THE FIXTURE THAT COULD NOT DISCRIMINATE, NOW FIXED IN BOTH DIRECTIONS.

    The first version of this test built a multipart body WITHOUT the ``Content-Type`` sub-header
    that every browser sends. The reader split the whole header block on ";", so the filename token
    ran past the closing quote and the app displayed

        SYNTHUSDT_1m.csv"\\r\\nContent-Type: text/csv

    as the user's filename. The test was green the entire time. It was found by uploading a file
    through the actual UI and reading the card — the same lesson as the demo's shared-input bug:
    the defect lived in the part of reality the fixture had left out, so no amount of extra
    assertion on the existing fixture could have reached it. Both shapes are now pinned.
    """
    name, text, _h = _read_multipart(
        "a,b\n1,2\n", filename="My File.csv", content_type=content_type
    )
    assert name == "My File.csv"
    assert "\r" not in name and "Content-Type" not in name
    assert text == "a,b\n1,2\n"


def test_the_own_scale_inset_carries_its_own_axis_and_says_what_it_changed():
    """CONTEXT-STRIPPING, applied to the one panel most likely to be cropped out on its own.

    A magnified forecast panel looks like a large move. So the inset must be self-describing when
    it travels alone: its own y-axis, labelled in PERCENT, plus a caption naming the rescale. The
    main panel keeps the true relative scale and is never replaced by this.
    """
    svg = dash._svg(_fixture_forecast())
    inset = _rect(svg, "inset")
    top, bot = float(inset.get("y")), float(inset.get("y")) + float(inset.get("height"))
    inside = [(t, px) for _x, y, px, _m, t, _a in _text_nodes(svg) if top <= y <= bot]
    text = " ".join(t for t, _ in inside)
    assert "OWN Y-SCALE" in text and "TRUE SIZE" in text
    pct_labels = [t for t, _ in inside if t.endswith("%")]
    assert len(pct_labels) >= 3, f"the inset must label its own axis in percent: {pct_labels}"
    assert any(t.startswith("+") for t in pct_labels) or any(
        t.startswith("-") for t in pct_labels
    ), "signed percent labels, so a cropped inset cannot be read as a price"


def test_the_main_panel_still_shows_the_forecast_at_the_true_shared_scale():
    """The inset is an ADDITION. If the main panel ever stops drawing the fan, the figure has
    started hiding how small the forecast is relative to the bars, which is its main finding."""
    f = _fixture_forecast()
    svg = dash._svg(f)
    inset = _rect(svg, "inset")
    top = float(inset.get("y"))
    root = ET.fromstring(svg)
    above = [
        el
        for el in root.iter("{http://www.w3.org/2000/svg}polygon")
        if min(float(pt.split(",")[1]) for pt in el.get("points").split()) < top
    ]
    assert len(above) == 2 * len(f.per_seed), (
        "the main panel must still draw both percentile bands for every seed at the shared scale"
    )


@pytest.mark.parametrize(
    "text,label",
    [
        ("", "empty"),
        ("timestamp,open,high,low,close\n1,1,1,1,1\n", "below-the-row-floor"),
        ("not,a,csv\nx,y,z\n", "missing-columns"),
    ],
)
def test_the_run_lock_is_released_on_every_refusal_path(text, label):
    """★ A LEAKED LOCK IS INVISIBLE UNTIL THE *NEXT* RUN, WHICH HANGS FOREVER WITH NO ERROR.

    Only one forecast runs at a time. The refusals all raise out of the middle of the guarded
    block, so every one of them has to unwind through the release. This is the shape that would
    otherwise be found by a user whose second upload sat on a loading screen with no lines, long
    after the run that broke it had been forgotten.
    """
    import time as _t

    job = dash.Job(id=label, filename="x.csv", h=15, started=_t.monotonic())
    dash._execute(job, text, "cpu")
    assert job.error, f"{label} should have been refused"
    assert job.done is True
    assert not dash._RUN_LOCK.locked(), "the run lock leaked; the next forecast would hang"
    assert job.stages and job.stages[-1]["state"] == "fail"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE TWO GUARDS AGAINST A RUN THAT CANNOT FAIL VISIBLY
# ─────────────────────────────────────────────────────────────────────────────────────────────
def test_the_watchdog_fails_a_stalled_stage_instead_of_hanging():
    """★ WATCHED FIRING. A stage that starts and never finishes must become a FAILED stage with
    its real elapsed time, not a green line the operator stares at for 40 minutes."""
    import threading
    import time as _t

    job = dash.Job(id="stall", filename="x.csv", h=15, started=_t.monotonic())
    dash._emit(job, "MTP ANCHORS", "seed 4", "run")
    threading.Thread(target=dash._watchdog, args=(job, 2.0), daemon=True).start()
    t0 = _t.time()
    while not job.done and _t.time() - t0 < 15:
        _t.sleep(0.2)
    assert job.done, "the watchdog never fired; a stalled stage would hang forever"
    assert job.stages[-1]["state"] == "fail"
    assert "MTP ANCHORS" in job.error and "budget" in job.error


def test_the_watchdog_does_NOT_fire_on_a_healthy_run():
    """NEGATIVE CONTROL. Without this the guard could be a stopwatch that fails every real run."""
    import threading
    import time as _t

    job = dash.Job(id="ok", filename="x.csv", h=15, started=_t.monotonic())
    dash._emit(job, "TOKENIZING", "seed 0", "run")
    threading.Thread(target=dash._watchdog, args=(job, 2.0), daemon=True).start()
    _t.sleep(0.5)
    dash._emit(job, "TOKENIZING", "seed 0: done", "ok")
    with dash._LOCK:
        job.done = True
    _t.sleep(3.0)
    assert job.error is None, "the watchdog failed a job that completed inside its budget"


def test_a_failed_verdict_is_sticky_against_a_late_worker():
    """An orphaned worker that finishes later must not overwrite the failure the operator saw."""
    import time as _t

    job = dash.Job(id="s", filename="x.csv", h=15, started=_t.monotonic())
    dash._emit(job, "SAMPLING TRAJECTORIES", "seed 4", "run")
    dash._fail(job, "TIMED OUT", "budget exceeded")
    n_before, err_before = len(job.stages), job.error
    dash._emit(job, "SAMPLING TRAJECTORIES", "seed 4: finished after all", "ok")
    dash._fail(job, "SOMETHING ELSE", "a different error")
    assert job.stages[-1]["state"] == "fail"
    assert len(job.stages) == n_before and job.error == err_before


def test_the_client_can_tell_a_dead_server_from_a_slow_stage():
    """★ THE DEFECT THAT PRODUCED THE REPORTED SYMPTOM, PINNED.

    The poll loop had NO try/catch. A dead server made `fetch` reject, the loop threw, and the
    overlay froze on the last stages it had drawn with `S.busy` stuck true — which looks EXACTLY
    like a stage that is still running. The operator could not tell the two apart, and neither
    could we. Three distinct states must now be reachable and named in the client:
    """
    js = dash._APP_JS
    assert "catch (e)" in js, "the poll loop must not be able to die silently"
    assert "SERVER NOT RESPONDING" in js, "an unreachable server must say so"
    assert "STOPPED RESPONDING" in js, "a give-up path must exist and name the cause"
    assert "no new stage for" in js, "alive-but-slow must be distinguishable from unreachable"
    assert "ABANDONED BY YOU" in js and "abandon" in js, "the UI must have an escape hatch"
    # ...and the give-up path must be BOUNDED, or it is the same infinite wait wearing a hat
    assert "fails >= 25" in js
    # the loop's fetch must be inside the try, not before it
    body = js.split("while (true){", 1)[1]
    fetch_at = body.index("fetch('/stages")
    try_at = body.index("try {")
    assert try_at < fetch_at, "the stage fetch must be inside the try block"


def test_the_run_lock_wait_is_bounded():
    """An orphaned worker holds the model lock forever; later runs must refuse, not queue."""
    src = DASH_PATH.read_text()
    assert "_RUN_LOCK.acquire(timeout=" in src, "an unbounded acquire re-creates the silent hang"
    assert "_RUN_LOCK.acquire()" not in src


def test_stage_completions_are_logged_unbuffered():
    """Instrumentation for the NEXT occurrence: if the process dies, the log still names the last
    stage it reached. Buffered output would be lost exactly when it is needed."""
    src = DASH_PATH.read_text()
    assert "flush=True" in src, "a buffered trail is no trail when the process is killed"


# ─────────────────────────────────────────────────────────────────────────────────────────────
# "WHAT THE THREE MODELS ARE" — the panel that corrects the misreading
# ─────────────────────────────────────────────────────────────────────────────────────────────
def test_the_seeds_panel_corrects_the_extreme_scenarios_misreading():
    """★ THE MISREADING IS THE POINT. Asked what the three numbers were, the operator guessed
    EXTREME SCENARIOS with a weighted average in the middle. They are not: they are three
    independent training runs differing only in their random starting point, and there is no
    fourth better number between them. Correcting that is worth more than any summary figure,
    so the denial is explicit rather than left to be inferred.
    """
    page = dash._shell(3, 1, 1)
    panel = page.split("What the three models are", 1)[1].split("</ul>", 1)[0]
    assert "only" in panel and "random starting point" in panel
    assert "not a best case, a worst case and a middle case" in panel
    assert "no fourth, better number hiding between them" in panel
    assert "the disagreement is the result" in panel
    assert "It is the finding" in panel


@pytest.mark.parametrize(
    "claim,source_key",
    [
        ("short 67.7%", "m6_skill_vs_bias.json"),
        ("short 65.5%", "m6_skill_vs_bias.json"),
        ("long 92.7%", "m6_skill_vs_bias.json"),
        ("26.25%", "m6_demo_seed_sign_disagreement.json"),
        ("39.12%", "m6_demo_seed_sign_disagreement.json"),
        ("118.03", "m6_skill_vs_bias.json"),
    ],
)
def test_every_number_on_the_seeds_panel_is_on_the_page_with_its_source(claim, source_key):
    page = dash._shell(3, 1, 1)
    assert claim in page, f"{claim} is cited in the panel content but not rendered"
    assert source_key in page, f"{claim} is shown without naming {source_key}"


def test_the_panel_numbers_match_the_receipts_they_cite():
    """★ RECOMPUTED FROM THE ARTIFACTS, NOT COPIED FROM THE INSTRUCTION.

    The 39.12% independence baseline in particular appears in NO receipt field — it is derived
    from the disagreement receipt's own per-seed marginals, and a figure put in front of a reader
    has to be checked against the thing it claims to come from rather than the message that
    supplied it.
    """
    import json

    bias = json.loads((REPO / "runs_manifest" / "m6_skill_vs_bias.json").read_text())
    by_seed = {r["seed"]: r for r in bias["rows"]}
    assert by_seed[0]["modal_sign"] == "SHORT" and round(by_seed[0]["modal_share"] * 100, 1) == 67.7
    assert by_seed[2]["modal_sign"] == "SHORT" and round(by_seed[2]["modal_share"] * 100, 1) == 65.5
    assert by_seed[4]["modal_sign"] == "LONG" and round(by_seed[4]["modal_share"] * 100, 1) == 92.7

    dis = json.loads((REPO / "runs_manifest" / "m6_demo_seed_sign_disagreement.json").read_text())
    assert round(dis["pairwise_sign_agreement"]["0&4"] * 100, 2) == 26.25
    m = dis["per_seed_marginal_p_mu_positive"]
    p0, p4 = m["0"], m["4"]
    baseline = p0 * p4 + (1 - p0) * (1 - p4)
    assert round(baseline * 100, 2) == 39.12, (
        "the independence baseline must be recomputed, not quoted"
    )
    assert dis["n_decisions"] == 80


def test_the_raw_disagreement_rate_is_NOT_quoted_alone():
    """The disagreement receipt's own READ_THIS_FIRST forbids it: 86.2% of decisions disagree,
    but independence ALONE predicts a high rate, so the bare number misleads. The panel makes the
    comparison against the baseline instead, which is the claim the receipt actually supports."""
    page = dash._shell(3, 1, 1)
    assert "86.2" not in page and "0.8625" not in page


def test_the_figure_says_whose_most_likely_scenario_it_is():
    """ "Most likely" with no owner reads as a single prediction. Each dashed line is the most
    likely path FOR THAT MODEL, and the three models disagree."""
    body = " ".join(dash.disclosure_lines(_fixture_forecast()))
    assert "most likely scenario FOR THAT MODEL" in body
    assert "not a single prediction" in body
    assert "random starting point" in body


# ─────────────────────────────────────────────────────────────────────────────────────────────
# THE Y-AXIS IS THE ASSET'S OWN PRICE — and tick formatting breaks by MAGNITUDE, so test by it
# ─────────────────────────────────────────────────────────────────────────────────────────────
def _fc_at_price(anchor: float, h: int = 15, n_ctx: int = 512):
    """A forecast whose closes sit around ``anchor`` — the one input tick formatting depends on."""
    rng = np.random.default_rng(3)
    close = anchor * np.exp(np.cumsum(rng.normal(0, 3e-4, n_ctx)))
    close = close * (anchor / close[-1])  # land exactly on the anchor
    per_seed, quantiles, anchors = {}, {}, {}
    for k, s in enumerate((0, 2, 4)):
        mu = (k - 1) * 4e-4
        step = np.linspace(0.0, mu, h)
        quantiles[s] = {q: step + (q - 50) / 50.0 * 2e-3 for q in (10, 25, 50, 75, 90)}
        per_seed[s] = {
            "mu": float(step[-1]),
            "pct": (float(np.exp(step[-1])) - 1) * 100,
            "price": anchor * float(np.exp(step[-1])),
            "mu_expectation": mu,
            "pct_expectation": (float(np.exp(mu)) - 1) * 100,
            "mc_mean_pct": 0.0,
        }
        anchors[s] = {a: float(step[min(a, h) - 1]) for a in (5, 15) if a <= h}
    stacked = np.concatenate([np.tile(quantiles[s][50], (48, 1)) for s in quantiles])
    from trikaal.demo.csv_forecast import path_quantiles

    return CsvForecast(
        h=h,
        decision_ts_ms=1_700_000_000_000,
        last_close=float(anchor),
        per_seed=per_seed,
        band={},
        context_ts=np.arange(n_ctx, dtype=np.int64) * 60_000,
        context_close=close,
        inferred_bar_ms=60_000,
        warnings=[],
        n_rows=n_ctx + 720,
        quantiles=quantiles,
        n_mc_samples=48,
        mtp_anchors=anchors,
        pooled=path_quantiles(stacked),
        n_pooled=144,
    )


def _y_tick_labels(svg: str) -> list[str]:
    root = ET.fromstring(svg)
    out = []
    for el in root.iter("{http://www.w3.org/2000/svg}text"):
        if el.get("text-anchor") == "end" and float(el.get("x", 0)) < 115:
            out.append(el.text or "")
    return out


@pytest.mark.parametrize(
    "anchor", [63_142.87, 2_481.30, 142.06, 0.4213, 0.10382, 0.00001053, 1_234_567.89]
)
def test_the_y_axis_is_the_assets_own_price_at_every_magnitude(anchor):
    """★ NORMALIZED TICKS ARE MEANINGLESS TO THE PERSON WHO SUPPLIED THE FILE.

    "1.0012 / 1.0005 / 0.9997" beside a BTCUSDT chart makes a $63,000 instrument unrecognisable as
    itself. The internal normalization stays — a shared scale is why the history and the fan are
    comparable — but the tick VALUES are multiplied back by the last close.

    ★ AND THE PRECISION IS DERIVED, NOT PICKED, WHICH IS WHY THIS IS PARAMETRIZED BY MAGNITUDE.
    Any fixed decimal count makes some asset unreadable: 2dp renders a micro-cap as
    "0.00 / 0.00 / 0.00", 8dp renders BTC as "63,142.87000000". Tick formatting is exactly the
    class of thing that reads fine at one scale and collapses at another, so every tick must be
    DISTINCT from its neighbour at every magnitude — that is the property, and it is checked here
    rather than eyeballed at one price.
    """
    svg = dash._svg(_fc_at_price(anchor))
    ticks = _y_tick_labels(svg)
    assert len(ticks) >= 4, ticks
    assert len(set(ticks)) == len(ticks), (
        f"tick labels collapsed to duplicates at anchor {anchor}: {ticks} — the derived precision "
        "is not resolving the spacing"
    )
    vals = sorted(float(t.replace(",", "")) for t in ticks)
    assert vals[0] < anchor < vals[-1] or any(abs(v - anchor) < 1e-12 for v in vals), (
        f"the last close {anchor} is not on the axis: {ticks}"
    )
    # thousands separators wherever they help, and never scientific notation
    assert not any("e" in t.lower() for t in ticks), f"scientific notation on the axis: {ticks}"
    if anchor >= 10_000:
        assert any("," in t for t in ticks), f"no thousands separator at {anchor}: {ticks}"


def test_the_axis_names_its_unit_without_inventing_a_currency():
    """A CSV has no currency column. Inferring "$" or "USD" would put a fact on the figure that
    the input never contained."""
    svg = dash._svg(_fc_at_price(63_142.87))
    text = " ".join(t for *_, t, _a in _text_nodes(svg))
    assert "PRICE, AS SUPPLIED IN YOUR CSV" in text
    assert "BOLD TICK = LAST CLOSE" in text
    for invented in ("USD", "$", "USDT", "EUR"):
        assert invented not in text, f"the figure invented a currency: {invented}"


def test_the_last_close_is_identifiable_on_the_axis():
    """It is the number every other value is anchored to, so it is shown rather than inferred."""
    anchor = 63_142.87
    svg = dash._svg(_fc_at_price(anchor))
    root = ET.fromstring(svg)
    bold = [
        el.text
        for el in root.iter("{http://www.w3.org/2000/svg}text")
        if "700" in (el.get("style") or "") and el.get("text-anchor") == "end"
    ]
    assert bold == [f"{anchor:,.0f}"], f"the last close is not the bold tick: {bold}"


@pytest.mark.parametrize("anchor", [63_142.87, 0.4213, 0.00001053])
def test_the_inset_stays_in_PERCENT_at_every_magnitude(anchor):
    """Two panels, two units, each labelled — they measure DIFFERENT THINGS (an absolute price
    level and a relative move), which is not the two-estimators-of-one-thing defect."""
    svg = dash._svg(_fc_at_price(anchor))
    inset = _rect(svg, "inset")
    top, bot = float(inset.get("y")), float(inset.get("y")) + float(inset.get("height"))
    labels = [
        t for _x, y, _px, _m, t, _a in _text_nodes(svg) if top <= y <= bot and t.endswith("%")
    ]
    assert len(labels) >= 3, labels
    assert all("," not in t for t in labels), f"the inset is showing prices, not percent: {labels}"
