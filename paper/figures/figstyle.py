"""Shared figure vocabulary for the Trikaal paper — one palette, one font, every figure.

RENDERING ONLY — not part of the anchored instrument; no measurement is produced here.

THE PALETTE IS PERMANENT. Colours are assigned to *meanings*, once, and are never reassigned
between figures. A reader who learns the vocabulary in Figure 2 can read Figure 9 without a
legend. Every colour is from Paul Tol's qualitative schemes and the set is checked to stay
separable under deuteranopia, protanopia and tritanopia.

    ARM AND CHANNEL COLOURS (semantic identity — never reused for anything else)
    ------------------------------------------------------------------------------------
    OHLCV    #4477AA  blue     price / OHLC-shape / volume channels; the OHLCV-only input arm
    MICRO    #999933  olive    the six microstructure channels; the +microstructure input arm
    FSQ      #44AA99  teal     the finite-scalar-quantizer arm
    BSQ      #882255  wine     the binary-spherical-quantizer arm
    PLACEBO  #999999  grey     the shuffled-microstructure placebo cell

    STATUS COLOURS (condition, not identity)
    ------------------------------------------------------------------------------------
    PASS     #117733  green    a gate met, a control behaving, a condition satisfied
    FAIL     #CC3311  red      a gate failed, a channel evicted, a control violated

    FIXTURE-ONLY ROLE (appears only in synthetic-fixture figures, never beside MICRO)
    ------------------------------------------------------------------------------------
    FILLER   #DDCC77  sand     the correlated non-signal block of the synthetic fixture

    STRUCTURE
    ------------------------------------------------------------------------------------
    INK      #222222  body text and axes
    RULE     #666666  thresholds, reference lines, annotation
    GRID     #DDDDDD  gridlines and light fills

RULE: status is always encoded redundantly — colour plus glyph, position or label — so a
figure never depends on colour alone to communicate pass versus fail.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- arm and channel identity (permanent) --------------------------------------------------
OHLCV = "#4477AA"
MICRO = "#999933"
FSQ = "#44AA99"
BSQ = "#882255"
PLACEBO = "#999999"

# ---- status (permanent) --------------------------------------------------------------------
PASS = "#117733"
FAIL = "#CC3311"

# ---- fixture-only role ---------------------------------------------------------------------
FILLER = "#DDCC77"

# ---- structure -----------------------------------------------------------------------------
INK = "#222222"
RULE = "#666666"
GRID = "#DDDDDD"

# ---- geometry ------------------------------------------------------------------------------
SINGLE_COL = 5.5  # in — text width of the preprint layout
HALF_COL = 2.65

# ---- deprecated aliases (kept so older scripts keep rendering; do not use in new figures) ---
BLUE, OLIVE, RED, GREY, PALE = OHLCV, MICRO, FAIL, RULE, GRID


def apply() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "figure.dpi": 200,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, out_dir, stem: str) -> None:
    """Vector PDF for the manuscript; PNG only for on-screen review."""
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


def minus(text: str) -> str:
    """Typographic minus, so figure numerals match the manuscript."""
    return text.replace("-", "−")
