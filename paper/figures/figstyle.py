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

# ---- THE TYPE LADDER (one ladder, every figure; never set a size not on it) -----------------
# Five steps, each ~15% apart, so a reader's eye reads hierarchy rather than noise. Figures that
# invented their own sizes are what made the set look like ten separate efforts.
T_FINDING = 7.4  # the one-line finding carried IN the raster
T_TITLE = 7.0  # panel titles
T_LABEL = 7.4  # axis labels
T_TICK = 6.6  # tick labels
T_ANNOT = 6.2  # in-plot annotation
T_MICRO = 5.6  # dense row labels, legends, value columns

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


def panel(ax, letter: str, *, dx: float = -0.055, dy: float = 1.035) -> None:
    """Panel label in the one place every figure puts it: above-left of the axes."""
    ax.text(
        dx,
        dy,
        f"({letter})",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=T_TITLE,
        weight="bold",
        color=INK,
        gid="panel",
    )


def panel_title(ax, letter: str, text: str, *, pad: float = 4.0) -> None:
    """Panel letter and title as ONE left-aligned string.

    A separate floating "(a)" above-left collides with a left-aligned title on any narrow panel,
    and the collision is invisible to a checker that only reads ax.texts -- a title is ax.title.
    Folding the letter into the title removes the collision class instead of tuning around it.
    """
    ax.set_title(f"({letter})  {text}", loc="left", fontsize=T_TITLE, pad=pad, color=INK)


def finding(fig, text: str, *, y: float = 0.995, color: str | None = None) -> None:
    """The ONE thing the figure shows, written INTO the raster.

    Context-stripping rule: a figure travels without its caption, so the finding cannot live only
    in the caption. This is the line a cropped panel must still carry.
    """
    fig.text(
        0.0,
        y,
        text,
        ha="left",
        va="top",
        fontsize=T_FINDING,
        weight="bold",
        color=color or INK,
    )


def scope(fig, text: str, *, y: float = 0.005) -> None:
    """The qualifier that must survive a crop — scope, arm, sample size, what this is NOT."""
    fig.text(0.0, y, text, ha="left", va="bottom", fontsize=T_MICRO, color=RULE, style="italic")


def minus(text: str) -> str:
    """Typographic minus, so figure numerals match the manuscript."""
    return text.replace("-", "−")


def assert_text_legible(fig, axes, tol: float = 1.0) -> None:
    """Measure EVERY text node on the RENDERED figure: nothing clipped, nothing overlapping.

    The render rule's corollary: figure geometry is coupled, and my own arithmetic is what
    produced the layout, so re-deriving the layout from it is the shared-input failure with both
    arms in one head. Matplotlib's renderer is the independent second arm -- the analogue of
    getBBox on an SVG.

    THIS CHECK HAS BEEN EXTENDED FOUR TIMES, EACH TIME BECAUSE A RENDER WAS LOOKED AT AND SHOWED
    SOMETHING IT MISSED. That history is the argument for looking, and it is kept here so nobody
    mistakes a green result for coverage:
        v1  containment only          -> passed a legend sitting on top of a data row
        v2  + overlap, per axes       -> passed a scope line running 232px off the canvas
        v3  + figure-level text       -> passed a panel label overlapping its own title
        v4  + titles and axis labels,
            and overlap made GLOBAL   -> a title spilling into the NEXT panel was invisible while
                                         overlap was computed one axes at a time
    Containment is measured against the axes for in-plot text and against the canvas for the
    things that belong outside it (titles, axis labels, panel letters, figure-level lines).

    WHAT "OVERFLOWS THE CANVAS" ACTUALLY COSTS, stated precisely rather than dramatically: with
    savefig.bbox="tight" such text is NOT clipped -- it widens the saved bounding box. The figure
    is then scaled down to fit \textwidth, so its type renders SMALLER than its siblings'. That is
    the mechanism behind a figure set that looks like separate efforts, and it is why this is an
    error rather than a warning.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fbox = fig.get_window_extent(renderer=r)
    bad: list[str] = []
    everything: list[tuple] = []  # (artist, extent) across ALL axes and the figure

    for t in fig.texts:
        if not t.get_text().strip() or t.get_gid() == "overlay":
            continue
        e = t.get_window_extent(renderer=r)
        over = max(fbox.x0 - e.x0, e.x1 - fbox.x1, fbox.y0 - e.y0, e.y1 - fbox.y1)
        if over > tol:
            bad.append(f"figure-level {t.get_text()[:45]!r} runs off the canvas by {over:.1f}px")
        everything.append((t, e))

    for ax in axes:
        box = ax.get_window_extent(renderer=r)
        # ax.title is the CENTRE title and is EMPTY whenever loc="left" is used -- which is every
        # panel title in this paper. Reading only ax.title made "the check covers titles" a check
        # that could not fail. Matplotlib keeps three title artists; take all of them.
        outside = [
            ax.title,
            getattr(ax, "_left_title", None),
            getattr(ax, "_right_title", None),
            ax.xaxis.label,
            ax.yaxis.label,
        ]
        outside = [t for t in outside if t is not None]
        for t in outside:
            t.set_gid("panel")
        items = [
            t for t in list(ax.texts) + outside if t.get_text().strip() and t.get_gid() != "overlay"
        ]
        for t in items:
            e = t.get_window_extent(renderer=r)
            ref = fbox if t.get_gid() == "panel" else box
            over = max(ref.x0 - e.x0, e.x1 - ref.x1, ref.y0 - e.y0, e.y1 - ref.y1)
            if over > tol:
                where = "the canvas" if t.get_gid() == "panel" else "its axes"
                bad.append(f"{t.get_text()[:40]!r} overflows {where} by {over:.1f}px")
            everything.append((t, e))

    # ONE global overlap pass. Per-axes was the bug: a title that spills into the neighbouring
    # panel overlaps text belonging to a DIFFERENT axes, and no per-axes loop can see it.
    for i, (a, ea) in enumerate(everything):
        for b, eb in everything[i + 1 :]:
            ox = min(ea.x1, eb.x1) - max(ea.x0, eb.x0)
            oy = min(ea.y1, eb.y1) - max(ea.y0, eb.y0)
            if ox > tol and oy > tol:
                bad.append(
                    f"{a.get_text()[:30]!r} overlaps {b.get_text()[:30]!r} by {ox:.1f}x{oy:.1f}px"
                )

    if bad:
        raise AssertionError("figure text is clipped or overlapping:\n  " + "\n  ".join(bad))
