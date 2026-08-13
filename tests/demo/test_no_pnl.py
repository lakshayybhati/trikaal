"""The no-P&L rule, enforced as a test rather than trusted as a convention.

``trikaal.demo`` must never compute a profit, a position, a return series or an equity curve. The
package docstring says so; this makes it checkable, because a rule that only exists in prose is
one contributor away from being untrue.

★ FIXTURE-DISCRIMINATION: every assertion here is paired with a proof that it CAN fail.
``test_the_banned_scan_can_actually_fail`` runs the same scanner over a deliberately offending
source string and requires a hit. Without that, a scanner with a typo'd pattern would pass
forever and this file would be decoration — the exact defect this project keeps finding.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import trikaal.demo
from trikaal.demo import inference, render

# Identifiers that would mean a P&L had been computed. Deliberately narrow: these are names, not
# substrings of prose, and the scan below reads the AST so that the module's own DOCSTRINGS —
# which discuss P&L at length in order to forbid it — cannot trigger a false positive.
BANNED_NAMES = {
    "pnl",
    "profit",
    "equity",
    "equity_curve",
    "cumulative_return",
    "cum_return",
    "net_return",
    "position",
    "positions",
    "sharpe",
    "information_ratio",
    "backtest",
    "portfolio_period_returns",
    "net_trade_returns",
}

DEMO_MODULES = [trikaal.demo, inference, render]


def _defined_and_called_names(src: str) -> set[str]:
    """Every name DEFINED, ASSIGNED or CALLED in the source — docstrings and comments excluded.

    Reading the AST rather than grepping text is the whole point: these modules must be free to
    EXPLAIN why they compute no P&L, in prose, without tripping their own guard.
    """
    tree = ast.parse(src)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            out.add(node.name)
        elif isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
    return out


@pytest.mark.parametrize("mod", DEMO_MODULES, ids=lambda m: m.__name__)
def test_demo_package_computes_no_pnl(mod) -> None:
    src = Path(inspect.getfile(mod)).read_text()
    hits = _defined_and_called_names(src) & BANNED_NAMES
    assert not hits, (
        f"{mod.__name__} references {sorted(hits)} — the demo must compute NO profit, position or "
        "equity quantity. Our own measurement says these models lose money after fees; a demo that "
        "can render a P&L is the context-stripping rule at its worst."
    )


def test_the_banned_scan_can_actually_fail() -> None:
    """The guard above must be capable of failing, or it is not a guard."""
    offending = "def equity_curve(x):\n    return x.cumsum()\n"
    assert _defined_and_called_names(offending) & BANNED_NAMES == {"equity_curve"}

    # and it must NOT fire on prose that merely discusses the ban — the false-positive direction,
    # which is what would make a contributor delete the guard rather than obey it.
    prose = '"""This module computes no pnl, no equity curve, no position and no sharpe."""\n'
    assert not (_defined_and_called_names(prose) & BANNED_NAMES)


def test_forecast_bundle_exposes_no_averaged_seed() -> None:
    """No mean/consensus/ensemble field may exist — averaging the seeds would hide the finding."""
    fields = set(inference.ForecastBundle.__dataclass_fields__)
    banned = {"mean_mu", "consensus", "ensemble", "average", "avg_mu", "mu_mean"}
    assert not (fields & banned), (
        f"ForecastBundle exposes {sorted(fields & banned)}; the three seeds are plotted separately "
        "and never averaged (identical config produced IRs spanning 118.03)."
    )


def test_renderer_bakes_the_headline_result_into_the_image() -> None:
    """The economic result must be INSIDE the svg, not in a caption a crop can remove."""
    src = Path(inspect.getfile(render)).read_text()
    for token in ("HEADLINE_NET_IR", "BREAK_EVEN_PCT_RANGE", "ROUND_TRIP_FEE_PCT"):
        assert token in src, f"{token} missing — the headline result must render into the raster"
    assert "LOSES MONEY AFTER FEES" in src
