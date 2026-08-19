"""The model card's numbers must equal the artifacts they claim to report.

WHY THIS FILE EXISTS. An independent audit of the published repository found **four** wrong
numbers on this one page, and every one of them was checkable against a receipt already in the
tree:

  * the training data was given as the whole 200-symbol lake when the draw is 40 symbols — a **5x**
    overstatement, and the correct figure was already registered as risk 7 in ``BUILD_RECORD``;
  * ``97.3%`` was published unqualified when it is one seed of cell 4, whose weights are not
    released;
  * the magnitude-dim range read ``0.8975-0.9320`` when the measured minimum is ``0.8962`` — wrong
    in the flattering direction, making the signed-vs-magnitude contrast look cleaner than it is;
  * the provenance sentence listed five identity keys the units do not carry.

Four independent slips on one page is not four mistakes, it is a missing guard. Prose drifts from
its receipts silently and only a reader who re-derives every figure notices. So the card is pinned
to the artifacts here, and a number that moves in either place fails the suite.

SCOPE, STATED HONESTLY. Everything asserted below is read from a **tracked** receipt under
``runs_manifest/``. The live unit manifests live under gitignored ``runs_cloud/``; where a figure
originates there, the tracked release manifest carries it and a separate test re-derives that
mirror against the live units when they are present. Nothing here skips: a check that goes quiet
when its input disappears is worse than no check.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "docs/MODEL_CARD.md"
LEGIBILITY = REPO / "runs_manifest/m6_micro_legibility_stop.json"
RELEASE = REPO / "runs_manifest/m6_weights_release.json"
SWEEP = REPO / "runs_manifest/m6_lambda_sweep.json"

# The card renders ranges with an EN DASH. Matching it needs the same character, and a bare
# literal trips ruff's ambiguous-unicode rule — so it is named once rather than noqa'd twice.
EN_DASH = "\u2013"

GATE_THRESHOLD = 0.90
SIGNED_DIMS = (7, 8)
MAGNITUDE_DIMS = (9, 10, 11, 12)


@pytest.fixture(scope="module")
def card() -> str:
    return CARD.read_text()


@pytest.fixture(scope="module")
def per_dim() -> dict[int, float]:
    d = json.loads(LEGIBILITY.read_text())
    raw = d["legibility_receipt"]["cell4_fsq_micro_seed0"]["per_dim"]
    return {int(k): float(v["sign_acc"]) for k, v in raw.items()}


@pytest.fixture(scope="module")
def release() -> dict:
    return json.loads(RELEASE.read_text())


def _shortfalls(per_dim: dict[int, float]) -> dict[int, float]:
    return {k: max(0.0, GATE_THRESHOLD - v) for k, v in per_dim.items()}


# ── the inputs are real, not empty ────────────────────────────────────────────────────────────
def test_the_receipt_is_populated(per_dim) -> None:
    """Refuse rather than pass vacuously if the receipt loses its per-dim block."""
    assert set(per_dim) == set(SIGNED_DIMS) | set(MAGNITUDE_DIMS), per_dim
    assert all(0.0 < v < 1.0 for v in per_dim.values()), per_dim


# ── T1.1 — the draw, not the lake ─────────────────────────────────────────────────────────────
def test_the_card_states_the_draw_and_the_draw_matches_the_release_manifest(card, release) -> None:
    draw = release["training_draw"]
    m = re.search(r"\*\*Training draw\*\* \| \*\*(\d+) symbols\*\*", card)
    assert m, "the card no longer has a Training draw row"
    assert int(m.group(1)) == draw["n_symbols"] == 40

    # THE DISTINCTION THIS GUARD ALREADY CAUGHT ONCE, in the very correction that created this
    # file: windows x seq_len is TRAINING THROUGHPUT (bar-POSITIONS consumed), not the size of the
    # draw. Conflating them restates a 5.06x pass count as if it were 5.06x the data.
    assert draw["n_windows"] * draw["seq_len"] == draw["n_bar_positions_consumed"], draw
    assert draw["n_bar_positions_consumed"] != draw["n_bars"], draw
    assert draw["n_bar_positions_consumed"] / draw["n_bars"] == pytest.approx(
        draw["passes_over_the_draw"]
    ), draw
    assert f"{draw['n_bars']:,}" in card, f"the card does not state {draw['n_bars']:,} bars"
    assert f"{draw['n_windows']:,}" in card
    assert f"{draw['n_bar_positions_consumed']:,}" in card, (
        "the card states windows and bars but not the bar-positions they multiply to, which is "
        "where the two get conflated"
    )


def test_the_card_does_not_present_the_lake_as_the_training_data(card) -> None:
    """The 5x overstatement, in the exact shape it had: the lake bar count sold as what it saw."""
    lake_row = re.search(r"\| \*\*Source corpus\*\* \|([^|]*)\|", card)
    assert lake_row, "the card no longer separates the source corpus from the draw"
    assert "304,625,181" in lake_row.group(1), "the lake belongs on the corpus row"
    training_row = re.search(r"\| \*\*Training draw\*\* \|([^|]*)\|", card)
    assert "304,625,181" not in training_row.group(1), (
        "the 200-symbol lake is back on the training row — that is the 5x overstatement"
    )


# ── T1.6 — the holdout boundary ───────────────────────────────────────────────────────────────
def test_the_card_states_the_holdout_boundary(card, release) -> None:
    """A reader asking 'can I backtest 2024?' must not be told the whole window is in-sample."""
    draw = release["training_draw"]
    boundary = draw["train_through"][:10]
    assert boundary == "2023-10-20", draw
    assert boundary in card, f"the card does not name the split date {boundary}"
    assert re.search(r"held out|out-of-sample", card, re.I), "the card never says what is held out"


def test_the_boundary_is_what_the_pinned_train_frac_implies(release) -> None:
    """Re-derive it rather than trust the string: 0.7 of the pinned window, to the minute."""
    import datetime as dt

    from trikaal.demo.inference import PINNED_TRAIN_FRAC, PINNED_WINDOW

    draw = release["training_draw"]
    assert [*PINNED_WINDOW] == draw["window"]
    assert PINNED_TRAIN_FRAC == draw["train_frac"]
    start = dt.datetime.fromisoformat(PINNED_WINDOW[0]).replace(tzinfo=dt.UTC)
    end = dt.datetime.fromisoformat(PINNED_WINDOW[1]).replace(tzinfo=dt.UTC)
    n = int((end - start).total_seconds() // 60)
    derived = start + dt.timedelta(minutes=int(n * PINNED_TRAIN_FRAC))
    assert derived == dt.datetime.fromisoformat(draw["train_through"]), derived


# ── T1.2 — 97.3% is cell 4, one seed, and it must recompute ───────────────────────────────────
def test_the_signed_share_recomputes_to_the_published_number(per_dim, card) -> None:
    sh = _shortfalls(per_dim)
    total = sum(sh.values())
    signed = sum(sh[d] for d in SIGNED_DIMS)
    assert signed / total == pytest.approx(0.9728, abs=5e-5), signed / total
    assert "97.3%" in card


def test_every_published_97_3_names_cell_4(card) -> None:
    """Unqualified, it reads as a property of the released weights. It is not — cell 4 is not
    released, so a reader cannot check it without being told which cell it belongs to.

    A PROXIMITY CHECK IS NOT ENOUGH AND THIS TEST USED TO BE ONE. Asking whether "cell 4" appears
    within N characters passed when the scope was stripped from the sentence itself, because an
    unrelated neighbouring sentence mentioned cell 4. Mutation testing caught it. The check is now
    on the CLAIM FORM: the unqualified phrasings are named and forbidden outright.
    """
    forbidden = (
        r"97\.3% of the shortfall",
        r"97\.3% of (?:its |our )?total shortfall",
        r"carry 97\.3%(?! of \*?cell)",
    )
    for pat in forbidden:
        m = re.search(pat, card, re.I)
        assert m is None, f"unqualified 97.3% claim: ...{card[m.start() - 80 : m.end() + 80]}..."

    # and the scoped form must actually be present, so deleting the claim is not a pass either
    assert re.search(r"97\.3% of \*?cell ?4", card, re.I), (
        "the scoped claim is gone entirely — this test must not pass by absence"
    )


def test_the_card_carries_the_replicate_spread(card) -> None:
    """One seed of one cell is not a stable statistic and the card must say so with numbers."""
    cfgs = json.loads(SWEEP.read_text())["configs"]
    shares = []
    for c in cfgs.values():
        sh = _shortfalls({int(k): float(v["sign_acc"]) for k, v in c["per_dim"].items()})
        shares.append(sum(sh[d] for d in SIGNED_DIMS) / sum(sh.values()))
    assert len(shares) == 18, len(shares)
    lo, hi = round(100 * min(shares)), round(100 * max(shares))
    assert (lo, hi) == (77, 93), (lo, hi)
    # BUILT FROM THE RECEIPT, NOT RESTATED. `"77" in card` was the first version of this line and
    # it passed with the spread deleted, because those digits occur elsewhere on the page.
    assert f"**{lo}% and {hi}%**" in card, (
        f"the card does not carry the replicate spread as measured ({lo}%-{hi}%)"
    )
    assert str(len(shares)) in card and "m6_lambda_sweep.json" in card, (
        "the spread is stated without saying how many replicates it came from, or from where"
    )


# ── T1.4 — the magnitude range, wrong in the flattering direction ─────────────────────────────
def test_the_magnitude_range_is_the_measured_min_and_max(per_dim, card) -> None:
    lo = min(per_dim[d] for d in MAGNITUDE_DIMS)
    hi = max(per_dim[d] for d in MAGNITUDE_DIMS)
    assert (lo, hi) == (0.8962, 0.9320), (lo, hi)
    assert f"{lo:.4f}{EN_DASH}{hi:.4f}" in card, f"the card omits the range {lo:.4f}-{hi:.4f}"
    assert f"0.8975{EN_DASH}0.9320" not in card, (
        "the superseded range is back: 0.8975 is dim 9's value, not the minimum (dim 11 is 0.8962)"
    )


def test_each_per_dim_value_on_the_card_matches_the_receipt(card, per_dim) -> None:
    """The card prints a per-dim table so a reader can recompute. Every row must be the receipt."""
    rows = re.findall(r"^\| (\d+) \| `(\w+)` \| \*{0,2}\w+\*{0,2} \| ([\d.]+) \|", card, re.M)
    assert len(rows) == 6, f"expected 6 per-dim rows, found {len(rows)}"
    for dim, _name, acc in rows:
        assert float(acc) == per_dim[int(dim)], (dim, acc, per_dim[int(dim)])


# ── T1.3 — the provenance list must equal what the units actually stamp ───────────────────────
def test_the_card_names_exactly_the_four_uncaptured_identity_keys(card) -> None:
    """The card claimed five keys the units do not carry. It now names the gap instead, and the
    gap is asserted against the code's own identity set rather than restated."""
    from trikaal.utils.provenance import PROVENANCE_IDENTITY_KEYS

    assert len(PROVENANCE_IDENTITY_KEYS) == 16, PROVENANCE_IDENTITY_KEYS
    missing = {"image", "git_commit", "lockfile_sha256", "platform_abi"}
    assert missing < set(PROVENANCE_IDENTITY_KEYS)
    for key in missing:
        assert f"`{key}`" in card, f"the card does not disclose the missing key {key}"
    assert "12" in card and "16" in card, "the card no longer states 12 of 16"
    assert "unavailable: AttributeError" in card, (
        "the card no longer discloses that driver_version holds a placeholder"
    )
