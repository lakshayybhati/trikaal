"""The demo's lake pin must check WHAT it reads, not only WHERE it reads from.

THE HISTORICAL DEFECT, and why the fix for it was incomplete. ``trikaal.data.lake.connect``
defaults to ``processed/bars`` — ONE symbol, 2023 only, 525,600 bars — while the money run scored
``processed/universe_bars``, 200 symbols and 2,103,840 bars each. The demo inherited the default,
served forecasts computed on a 4x shorter normalization history, and the acceptance gate passed
9/9 the whole time because BOTH of its arms called the same ``connect()``. An agreement test
cannot detect an error its two arms share.

``LAKE_ROOT`` closed that by pinning the path. It did not close the class: the only thing asserting
the lake's CONTENTS was a fail-loud "no bars" error, which fires when a symbol is absent and stays
silent when the path is right and the data is wrong. A wrong lake at the pinned path would still
have passed the agreement gate, for exactly the same reason as before — shared input.

WHAT THESE TESTS ASSERT. That ``verify_lake_identity`` accepts the real lake, and — the half that
makes it a check rather than a restatement — that it REJECTS each way of being wrong: too few
partitions (the original defect), an absent manifest, a manifest whose counts or Merkle differ,
and a manifest that describes a DIFFERENT lake than the tree it sits beside. Every negative case
below is constructed from the real manifest so it cannot pass by being malformed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from trikaal.demo.inference import (
    LAKE_ROOT,
    PINNED_LAKE_BARS,
    PINNED_LAKE_MERKLE,
    PINNED_LAKE_SYMBOLS,
    LakeIdentityError,
    verify_lake_identity,
)

REPO = Path(__file__).resolve().parents[2]
REAL_LAKE = REPO / LAKE_ROOT
REAL_MANIFEST = REPO / "processed/universe/universe_manifest.json"

needs_lake = pytest.mark.skipif(
    not (REAL_LAKE.is_dir() and REAL_MANIFEST.exists()),
    reason="the 15 GB universe lake is not present on this machine",
)


def _fake_lake(tmp: Path, doc: dict, symbols: list[str]) -> tuple[Path, Path]:
    root = tmp / "universe_bars"
    for s in symbols:
        (root / f"symbol={s}").mkdir(parents=True, exist_ok=True)
    mf = tmp / "universe_manifest.json"
    mf.write_text(json.dumps(doc))
    return root, mf


@pytest.fixture(scope="module")
def real_doc() -> dict:
    if not REAL_MANIFEST.exists():
        pytest.skip("no universe manifest on this machine")
    return json.loads(REAL_MANIFEST.read_text())


@needs_lake
def test_the_real_lake_passes() -> None:
    got = verify_lake_identity(REAL_LAKE, manifest=REAL_MANIFEST)
    assert got["n_symbol_partitions"] == PINNED_LAKE_SYMBOLS
    assert got["total_bars"] == PINNED_LAKE_BARS
    assert got["universe_merkle"] == PINNED_LAKE_MERKLE


@needs_lake
def test_the_single_symbol_lake_is_rejected(tmp_path, real_doc) -> None:
    """THE ORIGINAL DEFECT: processed/bars has one symbol and used to be served silently."""
    root, mf = _fake_lake(tmp_path, real_doc, ["BTCUSDT"])
    with pytest.raises(LakeIdentityError, match="1 symbol partitions"):
        verify_lake_identity(root, manifest=mf)


@needs_lake
def test_a_missing_manifest_is_rejected(tmp_path, real_doc) -> None:
    names = [s["symbol"] for s in real_doc["symbols"]]
    root, mf = _fake_lake(tmp_path, real_doc, names)
    mf.unlink()
    with pytest.raises(LakeIdentityError, match="lake identity cannot be checked"):
        verify_lake_identity(root, manifest=mf)


@needs_lake
@pytest.mark.parametrize(
    "field,value",
    [
        ("total_bars", PINNED_LAKE_BARS - 1),
        ("n_symbols", PINNED_LAKE_SYMBOLS + 1),
        ("universe_hash", "sha256:" + "0" * 64),
    ],
)
def test_a_lake_with_the_right_shape_and_wrong_identity_is_rejected(
    tmp_path, real_doc, field, value
) -> None:
    """200 partitions is NOT enough — a different universe of the same size must still fail."""
    doc = {**real_doc, field: value}
    names = [s["symbol"] for s in real_doc["symbols"]]
    root, mf = _fake_lake(tmp_path, doc, names)
    with pytest.raises(LakeIdentityError, match="lake identity"):
        verify_lake_identity(root, manifest=mf)


@needs_lake
def test_a_manifest_describing_another_lake_is_rejected(tmp_path, real_doc) -> None:
    """The cross-check: right counts, right Merkle, and the tree holds different instruments."""
    names = [s["symbol"] for s in real_doc["symbols"]]
    swapped = [f"NOTREAL{i}USDT" for i in range(3)] + names[3:]
    root, mf = _fake_lake(tmp_path, real_doc, swapped)
    with pytest.raises(LakeIdentityError, match="no partition under"):
        verify_lake_identity(root, manifest=mf)
