"""THIRTEEN SILENT WRONG ANSWERS FROM A STRANGER'S CSV, AND NOT ONE TEST FOR ANY OF THEM.

A *silent wrong answer* is confident, plausible output from malformed input with no warning. The
demo took a file it had never seen and produced a forecast for nine distinct classes of broken
input — a nanosecond timestamp read as microseconds, a timezone offset truncated away, a `price`
column shadowing `close`, duplicated rows sliding under a row floor that was checked before dedup,
a negative price rendered as a forecast of -93.79, swapped high/low columns, volume silently
fabricated from zeros into 2 of the 7 model-visible dims, a dead-flat series answered with a
CONFIDENT band, and Excel's default BOM refusing a file that had every column it needed.

The measured sting is that corruption made the model look MORE certain, not less: the audit's
corrupted file returned +0.333% with its whole band above zero, against +0.079% and a band
straddling zero on the clean one. A user reading the confident number has no way to know.

EVERY CASE HERE IS PAIRED. The malformed input must be refused or warned about, AND the clean
equivalent must still pass — otherwise a parser that rejected everything would score full marks.
Where the old behaviour was a pure function it is reproduced inline as a negative control, so the
defect is demonstrated rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from trikaal.demo import csv_forecast as cf

T0 = 1_700_000_000_000
HDR = "timestamp,open,high,low,close,volume"


def _bars(n: int = 1400, *, ts=None, vol: str = "1.0", flat: bool = False) -> list[str]:
    out = []
    for i in range(n):
        p = 100.0 if flat else 100.0 + float(np.sin(i / 50.0)) * 2.0
        stamp = ts(i) if ts else str(T0 + i * 60_000)
        out.append(f"{stamp},{p},{p + 0.5},{p - 0.5},{p},{vol}")
    return out


def _csv(rows: list[str], header: str = HDR, bom: bool = False) -> str:
    return ("﻿" if bom else "") + header + "\n" + "\n".join(rows) + "\n"


# ── fixture discrimination: the clean file must parse ─────────────────────────────────────────
def test_a_clean_file_parses() -> None:
    """If this failed, every refusal below would be meaningless."""
    b = cf.parse_csv(_csv(_bars()))
    assert b.ts_ms.shape[0] == 1400
    assert not [w for w in b.warnings if w.startswith(("VOLUME", "NO VOLUME", "NEAR-CONSTANT"))]


# ── (1) nanosecond epochs ─────────────────────────────────────────────────────────────────────
def test_nanosecond_timestamps_convert_to_milliseconds() -> None:
    ns = T0 * 1_000_000
    assert cf._to_ms(str(float(ns))) == T0


def test_NEGATIVE_CONTROL_the_old_nanosecond_rule_gave_microseconds() -> None:
    """`DatetimeIndex.astype("int64")` is nanoseconds — the most common way a Python user makes an
    integer timestamp column, and it was off by 1,000×."""
    ns = float(T0 * 1_000_000)
    assert int(ns / 1000) == T0 * 1000, "the old expression no longer reproduces"


@pytest.mark.parametrize(
    ("value", "want"), [(T0 / 1000, T0), (T0, T0), (T0 * 1000, T0), (T0 * 1_000_000, T0)]
)
def test_every_epoch_magnitude_lands_on_the_same_instant(value, want) -> None:
    assert cf._to_ms(str(float(value))) == want


# ── (3) timezone offsets ──────────────────────────────────────────────────────────────────────
def test_a_timezone_offset_is_honoured_not_truncated() -> None:
    z = cf._to_ms("2023-11-14T22:13:20Z")
    minus5 = cf._to_ms("2023-11-14T22:13:20-05:00")
    assert minus5 - z == 5 * 3_600_000, (minus5, z)


def test_NEGATIVE_CONTROL_truncating_to_19_chars_loses_the_offset() -> None:
    assert "2023-11-14T22:13:20-05:00"[:19] == "2023-11-14T22:13:20"


def test_a_naive_stamp_is_still_read_as_utc() -> None:
    assert cf._to_ms("2023-11-14 22:13:20") == cf._to_ms("2023-11-14T22:13:20Z")


# ── (9) the Excel BOM ─────────────────────────────────────────────────────────────────────────
def test_a_utf8_bom_does_not_hide_the_first_column() -> None:
    assert cf.parse_csv(_csv(_bars(), bom=True)).ts_ms.shape[0] == 1400


# ── (4) alias precedence ──────────────────────────────────────────────────────────────────────
def test_the_canonical_close_column_wins_over_the_price_alias() -> None:
    rows = [
        f"{T0 + i * 60_000},{100.0 + i * 0.01},999.5,99.0,999.0,{100.0 + i * 0.01}"
        for i in range(1400)
    ]
    b = cf.parse_csv(_csv(rows, "timestamp,open,high,low,price,close"))
    assert b.close[0] == pytest.approx(100.0), "the `price` column shadowed `close`"


# ── (2) the row floor after dedup ─────────────────────────────────────────────────────────────
def test_duplicates_cannot_slide_a_short_file_under_the_row_floor() -> None:
    rows = _bars(700) + _bars(700)  # 1,400 lines, 700 distinct bars
    assert len(rows) >= cf.MIN_ROWS, "fixture must clear the pre-dedup floor to be a valid probe"
    with pytest.raises(cf.CsvRefused, match="duplicate or out-of-order"):
        cf.parse_csv(_csv(rows))


def test_a_file_that_is_genuinely_long_enough_still_passes() -> None:
    assert cf.parse_csv(_csv(_bars(cf.MIN_ROWS))).ts_ms.shape[0] == cf.MIN_ROWS


# ── (6) negative prices ───────────────────────────────────────────────────────────────────────
def test_a_negative_price_is_refused_and_the_row_is_named() -> None:
    rows = _bars()
    rows[5] = f"{T0 + 5 * 60_000},-100.0,-99.5,-100.5,-100.0,1.0"
    with pytest.raises(cf.CsvRefused, match="non-positive") as e:
        cf.parse_csv(_csv(rows))
    assert "file line 7" in str(e.value), str(e.value)  # header + 0-based row 5


def test_a_zero_price_is_refused_too() -> None:
    rows = _bars()
    rows[9] = f"{T0 + 9 * 60_000},0,0,0,0,1.0"
    with pytest.raises(cf.CsvRefused, match="non-positive"):
        cf.parse_csv(_csv(rows))


# ── (7) swapped high/low ──────────────────────────────────────────────────────────────────────
def test_swapped_high_and_low_columns_are_detected() -> None:
    rows = [
        f"{T0 + i * 60_000},100.0,99.5,100.5,100.0,1.0"  # high < low
        for i in range(1400)
    ]
    with pytest.raises(cf.CsvRefused, match="swapped high/low"):
        cf.parse_csv(_csv(rows))


def test_a_high_below_the_close_is_detected() -> None:
    rows = _bars()
    rows[3] = f"{T0 + 3 * 60_000},100.0,100.0,99.0,101.0,1.0"  # close above high
    with pytest.raises(cf.CsvRefused, match="not a valid bar"):
        cf.parse_csv(_csv(rows))


# ── (5) fabricated volume ─────────────────────────────────────────────────────────────────────
def test_all_zero_volume_says_two_dimensions_carry_no_information() -> None:
    b = cf.parse_csv(_csv(_bars(vol="0")))
    assert any("VOLUME IS ZERO" in w for w in b.warnings), b.warnings
    assert any("2 of the 7" in w for w in b.warnings)


def test_an_absent_volume_column_says_the_dimensions_are_fabricated() -> None:
    rows = [r.rsplit(",", 1)[0] for r in _bars()]
    b = cf.parse_csv(_csv(rows, "timestamp,open,high,low,close"))
    assert any("NO VOLUME COLUMN" in w for w in b.warnings), b.warnings
    assert any("FABRICATED" in w for w in b.warnings)


def test_unparseable_volume_is_named_rather_than_zero_filled_in_silence() -> None:
    rows = _bars()
    rows[7] = rows[7].rsplit(",", 1)[0] + ",n/a"
    b = cf.parse_csv(_csv(rows))
    assert any("UNPARSEABLE VOLUME" in w for w in b.warnings), b.warnings


# ── (8) a dead-flat series ────────────────────────────────────────────────────────────────────
def test_a_constant_series_is_refused_rather_than_answered_confidently() -> None:
    with pytest.raises(cf.CsvRefused, match="constant"):
        cf.parse_csv(_csv(_bars(flat=True)))


def test_a_near_constant_series_is_warned_about() -> None:
    rows = []
    for i in range(1400):
        p = 100.0 + (i % 3) * 0.01  # three distinct prices
        rows.append(f"{T0 + i * 60_000},{p},{p + 0.5},{p - 0.5},{p},1.0")
    b = cf.parse_csv(_csv(rows))
    assert any("NEAR-CONSTANT" in w for w in b.warnings), b.warnings


# ── every refusal is a reason, never a traceback ──────────────────────────────────────────────
@pytest.mark.parametrize(
    "rows",
    [
        _bars(700) + _bars(700),
        [f"{T0 + i * 60_000},-1,-1,-1,-1,1" for i in range(1400)],
        [f"{T0 + i * 60_000},100,99,101,100,1" for i in range(1400)],
        _bars(flat=True),
    ],
    ids=["duplicates", "negative", "inverted", "constant"],
)
def test_every_refusal_is_a_csvrefused_with_an_explanation(rows) -> None:
    with pytest.raises(cf.CsvRefused) as e:
        cf.parse_csv(_csv(rows))
    msg = str(e.value)
    assert len(msg) > 80, f"refusal is too terse to act on: {msg}"
    assert msg[0].islower() or msg[0].isdigit() or msg[0].isupper()
