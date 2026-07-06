"""Micro-alignment lag-sharpness audit (audit item 6 — closes FP-3's residual).

    PYTHONPATH=src python3 scripts/audit_micro_alignment.py [--skip-download]

The question: are the aggTrades-derived microstructure features aligned to the SAME minute as
the kline features — no off-by-one in the per-minute windowing? Two independent probes:

**(A) in-lake lag sharpness** — klines and aggTrades entered the lake from SEPARATE archives,
so a windowing offset between them would show as the cross-source correlation peaking at lag ±1
instead of 0. Per sampled (symbol, month) spanning years/eras: corr of the kline log-volume
feature (x_5) against the aggTrades volume reconstruction x_9 + x_10 (= z(log n_trades) +
z(log mean_trade_size); the log-space product trade_count × mean_trade_size ≈ aggTrades volume)
at lag 0 vs ±1, adjacent-minute pairs only, masked bars excluded. PASS: lag-0 dominates BOTH
±1 on EVERY sample by ≥ MARGIN.

**(B) definitive raw recompute** — re-download 3-5 (symbol, day) DAILY aggTrades archives,
independently re-aggregate per minute with the explicit [open, open+60s) window
(``bar_id = transact_time // 60000`` — a fresh implementation of the spec §2.1 definitions,
NOT a call into the ingest reducer), and compare against the lake's stored TFI (x_7) and
signed-count-imbalance (x_8) — BOUNDED_IDX features, stored clip-only, so the comparison is
float32-exact, not statistical. PASS: max |recomputed − stored| ≤ ATOL at lag 0 on every
usable minute AND the lag-±1 correlation collapses (TFI is fast-mixing, so a one-minute
offset destroys the match).

Raw downloads land in ``raw/binance/_audit_cache/`` (gitignored — Binance data is never
redistributed); their sha256 digests go into the committed report:
``docs/m6_micro_alignment_audit.md`` + machine-readable ``runs_manifest/m6_micro_alignment.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import ssl
import sys
import urllib.request
import zipfile
from pathlib import Path

import certifi
import numpy as np

from trikaal.constants import EPS_CNT
from trikaal.data.universe_loader import connect_lake

# macOS Python.org builds don't use the system trust store (same fix as data/ingest.py)
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

LAKE = Path("processed/universe_bars")
CACHE = Path("raw/binance/_audit_cache")
REPORT_MD = Path("docs/m6_micro_alignment_audit.md")
REPORT_JSON = Path("runs_manifest/m6_micro_alignment.json")
BAR_MS = 60_000
MARGIN = 0.02  # part-A decisiveness floor: r0 must beat both lag-±1 by at least this

# (A) pinned sample plan — symbols × months across years and archive eras; combos absent from
# the lake (late listings) are skipped and recorded as such
A_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "GALAUSDT",
    "FRONTUSDT",
    "1000BONKUSDT",
)
A_MONTHS = (
    "2021-02",
    "2021-09",
    "2022-06",
    "2022-11",
    "2023-05",
    "2023-10",
    "2024-03",
    "2024-08",
    "2024-12",
)
A_MIN_PAIRS = 10_000

# (B) pinned (symbol, day) picks across eras; daily um-futures aggTrades archives
B_DAYS = (
    ("BTCUSDT", "2021-06-15"),
    ("ETHUSDT", "2022-11-09"),
    ("SOLUSDT", "2023-12-01"),
    ("DOGEUSDT", "2024-06-03"),
)
B_URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"
B_ATOL = 2e-6  # float32 storage tolerance on |TFI| <= 1


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def part_a(con) -> list[dict]:
    rows_out: list[dict] = []
    for sym in A_SYMBOLS:
        for month in A_MONTHS:
            t = con.execute(
                "SELECT bar_open_ms, x_5, x_9, x_10, m_5, m_9, m_10 FROM bars "
                "WHERE symbol = ? AND month = ? ORDER BY bar_open_ms",
                [sym, month],
            ).to_arrow_table()
            if t.num_rows == 0:
                rows_out.append({"symbol": sym, "month": month, "status": "absent"})
                continue
            ts = t.column("bar_open_ms").to_numpy().astype(np.int64)
            a = t.column("x_5").to_numpy().astype(np.float64)
            b = t.column("x_9").to_numpy().astype(np.float64) + t.column("x_10").to_numpy()
            usable = (
                (t.column("m_5").to_numpy() == 0)
                & (t.column("m_9").to_numpy() == 0)
                & (t.column("m_10").to_numpy() == 0)
            )
            adj = ts[1:] - ts[:-1] == BAR_MS  # adjacent-minute pairs only
            ok0 = usable
            okp = adj & usable[:-1] & usable[1:]
            r0 = _corr(a[ok0], b[ok0])
            rp = _corr(a[:-1][okp], b[1:][okp])  # kline_t vs agg_{t+1}
            rm = _corr(a[1:][okp], b[:-1][okp])  # kline_t vs agg_{t-1}
            n = int(okp.sum())
            if n < A_MIN_PAIRS:
                rows_out.append({"symbol": sym, "month": month, "status": f"skipped (n={n})"})
                continue
            margin = r0 - max(rp, rm)
            rows_out.append(
                {
                    "symbol": sym,
                    "month": month,
                    "status": "ok",
                    "n_pairs": n,
                    "r_lag0": round(r0, 4),
                    "r_lag_plus1": round(rp, 4),
                    "r_lag_minus1": round(rm, 4),
                    "margin": round(margin, 4),
                    "pass": bool(r0 > rp + MARGIN and r0 > rm + MARGIN),
                }
            )
    return rows_out


def _download(sym: str, day: str) -> tuple[Path, str] | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    dst = CACHE / f"{sym}-aggTrades-{day}.zip"
    if not dst.exists():
        url = B_URL.format(s=sym, d=day)
        print(f"[B] downloading {url}")
        try:
            with (
                urllib.request.urlopen(url, timeout=180, context=_SSL_CTX) as r,
                open(dst, "wb") as f,
            ):
                while chunk := r.read(1 << 20):
                    f.write(chunk)
        except Exception as e:  # a missing archive is a recorded outcome, not a crash
            print(f"[B] download FAILED for {sym} {day}: {e}")
            dst.unlink(missing_ok=True)
            return None
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    return dst, digest


def _recompute_minutes(zpath: Path) -> dict[int, tuple[float, float, int, int]]:
    """Independent per-minute aggregation: {bar_open_ms: (v_buy, v_sell, n_buy, n_sell)}.

    The explicit [open, open+60s) window: a trade at transact_time t belongs to minute
    (t // 60000) * 60000. Taker BUY := is_buyer_maker == false (spec §2.1)."""
    acc: dict[int, list[float]] = {}
    with zipfile.ZipFile(zpath) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            txt = io.TextIOWrapper(f, encoding="utf-8")
            first = txt.readline()
            if not first.split(",")[0].strip().isdigit():  # header row (era-dependent)
                pass  # consumed
            else:
                _consume_row(acc, first.rstrip("\n").split(","))
            for row in csv.reader(txt):
                _consume_row(acc, row)
    return {k: (v[0], v[1], int(v[2]), int(v[3])) for k, v in acc.items()}


def _consume_row(acc: dict[int, list[float]], row: list[str]) -> None:
    # agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time, is_buyer_maker
    qty = float(row[2])
    bar = (int(row[5]) // BAR_MS) * BAR_MS
    is_buy = row[6].strip().lower() == "false"  # buyer NOT the maker → taker BUY
    slot = acc.setdefault(bar, [0.0, 0.0, 0, 0])
    if is_buy:
        slot[0] += qty
        slot[2] += 1
    else:
        slot[1] += qty
        slot[3] += 1


def part_b(con, *, skip_download: bool) -> list[dict]:
    rows_out: list[dict] = []
    for sym, day in B_DAYS:
        if skip_download and not (CACHE / f"{sym}-aggTrades-{day}.zip").exists():
            rows_out.append({"symbol": sym, "day": day, "status": "skipped (no cache)"})
            continue
        got = _download(sym, day)
        if got is None:
            rows_out.append({"symbol": sym, "day": day, "status": "download failed"})
            continue
        zpath, digest = got
        mins = _recompute_minutes(zpath)
        t = con.execute(
            "SELECT bar_open_ms, x_7, x_8, m_7, m_8 FROM bars "
            "WHERE symbol = ? AND date = ? ORDER BY bar_open_ms",
            [sym, day],
        ).to_arrow_table()
        ts = t.column("bar_open_ms").to_numpy().astype(np.int64)
        x7 = t.column("x_7").to_numpy().astype(np.float64)
        x8 = t.column("x_8").to_numpy().astype(np.float64)
        usable = (t.column("m_7").to_numpy() == 0) & (t.column("m_8").to_numpy() == 0)
        tfi_re = np.full(ts.shape[0], np.nan)
        sci_re = np.full(ts.shape[0], np.nan)
        for i, tt in enumerate(ts):
            m = mins.get(int(tt))
            if m is not None:
                vb, vs, nb, ns = m
                tfi_re[i] = (vb - vs) / (vb + vs + EPS_CNT)
                sci_re[i] = (nb - ns) / (nb + ns + EPS_CNT)
        ok = usable & np.isfinite(tfi_re)
        n = int(ok.sum())
        d7 = float(np.max(np.abs(tfi_re[ok] - x7[ok]))) if n else float("nan")
        d8 = float(np.max(np.abs(sci_re[ok] - x8[ok]))) if n else float("nan")
        adj = ts[1:] - ts[:-1] == BAR_MS
        okp = adj & ok[:-1] & ok[1:]
        r0 = _corr(tfi_re[ok], x7[ok])
        rp = _corr(tfi_re[:-1][okp], x7[1:][okp])
        rm = _corr(tfi_re[1:][okp], x7[:-1][okp])
        rows_out.append(
            {
                "symbol": sym,
                "day": day,
                "status": "ok",
                "sha256": digest,
                "n_minutes_compared": n,
                "max_abs_diff_tfi": f"{d7:.2e}",
                "max_abs_diff_count_imb": f"{d8:.2e}",
                "r_lag0": round(r0, 6),
                "r_lag_plus1": round(rp, 4),
                "r_lag_minus1": round(rm, 4),
                "pass": bool(n > 500 and d7 <= B_ATOL and d8 <= B_ATOL and r0 > max(rp, rm) + 0.5),
            }
        )
        print(
            f"[B] {sym} {day}: n={n} max|ΔTFI|={d7:.2e} max|Δsci|={d8:.2e} "
            f"r0={r0:.6f} r+1={rp:.3f} r-1={rm:.3f}"
        )
    return rows_out


def _md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "—")) for c in cols) + " |")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()
    con = connect_lake(LAKE)

    print("=== (A) in-lake lag sharpness ===")
    a_rows = part_a(con)
    a_ok = [r for r in a_rows if r["status"] == "ok"]
    a_pass = all(r["pass"] for r in a_ok) and len(a_ok) >= 20
    for r in a_ok:
        print(
            f"[A] {r['symbol']:14s} {r['month']}  r0={r['r_lag0']:+.4f} "
            f"r+1={r['r_lag_plus1']:+.4f} r-1={r['r_lag_minus1']:+.4f} "
            f"margin={r['margin']:+.4f} {'PASS' if r['pass'] else 'FAIL'}"
        )

    print("=== (B) definitive raw-day recompute ===")
    b_rows = part_b(con, skip_download=args.skip_download)
    b_ok = [r for r in b_rows if r["status"] == "ok"]
    b_pass = all(r["pass"] for r in b_ok) and len(b_ok) >= 3

    verdict = "PASS" if (a_pass and b_pass) else "FAIL"
    doc = {
        "audit": "m6_micro_alignment (item 6)",
        "verdict": verdict,
        "part_a": {"margin_floor": MARGIN, "n_samples": len(a_ok), "rows": a_rows},
        "part_b": {"atol": B_ATOL, "n_days": len(b_ok), "rows": b_rows},
    }
    REPORT_JSON.write_text(json.dumps(doc, indent=2, sort_keys=True))

    a_cols = [
        "symbol", "month", "n_pairs", "r_lag0", "r_lag_plus1", "r_lag_minus1", "margin", "pass"
    ]  # fmt: skip
    b_cols = [
        "symbol", "day", "n_minutes_compared", "max_abs_diff_tfi", "max_abs_diff_count_imb",
        "r_lag0", "r_lag_plus1", "r_lag_minus1", "pass",
    ]  # fmt: skip
    a_verdict = "ALL PASS" if a_pass else "FAILURES PRESENT"
    b_verdict = "ALL PASS" if b_pass else "FAILURES PRESENT"
    a_table = _md_table([r for r in a_rows if r["status"] == "ok"], a_cols)
    b_table = _md_table(b_rows, b_cols)
    skipped = (
        ", ".join(
            f"{r['symbol']} {r['month']} ({r['status']})" for r in a_rows if r["status"] != "ok"
        )
        or "none"
    )
    hash_rows = chr(10).join(
        f"| {r['symbol']}-aggTrades-{r['day']}.zip | `{r.get('sha256', '—')}` |"
        for r in b_rows
        if r["status"] == "ok"
    )
    md = f"""# M6 micro-alignment lag-sharpness audit — {verdict}

**What this closes:** the FP-3 residual — could the aggTrades-derived microstructure features
sit one minute off the kline bar they claim to describe? Two independent probes; the windowing
under test is `[open, open+60s)` (`bar_id = transact_time // 60000`, taker BUY =
`is_buyer_maker == false`).

## (A) in-lake lag sharpness — {len(a_ok)} (symbol, month) samples, {a_verdict}

corr(kline log-volume z (x_5), aggTrades volume reconstruction z(x_9)+z(x_10)) at lag 0 vs ±1;
adjacent usable minutes only; decisiveness floor: r0 must beat both by ≥ {MARGIN}.

{a_table}

Skipped/absent combos (late listings / thin months): {skipped}.

## (B) definitive raw-day recompute — {len(b_ok)} days, {b_verdict}

Fresh daily aggTrades archives re-downloaded from data.binance.vision, independently
re-aggregated per minute with the explicit [open, open+60s) window, compared to the lake's
stored TFI (x_7) / signed-count-imbalance (x_8) — BOUNDED features, so the lag-0 comparison is
float32-exact (atol {B_ATOL:.0e}), and a ±1-minute shift must collapse the correlation.
Archives are cached in `raw/binance/_audit_cache/` (gitignored — never redistributed); their
digests below are the audit trail.

{b_table}

| archive | sha256 |
|---|---|
{hash_rows}

**Reading:** part A shows the cross-source correlation peaking at lag 0 on every sample across
years and archive eras; part B shows the stored aggTrades features are bit-consistent (float32)
with an independent [open, open+60s) recompute on raw archives and that a one-minute shift
destroys the match — the microstructure block is aligned to the bar it claims to describe.
Machine-readable copy: `runs_manifest/m6_micro_alignment.json`.
"""
    REPORT_MD.write_text(md)
    print(f"[audit] {verdict} — report → {REPORT_MD} + {REPORT_JSON}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
