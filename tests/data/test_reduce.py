"""aggTrades reduction + ingest URL unit tests (data-pipeline §2.1/§2.2) — no network."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("polars")  # the real data pipeline is an optional extra

from trikaal.data.ingest import DumpSpec
from trikaal.data.reduce import build_raw_stream, load_klines, reduce_aggtrades

_AGG_CSV = (
    "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    "1,100.0,2.0,1,1,60000,false\n"  # bar 1, taker BUY 2.0
    "2,100.0,3.0,2,2,60500,true\n"  # bar 1, taker SELL 3.0
    "3,100.0,1.0,3,3,60900,false\n"  # bar 1, taker BUY 1.0
    "4,100.0,5.0,4,4,120000,true\n"  # bar 2, taker SELL 5.0
)
_KLINE_CSV = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore\n"
    "60000,100,101,99,100.5,6.0,119999,600,3,3.0,300,0\n"
    "120000,100.5,100.5,99,99.5,5.0,179999,500,1,0,0,0\n"
)


def _zip_csv(tmp_path: Path, name: str, content: str) -> Path:
    csv = tmp_path / name
    csv.write_text(content)
    z = tmp_path / (name.replace(".csv", ".zip"))
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(csv, arcname=name)
    return z


def test_buy_sell_classification_and_bucketing(tmp_path):
    """is_buyer_maker==False → taker BUY; ==True → taker SELL; bar_id = transact_time // 60000."""
    z = _zip_csv(tmp_path, "BTCUSDT-aggTrades-2023-01.csv", _AGG_CSV)
    df = reduce_aggtrades(z).sort("bar_id")
    rows = {int(r["bar_id"]): r for r in df.to_dicts()}
    assert rows[1]["V_buy"] == 3.0 and rows[1]["V_sell"] == 3.0  # BUY 2+1, SELL 3
    assert rows[1]["N_buy"] == 2 and rows[1]["N_sell"] == 1
    assert rows[1]["V_agg"] == 6.0 and rows[1]["N_trades"] == 3
    assert sorted(rows[1]["sizes"]) == [1.0, 2.0, 3.0]
    assert rows[2]["V_buy"] == 0.0 and rows[2]["V_sell"] == 5.0


def test_build_raw_stream_join(tmp_path):
    kz = _zip_csv(tmp_path, "BTCUSDT-1m-2023-01.csv", _KLINE_CSV)
    az = _zip_csv(tmp_path, "BTCUSDT-aggTrades-2023-01.csv", _AGG_CSV)
    stream = build_raw_stream(load_klines(kz), reduce_aggtrades(az))
    assert len(stream) == 2
    assert stream.bar_open_ms.tolist() == [60000, 120000]
    assert np.isclose(stream.close[0], 100.5)
    assert np.isclose(stream.v_buy[0], 3.0) and np.isclose(stream.v_sell[0], 3.0)
    # microstructure is perp but funding/OI are deferred → empty event series
    assert stream.is_perp and stream.funding_ts.size == 0 and stream.oi_ts.size == 0


def test_dump_spec_urls():
    k = DumpSpec("BTCUSDT", "futures/um", "klines", "2023-01")
    a = DumpSpec("BTCUSDT", "futures/um", "aggTrades", "2023-01")
    assert k.url.endswith("futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2023-01.zip")
    assert a.url.endswith("futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-01.zip")
    assert k.store_market == "um" and a.filename == "BTCUSDT-aggTrades-2023-01.zip"
