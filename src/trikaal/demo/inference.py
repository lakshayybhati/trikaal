"""Demo inference — the PRODUCTION eval path, reused rather than reimplemented.

Every step here calls the same function the scored money run called: ``select_arm``,
``tokenize_features``, ``symbol_decisions``, ``predict_mu``. Nothing is reimplemented, so the demo
cannot drift from the measured model by construction — and what CAN still drift (which window,
which decision index, which sigma) is exactly what ``scripts/m6_demo_acceptance.py`` pins by
asserting bit-identity against a direct production call.

CAUSAL SAFETY IS INHERITED, NOT RE-EARNED. ``universe_loader`` documents that the lake's features
are causally self-normalized per symbol at build time (streaming EWMA z-scores over bars <= t),
so there is no train-fitted statistic and no frozen-stats artifact to ship or to get wrong here.
``symbol_decisions`` additionally refuses any decision bar with fewer than ``seq_len`` bars of
history, and the realized path shown beside a forecast is read only AFTER the forecast is formed.

★ NO P&L. See the package docstring. This module defines no function that returns a return, a
position, a profit or an equity curve, and the demo's chart cannot render one because none is
computed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from trikaal.data.universe_loader import calendar_boundary_ms, connect_lake
from trikaal.eval.predict import predict_mu
from trikaal.eval.xsection import BAR_MS, SymbolEval, XSectionConfig, symbol_decisions
from trikaal.model.predictor import TrikaalAR
from trikaal.run.matrix import load_symbol_arrays
from trikaal.tokenizer.model import TokenizerAE
from trikaal.train.arms import select_arm
from trikaal.train.checkpoint import artifact_hash
from trikaal.train.token_stream import tokenize_features

REPO = Path(__file__).resolve().parents[3]

# The demo serves cell 1 ONLY, and the label says so in every rendered image: cell 1 is
# BSQ + OHLCV-only, the BASELINE arm. It is NOT the microstructure contribution — the legibility
# gate refused cells 4 and 5 before Stage-2, so those arms have no scored artifacts at all.
DEMO_CELL = 1
DEMO_ARM = "ohlcv"
DEMO_H = 15  # PRIMARY_H — the pre-registered headline horizon
DEMO_SEQ_LEN = 512

# The three complete cell-1 money units. Seeds 0, 2, 4 — NOT 0, 1, 2: shards r0/r1/r2 carried
# seeds 0/2/4, and quoting them as 0/1/2 would misname which runs these are.
UNIT_DIRS: dict[int, Path] = {
    0: REPO / "runs_cloud/rescue/r0/cell1_bsq_ohlcv_seed0",
    2: REPO / "runs_cloud/rescue/r1/cell1_bsq_ohlcv_seed2",
    4: REPO / "runs_cloud/rescue/r2/cell1_bsq_ohlcv_seed4",
}

# Pinned observation window / split — must match the money run's own pins, asserted on load.
PINNED_WINDOW = ("2021-01-01", "2025-01-01")
PINNED_TRAIN_FRAC = 0.7

# ★ THE MONEY LAKE, NOT THE M2 SINGLE-SYMBOL LAKE — and this was a real bug, caught only by
# rendering more than one symbol. `trikaal.data.lake.connect` defaults to `processed/bars`, which
# holds ONE symbol (BTCUSDT, 2023 only); the 200-symbol universe the money run scored lives at
# `processed/universe_bars`. The demo pointed at the former and APPEARED to work, because BTCUSDT
# exists in both. Worse, the acceptance gate passed while it did: both of its arms shared the same
# connection, so they agreed with each other ON THE WRONG DATA. An agreement test between two
# paths cannot detect a source both paths share — so the source is now pinned here and asserted.
LAKE_ROOT = Path("processed/universe_bars")


@dataclass(frozen=True)
class Unit:
    """One loaded (tokenizer, predictor) pair with its identity proven, not assumed."""

    seed: int
    tok: TokenizerAE
    model: TrikaalAR
    tokenizer_hash: str
    predictor_hash: str
    n_params_realized: int
    n_params_mtp: int

    @property
    def n_params_excluding_mtp(self) -> int:
        return self.n_params_realized - self.n_params_mtp


def available_seeds() -> list[int]:
    """Seeds whose unit directory actually holds both required checkpoints."""
    return sorted(
        s
        for s, d in UNIT_DIRS.items()
        if (d / "tokenizer.pt").exists() and (d / "predictor.pt").exists()
    )


def load_unit(seed: int, *, device: str = "cpu") -> Unit:
    """Load one unit and VERIFY ITS IDENTITY against the run manifest before returning it.

    Identity is checked by BEHAVIOUR — the content hash is RECOMPUTED from the loaded weights and
    config and must agree three ways (recomputed == stored == manifest) — rather than by trusting
    the file name or the stored string. A demo serving a checkpoint that is not the scored one
    would be showing a different model from the one the paper reports.
    """
    d = UNIT_DIRS[seed]
    tok_p, pred_p = d / "tokenizer.pt", d / "predictor.pt"
    man = json.loads((d / "run_manifest.json").read_text())

    tsd = torch.load(tok_p, map_location=device, weights_only=False)
    psd = torch.load(pred_p, map_location=device, weights_only=False)

    # ★ THREE-WAY IDENTITY, AND IT IS RECOMPUTED RATHER THAN READ.
    # The canonical identity is the project's own `artifact_hash(state_dict, config)` — a CONTENT
    # hash that deliberately excludes `meta` so two runs producing identical weights hash alike.
    # It is NOT the sha256 of the file bytes: my first version hashed the bytes, the gate refused
    # all three real units, and the refusal was correct behaviour for the wrong reason.
    # RECOMPUTING it (rather than trusting the stored `hash` field) is what makes this an
    # integrity check: a tampered weight changes the recomputation but not the stored string, so
    # reading the field alone would be verifying EXISTENCE where BEHAVIOUR is required.
    for name, ck, want in (
        ("tokenizer", tsd, man["artifacts"]["tokenizer_hash"]),
        ("predictor", psd, man["artifacts"]["predictor_hash"]),
    ):
        recomputed = artifact_hash(ck["state_dict"], ck["config"])
        stored = ck.get("hash")
        if not (recomputed == stored == want):
            raise ValueError(
                f"seed {seed}: {name} identity FAILED — recomputed {recomputed}, stored {stored}, "
                f"manifest {want}. REFUSING to serve a model that is not the scored one."
            )
    if int(man["cell"]["id"]) != DEMO_CELL or man["cell"]["arm"] != DEMO_ARM:
        raise ValueError(
            f"seed {seed}: unit is cell {man['cell']['id']}/{man['cell']['arm']}, "
            f"expected {DEMO_CELL}/{DEMO_ARM}"
        )
    tok_h, pred_h = tsd["hash"], psd["hash"]
    tok = TokenizerAE(**tsd["config"])
    tok.load_state_dict(tsd["state_dict"])
    # ★ .to(device) IS LOAD-BEARING AND WAS MISSING. torch.load(map_location=...) places the
    # STATE DICT, not the MODULE: load_state_dict copies values into parameters that are still on
    # CPU, so a "cuda" unit silently stayed on CPU and blew up inside the embedding lookup with a
    # device mismatch. Never reachable on CPU, so 790 green tests could not see it — it surfaced
    # on the FIRST GPU execution, 19 seconds into a smoke test that exists precisely because a
    # component whose first run is on rented hardware is an untested component.
    tok.to(device)
    tok.eval()
    model = TrikaalAR(**psd["config"])
    model.load_state_dict(psd["state_dict"])
    model.to(device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    mtp = sum(p.numel() for n, p in model.named_parameters() if n.split(".")[0] == "mtp")
    return Unit(
        seed=seed,
        tok=tok,
        model=model,
        tokenizer_hash=tok_h,
        predictor_hash=pred_h,
        n_params_realized=total,
        n_params_mtp=mtp,
    )


@dataclass(frozen=True)
class SeedForecast:
    """One seed's forecast. NOTE WHAT IS ABSENT: no return, no position, no P&L."""

    seed: int
    mu_hat: float  # expected cumulative log-return over the next h minutes
    forecast_close: float  # close_t * exp(mu_hat) — a PRICE, for plotting only
    predictor_hash: str


@dataclass(frozen=True)
class ForecastBundle:
    """All seeds at one decision bar, plus the context and realized path for plotting.

    ★ THERE IS DELIBERATELY NO ``mean_mu`` / ``consensus`` / ``ensemble`` FIELD. Averaging the
    seeds would hide the finding: identical data and configuration produced headline IRs spanning
    118.03 and activity spanning 24.4x. The spread IS a result, so the renderer plots every seed
    and no averaging value exists in the code to be surfaced later.
    """

    symbol: str
    decision_ms: int
    h: int
    context_close: np.ndarray  # the model's actual 512-bar context
    context_ms: np.ndarray
    realized_close: np.ndarray  # the next h closes, read only AFTER the forecast is formed
    realized_ms: np.ndarray
    seeds: tuple[SeedForecast, ...]
    meta: dict = field(default_factory=dict)


def _cfg(device: str) -> XSectionConfig:
    return XSectionConfig(
        window_start=PINNED_WINDOW[0],
        window_end=PINNED_WINDOW[1],
        train_frac=PINNED_TRAIN_FRAC,
        h=DEMO_H,
        seq_len=DEMO_SEQ_LEN,
        cap_per_symbol=None,
        device=device,
        # the demo scores single decisions; it is not a money run and never claims to be
        money=False,
    )


@dataclass(frozen=True)
class PreparedSymbol:
    """One symbol loaded and tokenized ONCE per seed — the expensive part, done exactly once.

    WHY THIS EXISTS AND WHY IT IS NOT A SHORTCUT. Production tokenizes a symbol's ENTIRE series in
    one call, and ``tokenize_features`` chunks by segment with a fixed window, so the window
    alignment of any given bar depends on the whole series. Tokenizing only a 512-bar slice would
    be faster and would NOT reproduce production. So the demo keeps whole-symbol tokenization —
    measured at 29.3 s per (symbol, seed) on CPU for a 2,103,840-bar symbol — and simply refuses
    to repeat it. The first version re-tokenized per seed AND per decision; the acceptance gate did
    ~18 full tokenizations and timed out at ten minutes, which is how this was found.
    """

    symbol: str
    se: SymbolEval
    tokens: dict[int, tuple[np.ndarray, np.ndarray]]
    close: np.ndarray


def prepare_symbol(
    symbol: str,
    *,
    units: dict[int, Unit],
    device: str = "cpu",
    lake_root: Path | None = None,
) -> PreparedSymbol:
    """Load a symbol and tokenize it once per seed, the production way (whole series)."""
    cfg = _cfg(device)
    con = connect_lake(lake_root or LAKE_ROOT)
    boundary = calendar_boundary_ms(cfg.window_start, cfg.window_end, cfg.train_frac)
    raw = load_symbol_arrays(con, symbol, boundary_ms=boundary, tail_bars=None)
    if raw["x"].shape[0] == 0:
        raise ValueError(f"{symbol}: no bars in {lake_root or LAKE_ROOT}")

    se = SymbolEval(
        symbol=symbol,
        x=raw["x"],
        mask=raw["mask"],
        segment_id=raw["segment_id"],
        ts=raw["ts"],
        sigma=raw["sigma"],
        raw_ret_close=raw["raw_ret_close"],
    )
    x_arm, m_arm = select_arm(np.asarray(se.x, np.float32), se.mask, DEMO_ARM)
    tokens = {
        seed: tokenize_features(
            u.tok, x_arm, m_arm, se.segment_id, window=cfg.seq_len, device=device
        )
        for seed, u in sorted(units.items())
    }
    return PreparedSymbol(
        symbol=symbol,
        se=se,
        tokens=tokens,
        close=_close_from_raw_returns(se.raw_ret_close, se.segment_id),
    )


def forecast_at(
    symbol: str,
    decision_ms: int,
    *,
    units: dict[int, Unit],
    device: str = "cpu",
    lake_root: Path | None = None,
    context_bars: int = DEMO_SEQ_LEN,
    prepared: PreparedSymbol | None = None,
) -> ForecastBundle:
    """Forecast at ONE decision instant, through the production path, for every supplied seed."""
    cfg = _cfg(device)
    if prepared is None:
        prepared = prepare_symbol(symbol, units=units, device=device, lake_root=lake_root)
    if prepared.symbol != symbol:
        raise ValueError(f"prepared symbol {prepared.symbol!r} != requested {symbol!r}")
    se = prepared.se

    # symbol_decisions asserts the grid lives in the forward region and enforces the seq_len
    # history requirement — the demo gets the eval's causal guarantee rather than its own.
    grid = np.array([decision_ms], dtype=np.int64)
    dec = symbol_decisions(se, grid, cfg)
    if dec.size == 0:
        raise ValueError(
            f"{symbol} @ {decision_ms}: not a valid decision instant — the bar must exist, lie at "
            f"or after the train/eval boundary, carry >= {cfg.seq_len} bars of history and have "
            f"an in-range label bar {cfg.h} bars ahead."
        )
    i = int(dec[0])

    close = prepared.close
    lo = max(0, i - context_bars + 1)
    out: list[SeedForecast] = []
    for seed in sorted(units):
        u = units[seed]
        b_c, b_f = prepared.tokens[seed]
        mu = predict_mu(
            u.model,
            u.tok,
            b_c,
            b_f,
            se.ts,
            se.sigma,
            dec,
            h=cfg.h,
            seq_len=cfg.seq_len,
            device=device,
            estimator=cfg.mu_estimator,
        )
        m = float(mu[0])
        out.append(
            SeedForecast(
                seed=seed,
                mu_hat=m,
                forecast_close=float(close[i] * np.exp(m)),
                predictor_hash=u.predictor_hash,
            )
        )

    any_u = units[sorted(units)[0]]
    return ForecastBundle(
        symbol=symbol,
        decision_ms=int(se.ts[i]),
        h=cfg.h,
        context_close=close[lo : i + 1],
        context_ms=se.ts[lo : i + 1],
        realized_close=close[i : i + cfg.h + 1],
        realized_ms=se.ts[i : i + cfg.h + 1],
        seeds=tuple(out),
        meta={
            "cell": DEMO_CELL,
            "arm": DEMO_ARM,
            "quantizer": "bsq",
            "seq_len": cfg.seq_len,
            "estimator": cfg.mu_estimator,
            "n_params_realized": any_u.n_params_realized,
            "n_params_mtp": any_u.n_params_mtp,
            "bar_ms": BAR_MS,
        },
    )


def _close_from_raw_returns(raw_ret_close: np.ndarray, segment_id: np.ndarray) -> np.ndarray:
    """A RELATIVE close path (first bar of each segment = 1.0) from the banked log-returns.

    The lake stores causal log-returns, not raw prices, so the chart shows a normalized price
    path. This is honest for a forecast plot — the y-axis is labelled relative — and it removes
    any chance of the image being read as a currency-denominated P&L.
    """
    r = np.asarray(raw_ret_close, dtype=np.float64)
    seg = np.asarray(segment_id)
    out = np.empty(r.shape[0], dtype=np.float64)
    start = 0
    for k in range(1, r.shape[0] + 1):
        if k == r.shape[0] or seg[k] != seg[start]:
            block = r[start:k].copy()
            block[0] = 0.0  # no return into the first bar of a segment
            out[start:k] = np.exp(np.cumsum(np.nan_to_num(block, nan=0.0)))
            start = k
    return out
