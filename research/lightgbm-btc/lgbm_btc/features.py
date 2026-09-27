"""Causal, (mostly) stationary features for bar-level BTC prediction.

Every feature at row t only uses bars <= t (trailing rolling windows,
positive shifts, adjust=False EWMs). ``tests/test_leakage.py`` checks this
by recomputing the features on truncated data and comparing.

The feature families follow what the sources in the report found useful:
multi-horizon momentum/reversal (Liu & Tsyvinski 2021, G-Research top
solutions), trend / moving-average distance (FreqAI template, Hull MA from
G-Research 9th place), volatility (the most predictable quantity),
taker-flow imbalance (DRW 2025 order-flow features), calendar effects
(Shen et al. 2022 intraday momentum), cross-asset ETH and perp
funding/basis. Price levels are never used directly; returns and distances
are divided by recent volatility so a 1% move means the same in 2018 and 2026.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

EPS = 1e-12
VOL_SPAN = 168  # one week of hourly bars

ALL_GROUPS = (
    "momentum", "trend", "range", "oscillator", "volatility",
    "volume", "candle", "calendar", "cross_asset", "derivatives",
)
PRICE_ONLY_GROUPS = tuple(g for g in ALL_GROUPS if g not in ("cross_asset", "derivatives"))


# ----------------------------------------------------------------- helpers
def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def bar_volatility(close: pd.Series, span: int = VOL_SPAN) -> pd.Series:
    """EWMA std of one-bar log returns; the scale used to normalise features and labels."""
    return log_returns(close).ewm(span=span, adjust=False, min_periods=span).std()


def _ema(x: pd.Series, span: int) -> pd.Series:
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def _wma(x: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    vals = x.to_numpy(dtype=float)
    out = np.full(len(vals), np.nan)
    if len(vals) >= n:
        out[n - 1:] = np.convolve(vals, w[::-1], mode="valid")
    return pd.Series(out, index=x.index)


def _hma(x: pd.Series, n: int) -> pd.Series:
    """Hull moving average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    return _wma(2 * _wma(x, n // 2) - _wma(x, n), int(np.sqrt(n)))


def _zscore(x: pd.Series, n: int) -> pd.Series:
    return (x - x.rolling(n).mean()) / (x.rolling(n).std() + EPS)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / (dn + EPS))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> tuple[pd.Series, pd.Series]:
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    atr = _true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (atr + EPS)
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / (atr + EPS)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    adx = dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return adx, (plus_di - minus_di)


def _mfi(high, low, close, volume, n: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    flow = tp * volume
    pos = flow.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg = flow.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / (neg + EPS))


# ----------------------------------------------------------- feature groups
def _momentum(df: pd.DataFrame, sig: pd.Series) -> dict[str, pd.Series]:
    logc = np.log(df["close"])
    r1 = logc.diff()
    out = {}
    for k in (1, 2, 4, 8, 12, 24, 48, 72, 120, 168, 336, 504, 720):
        out[f"ret_{k}"] = (logc - logc.shift(k)) / (sig * np.sqrt(k))
    for lag in (1, 2, 3):
        out[f"ret_1_lag{lag}"] = r1.shift(lag) / sig
    # sign consistency of the last 24 hourly moves (trend "quality")
    out["up_frac_24"] = (r1 > 0).astype(float).rolling(24).mean()
    return out


def _trend(df: pd.DataFrame, sig: pd.Series) -> dict[str, pd.Series]:
    c = df["close"]
    logc = np.log(c)
    out = {}
    for n in (12, 24, 72, 168, 504):
        ema = _ema(c, n)
        out[f"ema_dev_{n}"] = (logc - np.log(ema)) / (sig * np.sqrt(n))
        out[f"ema_slope_{n}"] = (np.log(ema) - np.log(ema.shift(6))) / sig
    for n in (55, 210):  # Fibonacci-ish windows from the G-Research 9th place solution
        out[f"hma_dev_{n}"] = (logc - np.log(_hma(c, n).clip(lower=EPS))) / (sig * np.sqrt(n))
    out["ema_cross_24_168"] = (np.log(_ema(c, 24)) - np.log(_ema(c, 168))) / sig
    return out


def _range(df: pd.DataFrame, sig: pd.Series) -> dict[str, pd.Series]:
    c, h, l = df["close"], df["high"], df["low"]
    out = {}
    for n in (24, 168, 720):
        hh = h.rolling(n).max()
        ll = l.rolling(n).min()
        out[f"range_pos_{n}"] = (c - ll) / (hh - ll + EPS)
        out[f"dd_from_high_{n}"] = np.log(c / hh) / (sig * np.sqrt(n))
        out[f"up_from_low_{n}"] = np.log(c / ll) / (sig * np.sqrt(n))
    return out


def _oscillator(df: pd.DataFrame, sig: pd.Series) -> dict[str, pd.Series]:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    out = {f"rsi_{n}": _rsi(c, n) for n in (14, 48, 168)}
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    out["stoch_k_14"] = (c - ll) / (hh - ll + EPS)
    macd = _ema(c, 12) - _ema(c, 26)
    out["macd_hist"] = (macd - _ema(macd, 9)) / (c * sig)
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_pctb_20"] = (c - (ma20 - 2 * sd20)) / (4 * sd20 + EPS)
    adx, di_diff = _adx(h, l, c, 14)
    out["adx_14"] = adx
    out["di_diff_14"] = di_diff
    out["mfi_14"] = _mfi(h, l, c, v, 14)
    return out


def _volatility(df: pd.DataFrame, sig: pd.Series) -> dict[str, pd.Series]:
    c, h, l, o = df["close"], df["high"], df["low"], df["open"]
    r1 = np.log(c).diff()
    out = {}
    rv = {n: r1.rolling(n).std() for n in (6, 24, 72, 168, 720)}
    for n, s in rv.items():
        out[f"log_rv_{n}"] = np.log(s + EPS)
    out["rv_ratio_6_72"] = rv[6] / (rv[72] + EPS)
    out["rv_ratio_24_168"] = rv[24] / (rv[168] + EPS)
    out["rv_ratio_168_720"] = rv[168] / (rv[720] + EPS)
    hl2 = np.log(h / l) ** 2
    for n in (24, 168):
        out[f"log_parkinson_{n}"] = 0.5 * np.log(hl2.rolling(n).mean() / (4 * np.log(2)) + EPS)
    gk = 0.5 * hl2 - (2 * np.log(2) - 1) * np.log(c / o) ** 2
    out["log_garman_klass_24"] = 0.5 * np.log(gk.rolling(24).mean().clip(lower=EPS))
    out["atr_14_rel"] = _true_range(h, l, c).rolling(14).mean() / c / (sig + EPS)
    out["skew_168"] = r1.rolling(168).skew()
    out["kurt_168"] = r1.rolling(168).kurt()
    r2 = r1 ** 2
    for n in (24, 168):
        out[f"down_var_share_{n}"] = (r2 * (r1 < 0)).rolling(n).sum() / (r2.rolling(n).sum() + EPS)
    out["vol_of_vol_168"] = np.log(rv[24] + EPS).rolling(168).std()
    return out


def _volume(df: pd.DataFrame, sig: pd.Series) -> dict[str, pd.Series]:
    qv, trades = df["quote_volume"], df["trades"]
    buy = df["taker_buy_quote"]
    r1 = np.log(df["close"]).diff()
    lqv = np.log1p(qv)
    out = {
        "qvol_z_168": _zscore(lqv, 168),
        "qvol_z_720": _zscore(lqv, 720),
        "rel_qvol_24": qv / (qv.rolling(24).mean() + EPS),
        "trades_z_168": _zscore(np.log1p(trades), 168),
        "avg_trade_size_z_168": _zscore(np.log1p(qv / (trades + 1)), 168),
    }
    # taker order-flow imbalance in [-1, 1]: (buy - sell) / total over the window
    for n in (1, 6, 24, 168):
        b, t = buy.rolling(n).sum(), qv.rolling(n).sum()
        out[f"ofi_{n}"] = (2 * b - t) / (t + EPS)
    out["ofi_z_168"] = _zscore(out["ofi_1"], 168)
    amihud = (r1.abs() / (qv + 1.0)).rolling(24).mean()
    out["log_amihud_24"] = np.log(amihud + EPS)
    out["ret_vol_corr_168"] = r1.rolling(168).corr(lqv.diff())
    return out


def _candle(df: pd.DataFrame, sig: pd.Series) -> dict[str, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l) + EPS
    body = (c - o) / rng
    out = {
        "candle_body": body,
        "candle_upper": (h - np.maximum(o, c)) / rng,
        "candle_lower": (np.minimum(o, c) - l) / rng,
        "candle_range": np.log(h / l) / (sig + EPS),
        "candle_body_mean_24": body.rolling(24).mean(),
    }
    return out


def _calendar(df: pd.DataFrame, sig: pd.Series) -> dict[str, pd.Series]:
    idx = df.index
    return {
        "hour": pd.Series(idx.hour, index=idx, dtype=float),
        "dow": pd.Series(idx.dayofweek, index=idx, dtype=float),
    }


def _cross_asset(df: pd.DataFrame, sig: pd.Series, eth: pd.DataFrame) -> dict[str, pd.Series]:
    ec = eth["close"].reindex(df.index).ffill(limit=3)
    le = np.log(ec)
    esig = bar_volatility(ec)
    ratio = le - np.log(df["close"])
    rsig = ratio.diff().ewm(span=VOL_SPAN, adjust=False, min_periods=VOL_SPAN).std()
    out = {}
    for k in (1, 4, 24, 168):
        out[f"eth_ret_{k}"] = (le - le.shift(k)) / (esig * np.sqrt(k))
    for k in (24, 168):
        out[f"ethbtc_ret_{k}"] = (ratio - ratio.shift(k)) / (rsig * np.sqrt(k))
    out["btc_eth_corr_168"] = np.log(df["close"]).diff().rolling(168).corr(le.diff())
    out["eth_btc_vol_ratio"] = esig / (sig + EPS)
    return out


def _derivatives(df: pd.DataFrame, sig: pd.Series, funding: pd.Series | None,
                 premium: pd.Series | None) -> dict[str, pd.Series]:
    out = {}
    if funding is not None and len(funding):
        f = pd.DataFrame({
            "funding_last": funding,
            "funding_mean_3": funding.rolling(3).mean(),
            "funding_mean_21": funding.rolling(21).mean(),
            "funding_z_90": _zscore(funding, 90),
        })
        # a bar is decided at its close (open_time + 1 bar); only settlements up to then are known
        bar = df.index[1] - df.index[0]
        close_times = pd.DataFrame({"t": df.index + bar})
        f = f.reset_index().rename(columns={"index": "t"})
        f["t"] = f["t"].astype(close_times["t"].dtype)
        merged = pd.merge_asof(close_times, f.sort_values("t"), on="t", direction="backward")
        for col in ("funding_last", "funding_mean_3", "funding_mean_21", "funding_z_90"):
            out[col] = pd.Series(merged[col].to_numpy(), index=df.index)
    if premium is not None and len(premium):
        p = premium.reindex(df.index).ffill(limit=3)
        out["premium"] = p
        out["premium_mean_24"] = p.rolling(24).mean()
        out["premium_z_168"] = _zscore(p, 168)
        out["premium_chg_24"] = p - p.shift(24)
    return out


# ------------------------------------------------------------------ public
def build_features(
    btc: pd.DataFrame,
    eth: pd.DataFrame | None = None,
    funding: pd.Series | None = None,
    premium: pd.Series | None = None,
    groups: Iterable[str] = ALL_GROUPS,
) -> pd.DataFrame:
    """Return a feature matrix aligned with ``btc.index`` (NaN during warm-up)."""
    groups = set(groups)
    unknown = groups - set(ALL_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature groups: {sorted(unknown)}")
    sig = bar_volatility(btc["close"])
    builders = {
        "momentum": lambda: _momentum(btc, sig),
        "trend": lambda: _trend(btc, sig),
        "range": lambda: _range(btc, sig),
        "oscillator": lambda: _oscillator(btc, sig),
        "volatility": lambda: _volatility(btc, sig),
        "volume": lambda: _volume(btc, sig),
        "candle": lambda: _candle(btc, sig),
        "calendar": lambda: _calendar(btc, sig),
        "cross_asset": lambda: _cross_asset(btc, sig, eth) if eth is not None else {},
        "derivatives": lambda: _derivatives(btc, sig, funding, premium),
    }
    feats: dict[str, pd.Series] = {}
    group_of: dict[str, str] = {}
    for name in ALL_GROUPS:
        if name in groups:
            block = builders[name]()
            feats.update(block)
            group_of.update({col: name for col in block})
    X = pd.DataFrame(feats, index=btc.index)
    X = X.replace([np.inf, -np.inf], np.nan).astype("float32")
    X.attrs["group_of"] = group_of
    return X
