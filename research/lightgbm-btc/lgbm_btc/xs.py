"""Cross-sectional (multi-coin) variant: predict each coin's next-24h return
*relative to the equal-weight market*, then go long the top quintile and
short the bottom quintile. This is the setting in which LightGBM won the
G-Research competition (14 coins, market-residualised target) and the one
Numerai Crypto scores.

Every per-coin feature is also turned into a cross-sectional rank within
the coins available at that hour, so the model compares coins with each
other instead of timing the whole market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_klines
from .features import EPS, _zscore

COINS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "LTC", "LINK", "DOT",
         "AVAX", "TRX", "BCH", "ETC", "XLM", "ATOM", "UNI", "FIL", "NEAR", "AAVE"]
H = 24


def load_panel(start: str = "2019-01", end: str = "2026-08-31") -> dict[str, pd.DataFrame]:
    return {c: load_klines(f"{c}USDT", "1h", "spot", start, end) for c in COINS}


def _coin_features(df: pd.DataFrame, sig: pd.Series, mkt_logc: pd.Series) -> dict[str, pd.Series]:
    c, h, l = df["close"], df["high"], df["low"]
    logc = np.log(c)
    r1 = logc.diff()
    out = {}
    for k in (1, 4, 12, 24, 72, 168, 336, 720):
        out[f"ret_{k}"] = (logc - logc.shift(k)) / (sig * np.sqrt(k))
        out[f"rel_ret_{k}"] = ((logc - logc.shift(k)) - (mkt_logc - mkt_logc.shift(k))) / (sig * np.sqrt(k))
    for n in (24, 168, 720):
        ema = c.ewm(span=n, adjust=False, min_periods=n).mean()
        out[f"ema_dev_{n}"] = (logc - np.log(ema)) / (sig * np.sqrt(n))
        hh, ll = h.rolling(n).max(), l.rolling(n).min()
        out[f"range_pos_{n}"] = (c - ll) / (hh - ll + EPS)
    rv = {n: r1.rolling(n).std() for n in (24, 168, 720)}
    for n, s in rv.items():
        out[f"log_rv_{n}"] = np.log(s + EPS)
    out["rv_ratio_24_168"] = rv[24] / (rv[168] + EPS)
    out["skew_168"] = r1.rolling(168).skew()
    mr = mkt_logc.diff()
    out["beta_720"] = r1.rolling(720).cov(mr) / (mr.rolling(720).var() + EPS)
    out["corr_mkt_720"] = r1.rolling(720).corr(mr)
    qv = df["quote_volume"]
    lqv = np.log1p(qv)
    out["qvol_z_168"] = _zscore(lqv, 168)
    out["log_qvol_720"] = lqv.rolling(720).mean()  # size / liquidity proxy
    for n in (24, 168):
        b, t = df["taker_buy_quote"].rolling(n).sum(), qv.rolling(n).sum()
        out[f"ofi_{n}"] = (2 * b - t) / (t + EPS)
    out["log_amihud_168"] = np.log((r1.abs() / (qv + 1.0)).rolling(168).mean() + EPS)
    return out


def build_panel(panel: dict[str, pd.DataFrame], min_history: int = 24 * 60,
                eligible: pd.DataFrame | None = None):
    """Long table indexed by (time, coin) with features, target and forward return.

    ``eligible`` (hour × coin booleans) restricts rows and the market average
    to a point-in-time universe; by default every coin listed > ``min_history``.
    """
    grid = pd.date_range(min(d.index[0] for d in panel.values()),
                         max(d.index[-1] for d in panel.values()), freq="h")
    close = pd.DataFrame({c: d["close"] for c, d in panel.items()}).reindex(grid)
    listed = close.notna().cumsum()  # bars since listing
    if eligible is None:
        eligible = listed > min_history
    eligible = eligible.reindex(index=grid, columns=close.columns, fill_value=False)
    logret = np.log(close).diff()
    mkt_ret = logret.where(eligible).mean(axis=1).fillna(0.0)
    mkt_logc = mkt_ret.cumsum()

    frames = []
    for coin, d in panel.items():
        d = d.reindex(grid)
        ok = eligible[coin]
        sig = np.log(d["close"]).diff().ewm(span=168, adjust=False, min_periods=168).std()
        f = pd.DataFrame(_coin_features(d, sig, mkt_logc), index=grid)
        # a coin delisted inside the holding window is exited at its last traded price
        fwd = np.log(d["close"].ffill().shift(-H) / d["close"])
        f["fwd_ret"] = fwd
        f["adv30"] = d["quote_volume"].rolling(720, min_periods=240).sum() / 30
        f["sig"] = sig
        f["coin"] = coin
        frames.append(f[ok.to_numpy()])
    X = pd.concat(frames)
    X.index.name = "time"
    X = X.replace([np.inf, -np.inf], np.nan)

    feat_cols = [c for c in X.columns if c not in ("fwd_ret", "sig", "coin", "adv30")]
    g = X.groupby(level=0)
    ranks = g[feat_cols].rank(pct=True).add_prefix("xs_")
    # market-wide state, identical for every coin at a given hour
    mkt = pd.DataFrame({
        "mkt_ret_24": mkt_logc - mkt_logc.shift(24),
        "mkt_ret_168": mkt_logc - mkt_logc.shift(168),
        "mkt_rv_168": np.log(mkt_ret.rolling(168).std() + EPS),
        "n_coins": eligible.sum(axis=1).astype(float),
        "hour": pd.Series(grid.hour, index=grid, dtype=float),
    }, index=grid)
    X = pd.concat([X, ranks], axis=1).join(mkt, how="left")
    # target: forward return minus the cross-sectional mean, scaled by the coin's vol
    resid = X["fwd_ret"] - X.groupby(level=0)["fwd_ret"].transform("mean")
    X["target"] = resid / (X["sig"] * np.sqrt(H))
    X["coin_id"] = X["coin"].astype("category").cat.codes.astype(float)
    features = feat_cols + list(ranks.columns) + list(mkt.columns)
    return X, features
