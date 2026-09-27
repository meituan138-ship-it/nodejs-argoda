"""Weekly cross-sectional panel for the whole A-share market.

Timing (no look-ahead, executable):
- decision at the close of the last trading day of week t;
- buy at the next trading day's open; hold until the open of the trading
  day after the next decision; target = that open-to-open return, ranked
  cross-sectionally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def wide(stocks: dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    return pd.DataFrame({c: d[col] for c, d in stocks.items()}).sort_index()


def limit_pct(codes: pd.Index, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily price-limit per stock: 20% for STAR (688) and ChiNext (300/301)
    after 2020-08-24, otherwise 10% (ST 5% is caught by the conservative
    tradability rule in the backtest)."""
    lim = pd.DataFrame(0.10, index=dates, columns=codes)
    star = [c for c in codes if c[2:5] == "688"]
    gem = [c for c in codes if c[2:5] in ("300", "301")]
    lim[star] = 0.20
    lim.loc[dates >= "2020-08-24", gem] = 0.20
    return lim


def build(stocks: dict[str, pd.DataFrame]):
    o, c, h, l = (wide(stocks, k) for k in ("open", "close", "high", "low"))
    days = c.index
    vol = wide(stocks, "volume").reindex(days)
    rc = wide(stocks, "raw_close").reindex(days)
    traded = vol.fillna(0) > 0
    amount = (vol * 100 * rc).where(traded)  # CNY; volume is in lots of 100 shares
    lc = np.log(c)
    r1 = lc.diff().where(traded)
    age = c.notna().cumsum()
    mkt = r1.where(age > 120).mean(axis=1)
    mkt_lc = mkt.fillna(0).cumsum()

    f: dict[str, pd.DataFrame] = {}
    for k in (1, 5, 10, 20, 60, 120):
        f[f"ret_{k}"] = lc - lc.shift(k)
    f["mom_250_20"] = lc.shift(20) - lc.shift(250)
    f["intraday_1"] = np.log(c / o)
    f["overnight_1"] = np.log(o / c.shift(1))
    f["intraday_20"] = f["intraday_1"].rolling(20).mean()
    f["overnight_20"] = f["overnight_1"].rolling(20).mean()
    f["vol_20"] = r1.rolling(20).std()
    f["vol_60"] = r1.rolling(60).std()
    f["down_vol_20"] = r1.clip(upper=0).rolling(20).std()
    la = np.log1p(amount)
    f["log_amount_20"] = la.rolling(20, min_periods=10).mean()  # size / liquidity
    f["amount_ratio_5_20"] = la.rolling(5, min_periods=3).mean() - f["log_amount_20"]
    f["amount_ratio_20_120"] = f["log_amount_20"] - la.rolling(120, min_periods=60).mean()
    f["amihud_20"] = np.log((r1.abs() / (amount + 1)).rolling(20, min_periods=10).mean() + EPS)
    f["max_ret_20"] = r1.rolling(20).max()
    f["min_ret_20"] = r1.rolling(20).min()
    f["skew_20"] = r1.rolling(20).skew()
    f["range_20"] = np.log(h / l).rolling(20).mean()
    f["dist_high_250"] = np.log(c / c.rolling(250, min_periods=120).max())
    f["dist_low_250"] = np.log(c / c.rolling(250, min_periods=120).min())
    lim = limit_pct(c.columns, days)
    up_hit = (c / c.shift(1) - 1) >= (lim - 0.002)
    dn_hit = (c / c.shift(1) - 1) <= -(lim - 0.002)
    f["limit_up_20"] = up_hit.rolling(20).sum()
    f["limit_down_20"] = dn_hit.rolling(20).sum()
    f["log_price"] = np.log(rc)
    f["ma_dev_5"] = lc - np.log(c.rolling(5).mean())
    f["ma_dev_20"] = lc - np.log(c.rolling(20).mean())
    f["ma_dev_60"] = lc - np.log(c.rolling(60).mean())
    mv = mkt.rolling(120).var()
    cov = r1.mul(mkt, axis=0).rolling(120, min_periods=60).mean() - \
        r1.rolling(120, min_periods=60).mean().mul(mkt.rolling(120).mean(), axis=0)
    f["beta_120"] = cov.div(mv + EPS, axis=0)
    f["idio_vol_60"] = (r1.sub(mkt, axis=0)).rolling(60).std()
    f["pv_corr_20"] = r1.rolling(20).corr(la.diff())
    f["age"] = np.log1p(age)

    # weekly decision dates and executable open-to-open forward return
    wk = pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max()
    dec = pd.DatetimeIndex(wk.values)
    pos = days.get_indexer(dec)
    entry = np.minimum(pos + 1, len(days) - 1)
    nxt = np.r_[pos[1:], len(days) - 1]
    exit_ = np.minimum(nxt + 1, len(days) - 1)
    ov = o.to_numpy()
    fwd = pd.DataFrame(np.log(ov[exit_] / ov[entry]), index=dec, columns=c.columns)
    fwd.iloc[-1] = np.nan

    # universe on each decision date: >120 days listed, traded today, 20d avg amount >= 20M CNY
    elig = (age > 120) & traded & (amount.rolling(20, min_periods=10).mean() >= 2e7)
    keep = elig.reindex(dec).stack()
    keep = keep[keep].index
    X = pd.DataFrame({n: df.reindex(dec).stack(future_stack=True).reindex(keep) for n, df in f.items()})
    X["fwd"] = fwd.stack(future_stack=True).reindex(keep)
    X.index.names = ["date", "code"]
    X = X.reset_index().replace([np.inf, -np.inf], np.nan)
    X["board"] = X["code"].str[2:5].map(lambda s: 3 if s == "688" else 2 if s in ("300", "301") else 1 if s[:2] == "00" else 0).astype(float)
    base = list(f) + ["board"]
    ranks = X.groupby("date")[list(f)].rank(pct=True).add_prefix("xs_")
    mstate = pd.DataFrame({
        "mkt_ret_5": mkt_lc - mkt_lc.shift(5), "mkt_ret_20": mkt_lc - mkt_lc.shift(20),
        "mkt_vol_20": mkt.rolling(20).std(),
        "breadth_20": (f["ret_20"].where(age > 120) > 0).mean(axis=1),
    }).reindex(dec)
    X = pd.concat([X, ranks], axis=1).merge(mstate, left_on="date", right_index=True, how="left")
    X["target"] = X.groupby("date")["fwd"].rank(pct=True) - 0.5
    feats = base + list(ranks.columns) + list(mstate.columns)
    extra = {"open": o, "close": c, "high": h, "lim": lim, "days": days, "dec": dec}
    return X, feats, extra
