"""Weekly cross-sectional panel for S&P 500 stocks.

Decision at each week's last close; target = next 5 trading days' log
return minus the equal-weight universe mean, divided by the stock's
21-day volatility·√5 (the same market-residual target that worked in the
crypto study). Only stocks that were index members on the decision date
are used, which removes look-ahead in the universe (but see the report:
stocks whose price history Yahoo no longer serves are still missing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
H = 5  # trading days


def build(prices: dict[str, pd.DataFrame], member: pd.DataFrame, sector: dict[str, str]):
    close = pd.DataFrame({t: d["adjclose"] for t, d in prices.items()}).sort_index()
    close = close[close.index >= "2004-06-01"]
    raw = pd.DataFrame({t: d["close"] for t, d in prices.items()}).reindex(close.index)
    vol = pd.DataFrame({t: d["volume"] for t, d in prices.items()}).reindex(close.index)
    high = pd.DataFrame({t: d["high"] for t, d in prices.items()}).reindex(close.index)
    opn = pd.DataFrame({t: d["open"] for t, d in prices.items()}).reindex(close.index)
    days = close.index
    mem = member.reindex(days, method="ffill").reindex(columns=close.columns, fill_value=False).fillna(False)

    lc = np.log(close)
    r1 = lc.diff()
    mkt_r = r1.where(mem).mean(axis=1)
    mkt_lc = mkt_r.fillna(0).cumsum()
    sig = r1.rolling(21, min_periods=15).std()
    dv = raw * vol

    f: dict[str, pd.DataFrame] = {}
    for k in (1, 5, 21, 63, 126, 252):
        f[f"ret_{k}"] = (lc - lc.shift(k)) / (sig * np.sqrt(k))
        f[f"rel_{k}"] = ((lc - lc.shift(k)).sub(mkt_lc - mkt_lc.shift(k), axis=0)) / (sig * np.sqrt(k))
    f["mom_12_1"] = lc.shift(21) - lc.shift(252)
    f["log_vol_21"] = np.log(sig + EPS)
    f["log_vol_63"] = np.log(r1.rolling(63).std() + EPS)
    f["vol_ratio"] = sig / (r1.rolling(126).std() + EPS)
    mv = mkt_r.rolling(252).var()
    f["beta_252"] = r1.rolling(252).cov(mkt_r) / (mv.to_numpy()[:, None] + EPS)
    f["dist_52w_high"] = np.log(close / close.rolling(252).max())
    lo63, hi63 = close.rolling(63).min(), close.rolling(63).max()
    f["range_pos_63"] = (close - lo63) / (hi63 - lo63 + EPS)
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    f["rsi_14"] = 100 - 100 / (1 + up / (dn + EPS))
    ldv = np.log1p(dv)
    f["log_dollar_vol_21"] = ldv.rolling(21).mean()  # size / liquidity proxy
    f["dvol_z"] = (ldv.rolling(5).mean() - ldv.rolling(126).mean()) / (ldv.rolling(126).std() + EPS)
    f["amihud_21"] = np.log((r1.abs() / (dv + 1)).rolling(21).mean() + EPS)
    f["skew_63"] = r1.rolling(63).skew()
    f["gap_1"] = np.log(opn / raw.shift(1)).clip(-0.5, 0.5)
    f["max_ret_21"] = r1.rolling(21).max()  # lottery / MAX effect
    f["high_close_1"] = np.log(high / raw)

    # weekly decision dates: last trading day of each week
    wk = pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max()
    dec = pd.DatetimeIndex(wk.values)
    pos = days.get_indexer(dec)
    fwd_idx = np.minimum(pos + H, len(days) - 1)
    fwd = pd.DataFrame(lc.to_numpy()[fwd_idx] - lc.to_numpy()[pos], index=dec, columns=close.columns)
    fwd[pos + H > len(days) - 1] = np.nan

    rows = []
    sec_map = pd.Series({t: sector.get(t, "Unknown") for t in close.columns})
    for name, df in f.items():
        f[name] = df.reindex(dec)
    memd = mem.reindex(dec)
    for i, t in enumerate(dec):
        ok = memd.loc[t] & close.loc[t].notna()
        cols = ok[ok].index
        if len(cols) < 100:
            continue
        g = pd.DataFrame({n: df.loc[t, cols] for n, df in f.items()})
        g["fwd"] = fwd.loc[t, cols]
        g["sig"] = sig.loc[t, cols]
        g["sector"] = sec_map[cols].to_numpy()
        g["date"] = t
        rows.append(g)
    X = pd.concat(rows)
    X.index.name = "ticker"
    X = X.reset_index().replace([np.inf, -np.inf], np.nan)

    base = [c for c in f]
    grp = X.groupby("date")
    ranks = grp[base].rank(pct=True).add_prefix("xs_")
    # sector-relative momentum
    sec_mean = X.groupby(["date", "sector"])[["rel_5", "rel_21", "rel_63"]].transform("mean")
    secrel = (X[["rel_5", "rel_21", "rel_63"]] - sec_mean).add_prefix("sec_")
    mkt = pd.DataFrame({
        "mkt_ret_5": (mkt_lc - mkt_lc.shift(5)), "mkt_ret_21": (mkt_lc - mkt_lc.shift(21)),
        "mkt_ret_63": (mkt_lc - mkt_lc.shift(63)), "mkt_vol_21": np.log(mkt_r.rolling(21).std() + EPS),
        "dispersion_21": r1.where(mem).rolling(21).std().median(axis=1),
    }).reindex(dec)
    X = pd.concat([X, ranks, secrel], axis=1).merge(mkt, left_on="date", right_index=True, how="left")
    X["target"] = ((X["fwd"] - X.groupby("date")["fwd"].transform("mean")) / (X["sig"] * np.sqrt(H))).clip(-4, 4)
    features = base + list(ranks.columns) + list(secrel.columns) + list(mkt.columns)
    coverage = float(memd.loc[memd.index >= "2006-01-01"].sum().sum())
    return X, features, coverage
