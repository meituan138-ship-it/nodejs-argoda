"""Multi-asset trend following with ETFs (retail-implementable), 2007 → today.

    python run_trend.py

Pre-declared rules (textbook versions, no tuning):
- assets: SPY (US stocks), EFA (developed ex-US), EEM (emerging), TLT (long
  Treasuries), IEF (7-10y Treasuries), GLD (gold), DBC (commodities),
  VNQ (US REITs); cash = SHY (1-3y Treasuries);
- signal (Moskowitz-Ooi-Pedersen / Faber style): average of the signs of
  the 1-, 3-, 6- and 12-month total returns, in [-1, 1];
- A "long-only trend": each asset gets 1/8 of capital × max(signal, 0);
  the rest sits in SHY. No shorting, no leverage — doable in any broker;
- B "vol-targeted trend": weight_i = signal_i × (10% / σ_i) / 8 with σ_i the
  60-day realised vol, long and short, gross capped at 2× — the
  managed-futures version (needs margin / shorting);
- monthly rebalance on the last trading day, 5 bp per side;
- benchmarks: SPY buy & hold, 60/40 (SPY/IEF, monthly rebalanced),
  equal-weight buy & hold of the 8 assets.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from usxs.data import _yahoo

ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
CASH = "SHY"
FEE = 5e-4
OUT = Path(__file__).resolve().parent / "results"


def monthly_prices() -> pd.DataFrame:
    px = pd.DataFrame({t: _yahoo(t, "2005-01-01")["adjclose"] for t in ASSETS + [CASH]}).dropna()
    return px


def run(weights: pd.DataFrame, rets: pd.DataFrame) -> pd.Series:
    """weights decided at month-end t, earn returns of month t+1."""
    w = weights.shift(1).fillna(0.0)
    turn = weights.diff().abs().sum(axis=1).shift(1).fillna(0.0)
    return (w * rets).sum(axis=1) - turn * FEE


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    yrs = len(r) / 12
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "vol": float(r.std() * np.sqrt(12)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(12)), "max_dd": float((eq / eq.cummax() - 1).min()),
            "worst_month": float(r.min())}


def main() -> None:
    px = monthly_prices()
    daily_r = px.pct_change()
    m = px.resample("ME").last()
    mr = m.pct_change()
    sig = sum(np.sign(m[ASSETS] / m[ASSETS].shift(k) - 1) for k in (1, 3, 6, 12)) / 4
    vol = (daily_r[ASSETS].rolling(60).std() * np.sqrt(252)).resample("ME").last()
    start = sig.dropna().index[0]

    wa = (sig.clip(lower=0) / len(ASSETS))
    wa[CASH] = 1 - wa.sum(axis=1)
    wb = sig * (0.10 / vol) / len(ASSETS)
    gross = wb.abs().sum(axis=1)
    wb = wb.div(np.maximum(gross / 2.0, 1.0), axis=0)
    wb[CASH] = 1 - wb.clip(lower=0).sum(axis=1).clip(upper=1)  # idle long capital earns T-bill-like yield
    w6040 = pd.DataFrame({"SPY": 0.6, "IEF": 0.4}, index=m.index).reindex(columns=m.columns, fill_value=0.0)
    wew = pd.DataFrame(1 / len(ASSETS), index=m.index, columns=ASSETS).reindex(columns=m.columns, fill_value=0.0)
    wspy = pd.DataFrame({"SPY": 1.0}, index=m.index).reindex(columns=m.columns, fill_value=0.0)

    strategies = {"A 趋势·只做多（ETF+现金）": wa, "B 趋势·波动率目标多空": wb, "SPY 买入持有": wspy,
                  "60/40 股债": w6040, "8 资产等权持有": wew}
    rets = {k: run(w.reindex(columns=m.columns, fill_value=0.0).loc[start:], mr.loc[start:].fillna(0.0)).iloc[1:]
            for k, w in strategies.items()}
    out = {"period": f"{rets['SPY 买入持有'].index[0]:%Y-%m} → {rets['SPY 买入持有'].index[-1]:%Y-%m}",
           "stats": {k: stats(r) for k, r in rets.items()},
           "by_year": {k: {int(y): float((1 + r[r.index.year == y]).prod() - 1) for y in sorted(set(r.index.year))}
                       for k, r in rets.items()},
           "corr_with_spy": {k: float(r.corr(rets["SPY 买入持有"])) for k, r in rets.items()}}
    # crisis windows
    for name, a, b in (("2008 金融危机", "2007-11", "2009-02"), ("2020 疫情", "2020-02", "2020-03"),
                       ("2022 股债双杀", "2022-01", "2022-12")):
        out.setdefault("crises", {})[name] = {k: float((1 + r.loc[a:b]).prod() - 1) for k, r in rets.items()}
    OUT.mkdir(exist_ok=True)
    (OUT / "trend_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    pd.DataFrame(rets).to_csv(OUT / "trend_monthly_returns.csv", float_format="%.6f")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
