"""Monthly trades and P&L of the recommended setting (risk 5%/unit, cap 5x), 1000 USD from 2025-01-01."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lgbm_btc.data import load_funding, load_klines  # noqa: E402
from run_pyramid_risk import simulate  # noqa: E402

CFG = dict(n=20, m=2.0, risk=0.05, k=0.5, filter=False, lev_cap=5)
for sym in ("BTC", "ETH"):
    px = load_klines(f"{sym}USDT", "1h", "spot", start="2023-06", end="2026-08-31")
    fu = load_funding(f"{sym}USDT", start="2024-10", end="2026-08-31")
    fu.index = fu.index.floor("h")
    eq, tr = simulate(px, fu, CFG, "2025-01-01")
    tr["open_m"] = tr["open"].dt.strftime("%Y-%m")
    tr["close_m"] = tr["close"].dt.strftime("%Y-%m")
    months = pd.period_range("2025-01", "2026-08", freq="M").strftime("%Y-%m")
    me = eq.resample("ME").last()
    me.index = me.index.strftime("%Y-%m")
    start_eq = me.shift(1).fillna(1000.0)
    rep = pd.DataFrame(index=months)
    rep["开单"] = tr.groupby("open_m").size().reindex(months).fillna(0).astype(int)
    rep["平仓"] = tr[tr["exit"] != "still open"].groupby("close_m").size().reindex(months).fillna(0).astype(int)
    closed = tr[tr["exit"] != "still open"]
    rep["盈利笔"] = closed[closed.pnl > 0].groupby("close_m").size().reindex(months).fillna(0).astype(int)
    rep["已实现盈亏"] = closed.groupby("close_m")["pnl"].sum().reindex(months).fillna(0.0).round(0)
    rep["月末资金"] = me.reindex(months).round(0)
    rep["当月盈亏(含浮盈)"] = (me - start_eq).reindex(months).round(0)
    rep["当月%"] = ((me / start_eq - 1) * 100).reindex(months).round(1)
    print(f"\n===== {sym} =====")
    print(rep.to_string())
    print("合计: 开单", rep["开单"].sum(), " 已实现", round(closed.pnl.sum()), " 期末", round(eq.iloc[-1]),
          f" 最大回撤 {(eq / eq.cummax() - 1).min():.0%}")
    t = tr.copy()
    t["pnl"] = t.pnl.round(0)
    print(t[["open", "close", "tranches", "pnl", "exit"]].to_string(index=False))
