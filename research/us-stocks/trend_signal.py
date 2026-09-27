"""Monthly ETF trend-following allocation — run once a month.

    python trend_signal.py --capital 10000            # 100% trend strategy
    python trend_signal.py --capital 10000 --spy 0.5  # 50% SPY buy & hold + 50% trend

Rule (strategy A in run_trend.py, backtested 2007-2026):
- 8 ETFs: SPY EFA EEM TLT IEF GLD DBC VNQ, each gets 1/8 of the trend sleeve;
- signal = average of the signs of the 1, 3, 6, 12-month total returns;
  holding = 1/8 × max(signal, 0) (so 0, 1/4, 1/2, 3/4 or all of the slice);
- everything not invested goes to SHY (1-3 year Treasuries);
- rebalance on the last trading day of each month (or first day of the next).

Uses the latest Yahoo prices (fresh download, no cache). Signals use month-end
closes; if you run it mid-month it also shows where the signal stands now.
"""
from __future__ import annotations

import argparse
import json
import urllib.request

import numpy as np
import pandas as pd

ASSETS = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
CASH = "SHY"
NAMES = {"SPY": "美股大盘", "EFA": "欧日等发达市场股", "EEM": "新兴市场股", "TLT": "20年+美债",
         "IEF": "7-10年美债", "GLD": "黄金", "DBC": "大宗商品", "VNQ": "美国房地产信托", "SHY": "1-3年短债(现金)"}


def fetch(sym: str) -> pd.Series:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=2y&interval=1d"
           f"&events=div%2Csplits&includeAdjustedClose=true")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.load(r)["chart"]["result"][0]
    idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
    adj = pd.Series(res["indicators"]["adjclose"][0]["adjclose"], index=idx, name=sym)
    raw = pd.Series(res["indicators"]["quote"][0]["close"], index=idx, name=sym)
    return pd.concat([adj.rename("adj"), raw.rename("close")], axis=1).dropna()


def signals(adj: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    m = adj[adj.index <= asof].resample("ME").last()
    m.iloc[-1] = adj[adj.index <= asof].iloc[-1]  # last point = as-of close
    rows = {}
    for s in ASSETS:
        r = {f"{k}个月": m[s].iloc[-1] / m[s].iloc[-1 - k] - 1 for k in (1, 3, 6, 12)}
        sig = np.mean([np.sign(v) for v in r.values()])
        rows[s] = {**r, "信号": sig, "权重": max(sig, 0) / len(ASSETS)}
    return pd.DataFrame(rows).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--spy", type=float, default=0.0, help="fraction kept in SPY buy & hold (0-1)")
    args = ap.parse_args()

    data = {s: fetch(s) for s in ASSETS + [CASH]}
    adj = pd.DataFrame({s: d["adj"] for s, d in data.items()}).dropna()
    last_px = {s: float(d["close"].iloc[-1]) for s, d in data.items()}
    last_day = adj.index[-1]
    month_end = adj.index[adj.index.to_period("M") < last_day.to_period("M")][-1]
    is_month_end = (last_day + pd.offsets.BDay(1)).month != last_day.month

    asof = last_day if is_month_end else month_end
    sig = signals(adj, asof)
    trend_w = sig["权重"].to_dict()
    trend_w[CASH] = 1 - sum(trend_w.values())
    final = {s: (1 - args.spy) * w for s, w in trend_w.items()}
    final["SPY"] = final.get("SPY", 0) + args.spy

    pd.set_option("display.width", 160)
    print(f"数据截至 {last_day:%Y-%m-%d}；信号基于 {asof:%Y-%m-%d} 收盘"
          f"{'（本月最后交易日，可以调仓）' if is_month_end else '（上月末，本月持仓）'}")
    show = sig.copy()
    for c in ("1个月", "3个月", "6个月", "12个月", "权重"):
        show[c] = (show[c] * 100).map(lambda v: f"{v:+.1f}%" if c != "权重" else f"{v:.1f}%")
    show["信号"] = sig["信号"].map(lambda v: f"{v:+.2f}")
    show.insert(0, "资产", [NAMES[s] for s in show.index])
    print(show.to_string())

    print(f"\n目标持仓（总资金 {args.capital:,.0f} 美元，其中 SPY 长期持有 {args.spy:.0%}，趋势部分 {1 - args.spy:.0%}）：")
    rows = []
    for s in ASSETS + [CASH]:
        w = final.get(s, 0.0)
        if w < 1e-9:
            continue
        dollars = w * args.capital
        rows.append({"ETF": s, "资产": NAMES[s], "比例": f"{w:.1%}", "金额": f"{dollars:,.0f}",
                     "最新价": f"{last_px[s]:.2f}", "约合股数": int(dollars // last_px[s])})
    print(pd.DataFrame(rows).to_string(index=False))

    if not is_month_end:
        now = signals(adj, last_day)
        changed = [s for s in ASSETS if abs(now.loc[s, "权重"] - sig.loc[s, "权重"]) > 1e-9]
        if changed:
            print("\n提示：按今天的价格，下列资产的信号已经变化，月底调仓时可能会变动：" +
                  "、".join(f"{s}({NAMES[s]})" for s in changed))
        else:
            print("\n提示：按今天的价格，所有信号和月初一致。")
    print("\n规则：每月最后一个交易日收盘前后调仓一次，其余时间不动。这不是投资建议。")


if __name__ == "__main__":
    main()
