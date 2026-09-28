"""Same LightGBM predictions, simulated for a 10,000 CNY account.

    python run_small_capital.py

Real small-account constraints:
- whole lots of 100 shares, sized with the raw (unadjusted) open price;
- only stocks whose lot costs <= capital / N are eligible, so N positions fit;
- commission max(0.025% × amount, 5 CNY) per trade (typical broker), or
  0.025% with no minimum (brokers offering 免五), stamp duty on sells,
  0.1% slippage per side;
- positions whose ranking stays inside the top 2N are kept (lower turnover);
- same limit-up / limit-down / suspension rules as run_ashare.py;
- leftover cash earns nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lightgbm-btc"))
from ash.data import load_all  # noqa: E402
from ash.panel import build, wide  # noqa: E402

RES = Path(__file__).resolve().parent / "results"


def simulate(T, ex, raw_open, capital0=10_000.0, n=5, min_comm=5.0, slip=0.001, keep_mult=2):
    o, c, lim, days = ex["open"], ex["close"], ex["lim"], ex["days"]
    px = o.combine_first(c.ffill())
    gap = o / c.ffill().shift(1) - 1
    no_buy = (gap >= lim - 0.002) | ((gap >= 0.045) & (o >= ex["high"] - 1e-9)) | o.isna()
    no_sell = o.isna() | (gap <= -(lim - 0.002)) | (gap <= -0.045)
    dates = sorted(T["date"].unique())
    pos_of = {t: days.get_loc(t) for t in dates}
    cash, shares = capital0, {}            # shares in lots of raw shares; value tracked via hfq ratio
    hold_val = {}                          # current market value of each position (CNY)
    rows = []

    def comm(amount):
        return max(amount * 0.00025, min_comm) if amount > 0 else 0.0

    for t, t_next in zip(dates[:-1], dates[1:]):
        e, e2 = pos_of[t] + 1, pos_of[t_next] + 1
        if e2 >= len(days):
            break
        de = days[e]
        stamp = 0.001 if de < pd.Timestamp("2023-08-28") else 0.0005
        equity = cash + sum(hold_val.values())
        g = T.loc[T["date"] == t, ["code", "lgbm"]].dropna().set_index("code")["lgbm"].sort_values(ascending=False)
        ro = raw_open.iloc[e]
        keep_set = set(g.index[: n * keep_mult])
        # sells
        for s in list(hold_val):
            if s not in keep_set and not no_sell.iloc[e].get(s, True):
                amt = hold_val.pop(s)
                cash += amt * (1 - slip) - comm(amt) - amt * stamp
        # buys: best-ranked affordable names until N positions
        budget = equity / n
        for s in g.index:
            if len(hold_val) >= n:
                break
            if s in hold_val or no_buy.iloc[e].get(s, True):
                continue
            p = ro.get(s, np.nan)
            if not np.isfinite(p) or p <= 0:
                continue
            lots = int(min(budget, cash) // (p * 100 * (1 + slip) + 0.0))
            if lots < 1:
                continue
            amt = lots * 100 * p
            cost = amt * slip + comm(amt)
            if amt + cost > cash:
                continue
            cash -= amt + cost
            hold_val[s] = amt
        # mark to market to the next rebalance open (hfq ratio = true total return)
        for s in list(hold_val):
            r = px.iloc[e2].get(s, np.nan) / px.iloc[e].get(s, np.nan)
            hold_val[s] *= r if np.isfinite(r) else 1.0
        rows.append({"date": de, "equity": cash + sum(hold_val.values()), "n": len(hold_val), "cash": cash})
    return pd.DataFrame(rows).set_index("date")


def summary(bt, capital0=10_000.0):
    eq = bt["equity"]
    r = eq.pct_change().fillna(eq.iloc[0] / capital0 - 1)
    yrs = len(r) / 52
    by_year = {int(y): float(eq[eq.index.year == y].iloc[-1] / (eq[eq.index.year < y].iloc[-1] if (eq.index.year < y).any() else capital0) - 1)
               for y in sorted(set(eq.index.year))}
    return {"final_equity": float(eq.iloc[-1]), "cagr": float((eq.iloc[-1] / capital0) ** (1 / yrs) - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(52)), "max_dd": float((eq / eq.cummax() - 1).min()),
            "avg_positions": float(bt["n"].mean()), "avg_cash_share": float((bt["cash"] / bt["equity"]).mean()),
            "by_year": by_year}


def main():
    stocks = load_all()
    X, feats, ex = build(stocks)
    X["lgbm"] = pd.read_parquet(RES / "ashare_preds.parquet")["lgbm"].to_numpy()
    T = X[X["lgbm"].notna()]
    raw_open = wide(stocks, "raw_open").reindex(ex["days"])
    out = {}
    for n in (5, 8, 10):
        for mc, label in ((5.0, "最低5元佣金"), (0.0, "免五")):
            name = f"持有{n}只·{label}"
            s = summary(simulate(T, ex, raw_open, n=n, min_comm=mc))
            out[name] = s
            print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "by_year"},
                  {y: round(v * 100, 1) for y, v in s["by_year"].items()}, flush=True)
    (RES / "small_capital_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
