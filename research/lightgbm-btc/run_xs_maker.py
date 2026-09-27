"""Multi-coin model with realistic execution: maker (limit) orders that only
fill if the market trades *through* the limit price, plus lower turnover.

    python run_xs_maker.py

Uses the cached walk-forward predictions of run_xs_dynamic.py (same model,
same point-in-time universe). Execution rules, fixed before running:
- decision at the close of the 00:00 UTC bar; limit order at that close P;
- a buy fills if some low in the next F=4 hourly bars is <= P·(1 − 1bp)
  (price must trade through the limit: queue position is not free);
  a sell fills if some high >= P·(1 + 1bp);
- "skip": unfilled orders are cancelled (position stays as it was);
  "chase": unfilled orders are sent as taker at the close of bar t+F,
  paying taker fee + the liquidity slippage model;
- hysteresis: keep a long while its rank stays in the top 35% (short:
  bottom 35%), refill the book from the top / bottom 20%;
- maker fee 0 bp (MEXC futures) or 1 bp (reported API maker rate);
  taker 2 bp + clip(2·sqrt(1e9/ADV), 1, 30) bp slippage;
- funding charged on every held position (real settlement history).
Adverse selection is built in: fills happen exactly when price moves
against the order first.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_xs as R
import run_xs_dynamic as Dy
from lgbm_btc.universe import load_candidates, monthly_universe
from lgbm_btc.xs import build_panel

F = 4
THROUGH = 1e-4


def main() -> None:
    klines, funding = load_candidates()
    start = min(d.index[0] for d in klines.values())
    grid = pd.date_range(start, pd.Timestamp("2026-08-31 23:00").tz_localize(start.tz), freq="h")
    member = monthly_universe(klines, funding, grid)
    X, _ = build_panel(klines, eligible=member)
    X = X.set_index("coin", append=True)
    P = pd.read_parquet(R.RESULTS / "cache" / "xs_dyn_preds.parquet")
    D = X[["adv30"]].join(P)
    D = D[D["lgbm"].notna()].reset_index(level=1)

    close = pd.DataFrame({c: d["close"] for c, d in klines.items()}).reindex(grid).ffill()
    high = pd.DataFrame({c: d["high"] for c, d in klines.items()}).reindex(grid)
    low = pd.DataFrame({c: d["low"] for c, d in klines.items()}).reindex(grid)
    adv = pd.DataFrame({c: d["quote_volume"] for c, d in klines.items()}).reindex(grid).fillna(0) \
        .rolling(720, min_periods=240).sum() / 30
    times = D.index.unique().sort_values()
    fund = Dy.funding_window(funding, times)
    pos = {t: grid.get_loc(t) for t in times}

    def slip_bps(c, t):
        a = adv.at[t, c]
        return float(np.clip(2 * np.sqrt(1e9 / max(a, 1e6)), 1, 30)) if np.isfinite(a) else 30.0

    def simulate(mode: str, hysteresis: bool, maker_fee: float = 0.0):
        w = pd.Series(dtype=float)
        rows, fills, orders = [], 0, 0
        for t in times:
            i = pos[t]
            if i + 24 >= len(grid):
                break
            g = D.loc[[t]].set_index("coin")["lgbm"].dropna()
            if len(g) < 10:
                continue
            rk = g.rank(pct=True)
            k = max(1, int(round(0.2 * len(g))))
            if hysteresis:
                longs = [c for c in w.index if w[c] > 0 and rk.get(c, 0) >= 0.65]
                shorts = [c for c in w.index if w[c] < 0 and rk.get(c, 1) <= 0.35]
                longs += [c for c in rk.sort_values(ascending=False).index if c not in longs][: max(0, k - len(longs))]
                shorts += [c for c in rk.sort_values().index if c not in shorts][: max(0, k - len(shorts))]
            else:
                longs = list(rk.sort_values().index[-k:])
                shorts = list(rk.sort_values().index[:k])
            target = pd.Series(0.0, index=sorted(set(w.index) | set(longs) | set(shorts)))
            target[longs] = 0.5 / len(longs)
            target[shorts] = -0.5 / len(shorts)
            cur = w.reindex(target.index, fill_value=0.0)
            new = cur.copy()
            p0 = close.iloc[i]
            p24 = close.iloc[i + 24]
            pF = close.iloc[i + F]
            lo = low.iloc[i + 1:i + F + 1].min()
            hi = high.iloc[i + 1:i + F + 1].max()
            cost, adj = 0.0, 0.0
            for c in target.index:
                dw = target[c] - cur[c]
                if abs(dw) < 1e-12:
                    continue
                if mode == "taker":
                    new[c] = target[c]
                    cost += abs(dw) * (2 + slip_bps(c, t)) * 1e-4
                    continue
                orders += 1
                buy = dw > 0
                filled = (lo.get(c, np.nan) <= p0[c] * (1 - THROUGH)) if buy else (hi.get(c, np.nan) >= p0[c] * (1 + THROUGH))
                if filled:
                    fills += 1
                    new[c] = target[c]
                    cost += abs(dw) * maker_fee * 1e-4
                elif mode == "chase":
                    new[c] = target[c]
                    cost += abs(dw) * (2 + slip_bps(c, t)) * 1e-4
                    # the chased slice only earns the move from t+F, not from t
                    adj += dw * ((p24[c] / pF[c]) - (p24[c] / p0[c]))
            new = new[new.abs() > 1e-12]
            ret = (p24.reindex(new.index) / p0.reindex(new.index) - 1).fillna(0.0)
            gross = float((new * ret).sum()) + adj
            fnd = float((new * fund.loc[t].reindex(new.index).fillna(0.0)).sum())
            rows.append({"time": t, "net": gross - cost - fnd, "gross": gross, "cost": cost,
                         "turnover": float((new - cur.reindex(new.index, fill_value=0.0)).abs().sum()
                                           + cur[~cur.index.isin(new.index)].abs().sum())})
            w = new
        bt = pd.DataFrame(rows).set_index("time")
        return bt, (fills / orders if orders else 1.0)

    out = {}
    variants = {
        "旧：每天全换·吃单": ("taker", False, 0.0),
        "降换手·吃单": ("taker", True, 0.0),
        "降换手·挂单0bp·未成交放弃": ("skip", True, 0.0),
        "降换手·挂单0bp·未成交4h后追单": ("chase", True, 0.0),
        "降换手·挂单1bp(API)·未成交放弃": ("skip", True, 1.0),
    }
    for name, (mode, hyst, mf) in variants.items():
        bt, fill_rate = simulate(mode, hyst, mf)
        r = bt["net"]
        s = R.stats(r)
        rec = r[r.index >= "2025-01-01"]
        r26 = r[r.index >= "2026-01-01"]
        eq26 = 1000 * (1 + r26).cumprod()
        res = {**s, "fill_rate": fill_rate, "turnover_per_day": float(bt["turnover"].mean()),
               "sharpe_2025_26": float(rec.mean() / rec.std() * np.sqrt(365)), "ann_2025_26": float(rec.mean() * 365),
               "usd1000_2026_end_1x": float(eq26.iloc[-1]),
               "by_year": {int(y): float((1 + r[r.index.year == y]).prod() - 1) for y in sorted(set(r.index.year))},
               "months_2026": {k: float(v) for k, v in ((1 + r26).groupby(r26.index.strftime("%m")).prod() - 1).items()}}
        out[name] = res
        print(f"{name:26s} fill {fill_rate:.0%} turn/day {res['turnover_per_day']:.2f} CAGR {s['cagr']:+.1%} "
              f"Sharpe {s['sharpe']:.2f} MDD {s['max_dd']:.1%} | 25-26 Sharpe {res['sharpe_2025_26']:+.2f} "
              f"| $1000→2026-08 ${res['usd1000_2026_end_1x']:.0f}", flush=True)
    (R.RESULTS / "xs_maker_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
