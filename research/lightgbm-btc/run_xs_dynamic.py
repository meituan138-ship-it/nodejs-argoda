"""Stress test of the multi-coin result: point-in-time universe including
dead coins, real perp funding, liquidity-dependent slippage, and a slower
3-day holding variant.

    python run_xs_dynamic.py

Same model and features as run_xs.py (decided before this run). Changes:
- universe = each month's top 30 of ~70 candidates by trailing 30-day volume,
  only coins that had a USD-M perp at the time (lgbm_btc/universe.py);
- a coin delisted during a holding period is exited at its last price;
- costs: 5 bp fee + slippage = clip(2·sqrt(1e9 / ADV), 1, 30) bp per side,
  where ADV is the coin's trailing 30-day average daily USDT volume
  (≈1 bp for BTC, ≈9 bp at $50M/day, 20 bp at $10M/day);
- funding: every settlement inside the 24h holding window is charged
  (longs pay positive funding, shorts receive it).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import run_xs as R
from lgbm_btc.universe import load_candidates, monthly_universe
from lgbm_btc.xs import H, build_panel

R.TRAIN_EVERY_H = 8  # ~70 candidates: halve the training rows to keep runtime similar


def funding_window(funding: dict[str, pd.Series], times: pd.DatetimeIndex) -> pd.DataFrame:
    """Sum of settlements in (t, t+24h] per coin for each decision time t."""
    out = {}
    for c, f in funding.items():
        cum = f.cumsum()
        a = cum.reindex(times, method="ffill").fillna(0.0).to_numpy()
        b = cum.reindex(times + pd.Timedelta(hours=H), method="ffill").fillna(0.0).to_numpy()
        out[c] = b - a
    return pd.DataFrame(out, index=times)


def simulate(D: pd.DataFrame, col: str, fund: pd.DataFrame, hold_days: int = 1,
             fee_bps: float = 5.0, slip: bool = True, use_funding: bool = True, q: float = 0.2) -> pd.DataFrame:
    """Daily long/short book; with hold_days>1 the book is the average of the last
    hold_days target books (overlapping sub-portfolios, lower turnover)."""
    books, rows, prev = [], [], pd.Series(dtype=float)
    for t, g in D.groupby(level=0):
        g = g.dropna(subset=[col]).set_index("coin")
        if len(g) < 10:
            continue
        k = max(1, int(round(q * len(g))))
        o = g[col].sort_values()
        target = pd.Series(0.0, index=g.index)
        target[o.index[-k:]] = 0.5 / k
        target[o.index[:k]] = -0.5 / k
        books = (books + [target])[-hold_days:]
        w = pd.concat(books, axis=1).fillna(0.0).mean(axis=1)
        w = w[w != 0]
        # every held coin needs a realised return; the panel only has rows for current members,
        # so held non-members fall back to their last known forward return (rare)
        ret = D.loc[[t]].set_index("coin")["fwd_ret"].reindex(w.index)
        simple = (np.exp(ret) - 1).fillna(0.0)
        gross = float((w * simple).sum())
        dw = w.sub(prev, fill_value=0.0).abs()
        adv = D.loc[[t]].set_index("coin")["adv30"].reindex(dw.index)
        slip_bps = (2 * np.sqrt(1e9 / adv.clip(lower=1e6))).clip(1, 30).fillna(30) if slip else 0.0
        cost = float((dw * (fee_bps + slip_bps)).sum()) * 1e-4
        fnd = float((w * fund.loc[t].reindex(w.index).fillna(0.0)).sum()) if use_funding else 0.0
        prev = w
        rows.append({"time": t, "gross": gross, "fee_slip": cost, "funding": fnd,
                     "net": gross - cost - fnd, "turnover": float(dw.sum())})
    return pd.DataFrame(rows).set_index("time")


def main() -> None:
    klines, funding = load_candidates()
    start = min(d.index[0] for d in klines.values())
    grid = pd.date_range(start, pd.Timestamp("2026-08-31 23:00").tz_localize(start.tz), freq="h")
    member = monthly_universe(klines, funding, grid)
    X, features = build_panel(klines, eligible=member)
    X = X.set_index("coin", append=True)
    n_members = member.loc["2021":].resample("MS").first().sum(axis=1)
    ever = member.loc["2021":].any()
    print(f"rows={len(X)} candidates={len(klines)} ever-in-universe(2021+)={int(ever.sum())} "
          f"members/month={n_members.mean():.1f}")

    cache = R.RESULTS / "cache" / "xs_dyn_preds.parquet"
    if cache.exists():
        P = pd.read_parquet(cache)
    else:
        Xr = X.reset_index(level=1)
        P = pd.DataFrame({"lgbm": R.walk_forward(Xr, features, "lgbm").to_numpy()}, index=X.index)
        P.to_parquet(cache)
    D = X[["target", "fwd_ret", "adv30"]].join(P)
    D = D[D["lgbm"].notna()].reset_index(level=1)
    times = D.index.unique()
    fund = funding_window(funding, times)

    ic = R.daily_ic(D, "lgbm")
    out = {
        "universe": {"candidates": len(klines), "ever_member_2021plus": sorted(ever[ever].index),
                     "avg_members": float(n_members.mean()),
                     "dead_or_delisted_members": sorted(c for c in ever[ever].index
                                                        if klines[c].index[-1] < pd.Timestamp("2026-08-01", tz="UTC"))},
        "ic": {"mean": float(ic.mean()), "tstat": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
               "by_year": {int(y): float(ic[ic.index.year == y].mean()) for y in sorted(set(ic.index.year))}},
    }
    steps = {
        "1_gross": dict(fee_bps=0.0, slip=False, use_funding=False),
        "2_fee_5bp": dict(fee_bps=5.0, slip=False, use_funding=False),
        "3_fee+slippage": dict(fee_bps=5.0, slip=True, use_funding=False),
        "4_fee+slippage+funding": dict(fee_bps=5.0, slip=True, use_funding=True),
    }
    res = {}
    for hold in (1, 3):
        for name, kw in steps.items():
            bt = simulate(D, "lgbm", fund, hold_days=hold, **kw)
            s = R.stats(bt["net"])
            s.update({"turnover_per_day": float(bt["turnover"].mean()),
                      "fee_slip_per_year": float(bt["fee_slip"].mean() * 365),
                      "funding_per_year": float(bt["funding"].mean() * 365),
                      "win_rate_days": float((bt["net"] > 0).mean()),
                      "pl_ratio_days": float(bt["net"][bt["net"] > 0].mean() / -bt["net"][bt["net"] < 0].mean())})
            if name.startswith("4"):
                s["by_year_sharpe"] = {int(y): R.stats(bt.loc[bt.index.year == y, "net"])["sharpe"]
                                       for y in sorted(set(bt.index.year))}
                bt["net"].to_csv(R.RESULTS / f"xs_dyn_daily_hold{hold}.csv")
            res[f"hold{hold}d/{name}"] = s
            print(f"hold {hold}d {name:24s} sharpe {s['sharpe']:+.2f} cagr {s['cagr']:+.1%} "
                  f"mdd {s['max_dd']:+.1%}", flush=True)
    out["backtest"] = res
    (R.RESULTS / "xs_dynamic_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
