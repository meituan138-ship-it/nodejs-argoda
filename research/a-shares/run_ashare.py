"""Whole-market A-share stock selection with LightGBM — long-only, executable.

    python run_ashare.py

Pre-declared rules (fixed before any result was seen):
- universe: every A-share code incl. later-delisted ones; on each decision
  date a stock needs >120 trading days of history, a trade that day and a
  20-day average traded amount >= 20M CNY;
- signal at the last close of each week; trade at the next open;
- LightGBM regression on the cross-sectional rank of the next-week
  open-to-open return; 300 trees, lr 0.02, 31 leaves, min_data_in_leaf 2000,
  2 seeds; retrained every quarter on all data from 2013 up to two weeks
  before the quarter (label purge + embargo). Test 2016-01 → 2026-09;
- portfolio: equal-weight top 50 (and top 100) predictions, long only
  (retail cannot short single stocks); a stock that opens at its limit-up
  cannot be bought, one that opens at limit-down or is suspended cannot be
  sold and is carried; conservative ST rule: any open gap >= +4.5% with
  open == high blocks a buy, <= -4.5% with open == low blocks a sell;
- costs: commission 0.025% per side, stamp duty on sells 0.1% (0.05% from
  2023-08-28), slippage 0.10% per side (0.20% in the stress case);
- benchmarks: CSI 300 / 500 / 1000 price indices, equal-weight universe,
  and single classic factors (small size, 1-month reversal, low volatility).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lightgbm-btc"))
from ash.data import load_all, load_index  # noqa: E402
from ash.panel import build  # noqa: E402
from lgbm_btc.model import fit_predict  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
TEST_START = "2016-01-01"
PARAMS = {"min_data_in_leaf": 2000}


def walk_forward(X: pd.DataFrame, feats: list[str]) -> np.ndarray:
    pred = np.full(len(X), np.nan)
    d = X["date"]
    ok = X["target"].notna().to_numpy()
    qs = pd.date_range(TEST_START, d.max() + pd.offsets.QuarterBegin(1), freq="QS")
    for a, b in zip(qs[:-1], qs[1:]):
        cut = a - pd.Timedelta(days=14)
        tr = np.where(ok & (d >= "2013-01-01").to_numpy() & (d < cut).to_numpy())[0]
        te = np.where(((d >= a) & (d < b)).to_numpy())[0]
        if not len(te):
            continue
        Xtr, ytr = X.iloc[tr][feats], X["target"].to_numpy()[tr]
        p, _ = fit_predict(Xtr, ytr, Xtr.iloc[:0], ytr[:0], X.iloc[te][feats], params=PARAMS, seeds=(0, 1), n_trees=300)
        pred[te] = p
        print(f"  {a:%Y-%m} train={len(tr)} test={len(te)}", flush=True)
    return pred


def backtest(X: pd.DataFrame, col: str, ex: dict, n: int = 50, slip: float = 0.001) -> pd.DataFrame:
    o, c, h, lim, days = ex["open"], ex["close"], ex["high"], ex["lim"], ex["days"]
    lo = None
    px = o.combine_first(c.ffill())            # value a position at the open; suspended = last price
    prev_c = c.ffill().shift(1)
    gap = o / prev_c - 1
    no_buy = (gap >= lim - 0.002) | ((gap >= 0.045) & (o >= ex["high"] - 1e-9)) | o.isna()
    low = ex.get("low")
    dates = sorted(X["date"].unique())
    pos_of = {t: days.get_loc(t) for t in dates}
    w = pd.Series(dtype=float)
    rows = []
    for t, t_next in zip(dates[:-1], dates[1:]):
        e, e2 = pos_of[t] + 1, pos_of[t_next] + 1
        if e2 >= len(days):
            break
        g = X.loc[X["date"] == t, ["code", col]].dropna().set_index("code")[col]
        if len(g) < n:
            continue
        target = set(g.sort_values(ascending=False).index[:n])
        de = days[e]
        stamp = 0.001 if de < pd.Timestamp("2023-08-28") else 0.0005
        o_e = o.iloc[e]
        blocked_sell = o_e.isna() | (gap.iloc[e] <= -(lim.iloc[e] - 0.002)) | (gap.iloc[e] <= -0.045)
        keep = [s for s in w.index if s in target or blocked_sell.get(s, True)]
        sells = [s for s in w.index if s not in keep]
        buys = [s for s in target if s not in w.index and not no_buy.iloc[e].get(s, True)]
        hold = keep + buys
        new = pd.Series(1.0 / len(hold), index=hold) if hold else pd.Series(dtype=float)
        stuck = [s for s in keep if s not in target]       # unsellable, keep drifted weight
        if stuck:
            new[stuck] = w[stuck]
            free = 1 - w[stuck].sum()
            others = [s for s in hold if s not in stuck]
            if others:
                new[others] = max(free, 0) / len(others)
        dw = new.sub(w, fill_value=0.0)
        cost = float(dw.clip(lower=0).sum() * (0.00025 + slip) + (-dw.clip(upper=0)).sum() * (0.00025 + stamp + slip))
        r = (px.iloc[e2].reindex(new.index) / px.iloc[e].reindex(new.index) - 1).fillna(0.0)
        gross = float((new * r).sum())
        rows.append({"date": de, "gross": gross, "net": gross - cost, "turnover": float(dw.abs().sum()),
                     "n": len(new), "stuck": len(stuck)})
        w = new * (1 + r)
        w = w / w.sum() if w.sum() > 0 else w
    return pd.DataFrame(rows).set_index("date")


def stats(r: pd.Series, per_year: float = 52) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / per_year
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(per_year)),
            "max_dd": float((eq / eq.cummax() - 1).min()), "win_weeks": float((r > 0).mean())}


def main() -> None:
    stocks = load_all()
    X, feats, ex = build(stocks)
    print(f"stocks={len(stocks)} rows={len(X)} features={len(feats)} per-week={X.groupby('date').size().mean():.0f}")
    cache = RES / "ashare_preds.parquet"
    if cache.exists():
        X["lgbm"] = pd.read_parquet(cache)["lgbm"].to_numpy()
    else:
        X["lgbm"] = walk_forward(X, feats)
        RES.mkdir(exist_ok=True)
        X[["lgbm"]].to_parquet(cache)
    X["small_size"] = -X["log_amount_20"]
    X["reversal_1m"] = -X["ret_20"]
    X["low_vol"] = -X["vol_20"]
    T = X[X["lgbm"].notna()].copy()

    out = {"stocks_total": len(stocks), "stocks_per_week": float(T.groupby("date").size().mean()),
           "weeks": int(T["date"].nunique())}
    ic = {}
    for col in ("lgbm", "small_size", "reversal_1m", "low_vol"):
        s = T.groupby("date").apply(lambda g: g[col].corr(g["target"], method="spearman"), include_groups=False)
        ic[col] = {"mean": float(s.mean()), "tstat": float(s.mean() / s.std() * np.sqrt(len(s))),
                   "by_year": {int(y): float(s[s.index.year == y].mean()) for y in sorted(set(s.index.year))}}
    out["ic"] = ic

    series = {}
    for name, col, n, slip in (("LGBM top50", "lgbm", 50, 0.001), ("LGBM top100", "lgbm", 100, 0.001),
                               ("LGBM top50 滑点0.2%", "lgbm", 50, 0.002), ("小市值 top50", "small_size", 50, 0.001),
                               ("1月反转 top50", "reversal_1m", 50, 0.001), ("低波动 top50", "low_vol", 50, 0.001)):
        bt = backtest(T, col, ex, n, slip)
        series[name] = bt["net"]
        out.setdefault("portfolios", {})[name] = {**stats(bt["net"]), "gross_cagr": stats(bt["gross"])["cagr"],
                                                  "turnover_per_week": float(bt["turnover"].mean()),
                                                  "avg_stuck": float(bt["stuck"].mean())}
        print(name, out["portfolios"][name], flush=True)
    # benchmarks on the same schedule (open-to-open of the next trading day)
    days = ex["days"]
    dates = [pd.Timestamp(d) for d in series["LGBM top50"].index]
    for code, nm in (("sh000300", "沪深300"), ("sh000905", "中证500"), ("sh000852", "中证1000")):
        idx = load_index(code)
        if idx is None:
            continue
        op = idx["open"].reindex(days).ffill()
        r = pd.Series(op.shift(-0).to_numpy(), index=days)
        rr = []
        for d0, d1 in zip(dates[:-1], dates[1:]):
            rr.append(r.loc[d1] / r.loc[d0] - 1)
        series[nm] = pd.Series(rr + [np.nan], index=dates)
        out.setdefault("benchmarks", {})[nm] = stats(series[nm])
    df = pd.DataFrame(series)
    out["by_year"] = {k: {int(y): float((1 + v[v.index.year == y].dropna()).prod() - 1)
                          for y in sorted(set(v.index.year))} for k, v in df.items()}
    for bench in ("中证1000", "中证500"):
        if bench in df:
            exr = (df["LGBM top50"] - df[bench]).dropna()
            out.setdefault("excess_top50", {})[bench] = {
                "ann": float(exr.mean() * 52), "tstat": float(exr.mean() / exr.std() * np.sqrt(len(exr))),
                "by_year": {int(y): float(exr[exr.index.year == y].sum()) for y in sorted(set(exr.index.year))}}
    df.to_csv(RES / "ashare_weekly_returns.csv", float_format="%.6f")
    (RES / "ashare_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
