"""A-share-listed ETF momentum rotation for a small (10,000 CNY) account.

    python run_etf.py

Pre-declared rules (the common "ETF 轮动" recipe, not tuned):
- basket (all listed in Shanghai / Shenzhen, tradable in any A-share account):
  沪深300 510300, 中证500 510500, 中证1000 512100, 创业板 159915, 科创50 588000,
  红利 510880, 黄金 518880, 纳指 513100, 标普500 513500, 恒生 159920,
  十年国债 511260; safe asset = 银华日利 511880 (money-market ETF);
- signal at the last close of each period: N-day total return (N = 60 main,
  20 as a variant), using back-adjusted prices; only ETFs with >= N days of
  history take part;
- hold the top-1 (or top-2) ETF if its return is > 0, otherwise the money ETF;
- trade at the next open; rebalance monthly (main) or weekly (variant);
- costs: commission max(0.01% × amount, 5 CNY) per trade (ETFs pay no stamp
  duty), 0.05% slippage per side; whole lots of 100 shares; 10,000 CNY start.
All variants are reported; benchmarks: buy & hold 510300 and 510500.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ash.data import fetch_stock

RES = Path(__file__).resolve().parent / "results"
BASKET = {"sh510300": "沪深300", "sh510500": "中证500", "sh512100": "中证1000", "sz159915": "创业板",
          "sh588000": "科创50", "sh510880": "红利", "sh518880": "黄金", "sh513100": "纳指",
          "sh513500": "标普500", "sz159920": "恒生", "sh511260": "十年国债"}
SAFE = "sh511880"


def load():
    data = {c: fetch_stock(c) for c in list(BASKET) + [SAFE]}
    o = pd.DataFrame({c: d["open"] for c, d in data.items() if d is not None}).sort_index()
    c = pd.DataFrame({c: d["close"] for c, d in data.items() if d is not None}).sort_index()
    raw_o = pd.DataFrame({k: d["raw_open"] for k, d in data.items() if d is not None}).sort_index()
    # suspended days (e.g. QDII premium halts): carry the last price forward
    return o.ffill(), c.ffill(), raw_o.ffill()


def simulate(o, c, raw_o, lookback=60, top=1, freq="ME", capital=10_000.0, min_comm=5.0, rate=0.0001, slip=0.0005,
             start="2014-01-01"):
    days = c.index[c.index >= start]
    periods = pd.Series(days, index=days).groupby(days.to_period(freq[0] if freq != "ME" else "M")).max()
    dec = list(periods.values)
    cash, pos = capital, {}  # pos: code -> shares
    hist = []
    for t, t2 in zip(dec[:-1], dec[1:]):
        i = c.index.get_loc(t)
        e, e2 = i + 1, c.index.get_loc(t2) + 1
        if e2 >= len(c.index):
            break
        lbs = (10, 20, 40, 60) if lookback == 0 else (lookback,)  # 0 = blend of several lookbacks
        if i < max(lbs):
            mom = pd.Series(dtype=float)
        else:
            ok = c[list(BASKET)].iloc[:i + 1].notna().sum() >= max(lbs) + 1
            rets = pd.DataFrame({lb: (c.iloc[i] / c.iloc[i - lb] - 1)[list(BASKET)] for lb in lbs})[ok].dropna()
            # rank-average across lookbacks; keep the mean return for the "> 0" cash filter
            mom = rets.rank(pct=True).mean(axis=1) - 0.5 + 1e-9 * rets.mean(axis=1)
            mom = mom.where(rets.mean(axis=1) > 0, -1.0)
        best = mom.sort_values(ascending=False).head(top)
        target = [s for s, v in best.items() if v > 0]
        n_safe = top - len(target)
        want = target + ([SAFE] if n_safe > 0 else [])
        weight = {s: 1 / top for s in target}
        if n_safe > 0:
            weight[SAFE] = n_safe / top
        po = raw_o.iloc[e]
        # value at today's open (raw price for share counting)
        equity = cash + sum(sh * po[s] for s, sh in pos.items())
        # sell what is not wanted (or reduce), then buy
        for s in list(pos):
            if s not in weight:
                amt = pos.pop(s) * po[s]
                cash += amt * (1 - slip) - max(amt * rate, min_comm)
        for s, w in weight.items():
            have = pos.get(s, 0) * po[s]
            need = w * equity - have
            if abs(need) < 0.02 * equity:  # skip tiny adjustments
                continue
            lots = int(abs(need) // (po[s] * 100))
            if lots == 0:
                continue
            amt = lots * 100 * po[s]
            if need > 0:
                cost = amt * slip + max(amt * rate, min_comm)
                if amt + cost > cash:
                    lots = int((cash - min_comm) // (po[s] * 100 * (1 + slip)))
                    if lots <= 0:
                        continue
                    amt = lots * 100 * po[s]
                    cost = amt * slip + max(amt * rate, min_comm)
                cash -= amt + cost
                pos[s] = pos.get(s, 0) + lots * 100
            else:
                pos[s] -= lots * 100
                cash += amt * (1 - slip) - max(amt * rate, min_comm)
        # mark at next rebalance open using back-adjusted returns (captures distributions)
        for s in list(pos):
            r = o[s].iloc[e2] / o[s].iloc[e]
            # convert value growth into equivalent raw shares so accounting stays in raw prices
            pos[s] = pos[s] * r * raw_o[s].iloc[e] / raw_o[s].iloc[e2] if np.isfinite(r) else pos[s]
        eq = cash + sum(sh * raw_o[s].iloc[e2] for s, sh in pos.items())
        hist.append({"date": c.index[e2], "equity": eq, "hold": "+".join(BASKET.get(s, "货币") for s in weight)})
    return pd.DataFrame(hist).set_index("date")


def stats(eq: pd.Series, capital=10_000.0) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    m = eq.resample("ME").last()
    r = m.pct_change().dropna()
    by_year = {}
    prev = capital
    for y in sorted(set(eq.index.year)):
        last = eq[eq.index.year == y].iloc[-1]
        by_year[int(y)] = float(last / prev - 1)
        prev = last
    return {"final": float(eq.iloc[-1]), "cagr": float((eq.iloc[-1] / capital) ** (1 / yrs) - 1),
            "max_dd": float((eq / eq.cummax() - 1).min()), "monthly_mean": float(r.mean()),
            "monthly_median": float(r.median()), "months_up": float((r > 0).mean()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(12)), "by_year": by_year}


def main():
    o, c, raw_o = load()
    out = {}
    variants = [(f"{lb}日动量·持{top}只·{'每月' if fq == 'ME' else '每周'}", lb, top, fq)
                for fq in ("ME", "W") for top in (1, 2) for lb in (10, 20, 40, 60, 120)]
    variants += [(f"多周期混合·持{top}只·{'每月' if fq == 'ME' else '每周'}", 0, top, fq) for fq in ("ME", "W") for top in (1, 2)]
    for name, lb, top, fq in variants:
        bt = simulate(o, c, raw_o, lb, top, fq)
        out[name] = stats(bt["equity"])
        out[name]["latest_hold"] = bt["hold"].iloc[-1]
        s = out[name]
        print(f"{name}: 1万→{s['final']:,.0f} 年化 {s['cagr']:.1%} 回撤 {s['max_dd']:.1%} 月均 {s['monthly_mean']:.2%} "
              f"赚钱月 {s['months_up']:.0%} | " + " ".join(f"{y}:{v:+.0%}" for y, v in s["by_year"].items()), flush=True)
    for code, nm in (("sh510300", "买入持有沪深300ETF"), ("sh510500", "买入持有中证500ETF")):
        px = o[code].loc["2014-01-01":].dropna()
        eq = 10_000 * px / px.iloc[0]
        out[nm] = stats(eq)
        s = out[nm]
        print(f"{nm}: 1万→{s['final']:,.0f} 年化 {s['cagr']:.1%} 回撤 {s['max_dd']:.1%} | " +
              " ".join(f"{y}:{v:+.0%}" for y, v in s["by_year"].items()), flush=True)
    RES.mkdir(exist_ok=True)
    (RES / "etf_rotation_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
