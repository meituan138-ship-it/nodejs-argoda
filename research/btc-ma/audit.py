"""Audit of the user's 3-MA regime strategy (btc_ma_strategy.py).

A faithful port of its backtest loop with switches for:
  * mark-to-market equity (the original curve only books realised P&L)
  * real funding (Binance BTCUSDT perp, 8h settlements; 0.01% assumed before 2020)
  * variants: no short TP, a real initial stop at 1R, long-only / short-only,
    neighbouring MA / TP parameters, higher risk per trade
First checks that the port reproduces the original backtest exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lightgbm-btc"))
import btc_ma_strategy as S  # noqa: E402
from lgbm_btc.data import load_funding  # noqa: E402

BASE = dict(ma=(7, 25, 99), risk=0.005, cap=3.0, tp_short=1.0, tp_long=None, init_stop=None,
            sides=(1, -1), funding=False, be=1.0)


def indicators(df, ma):
    d = df.copy()
    c = d["close"]
    f, m, s = (c.rolling(n).mean() for n in ma)
    pc = c.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    ok = s.notna() & d["atr"].notna()
    d["want"] = np.where(ok & (f > s) & (m > s), 1, np.where(ok & (f < s) & (m < s), -1, 0))
    d["ok"] = ok
    return d.iloc[S.WARMUP:].reset_index(drop=True)


def run(df, fund=None, **kw):
    p = {**BASE, **kw}
    d = indicators(df, p["ma"])
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr, want, ok, tm = d["atr"].to_numpy(), d["want"].to_numpy(), d["ok"].to_numpy(), d["time"]
    fr = np.zeros(len(d))
    if p["funding"] and fund is not None:
        # sum of 8h settlements falling inside each 4h bar
        idx = pd.DatetimeIndex(tm)
        f = fund.reindex(idx.floor("4h")).fillna(0.0).to_numpy().copy()
        pre = (idx < pd.Timestamp("2020-01-01", tz=idx.tz)) & (idx.hour % 8 == 0)
        f[pre] = 0.0001
        fr = f
    eq = 100_000.0
    pos = None
    blocked = 0
    trades, mtm, real = [], [], []
    FEE = S.FEE_ONE_WAY

    def close(px, reason, i):
        nonlocal eq, pos
        pnl = pos["dir"] * (px - pos["entry"]) * pos["qty"] - px * pos["qty"] * FEE
        eq += pnl
        trades.append({"dir": pos["dir"], "open": pos["t"], "close": tm[i], "pnl": pnl, "reason": reason,
                       "r": pos["dir"] * (px - pos["entry"]) / pos["entry"] / pos["rd"]})
        pos = None

    for i in range(1, len(d)):
        if not ok[i]:
            mtm.append(eq); real.append(eq)
            continue
        if pos is not None and not pos["be"] and pos["dir"] * (c[i - 1] - pos["entry"]) / pos["entry"] >= p["be"] * pos["rd"]:
            e = pos["entry"]
            pos["stop"] = e if pos["stop"] is None else (max(pos["stop"], e) if pos["dir"] == 1 else min(pos["stop"], e))
            pos["be"] = True
        if pos is not None and pos["stop"] is not None and \
                ((pos["dir"] == 1 and o[i] <= pos["stop"]) or (pos["dir"] == -1 and o[i] >= pos["stop"])):
            blocked = pos["dir"]
            close(o[i], "stop_gap", i)
        w = int(want[i - 1])
        if w not in p["sides"]:
            w = 0
        if blocked != 0 and w != blocked:
            blocked = 0
        cur = 0 if pos is None else pos["dir"]
        if w != cur:
            if pos is not None:
                close(o[i], "signal_exit", i)
            if w != 0 and w != blocked:
                rd = 2.0 * atr[i - 1] / o[i]
                if 0 < rd < S.GATE:
                    notional = min(p["risk"] * eq / rd, p["cap"] * eq)
                    qty = np.floor(notional / o[i] / 0.001 + 1e-9) * 0.001
                    if qty >= 0.001:
                        tpr = p["tp_short"] if w == -1 else p["tp_long"]
                        pos = {"dir": w, "entry": o[i], "qty": qty, "rd": rd, "be": False, "t": tm[i],
                               "tp": o[i] * (1 + w * tpr * rd) if tpr else None,
                               "stop": o[i] * (1 - w * p["init_stop"] * rd) if p["init_stop"] else None}
                        eq -= o[i] * qty * FEE
        if pos is not None:
            if fr[i]:
                eq -= pos["dir"] * fr[i] * pos["qty"] * o[i]
            q, e0 = pos["qty"], pos["entry"]
            lp = (q * e0 - eq) / (q * (1 - S.MMR)) if pos["dir"] == 1 else (eq + q * e0) / (q * (1 + S.MMR))
            if eq > 0 and ((pos["dir"] == 1 and l[i] <= lp) or (pos["dir"] == -1 and h[i] >= lp)):
                blocked = pos["dir"]
                close(lp, "liquidation", i)
            else:
                st, tp = pos["stop"], pos["tp"]
                if st is not None and ((pos["dir"] == 1 and l[i] <= st) or (pos["dir"] == -1 and h[i] >= st)):
                    blocked = pos["dir"]
                    close(st, "stop", i)
                elif tp is not None and ((pos["dir"] == 1 and h[i] >= tp) or (pos["dir"] == -1 and l[i] <= tp)):
                    blocked = pos["dir"]
                    close(tp, "take_profit", i)
        real.append(eq)
        mtm.append(eq + (pos["dir"] * (c[i] - pos["entry"]) * pos["qty"] if pos else 0.0))
    if pos is not None:
        close(c[-1], "end_of_data", len(d) - 1)
        real[-1] = mtm[-1] = eq
    t = pd.DataFrame(trades)
    curve = pd.Series(mtm, index=pd.DatetimeIndex(tm[1:]))
    rcurve = pd.Series(real, index=curve.index)
    return t, curve, rcurve


def summary(t, curve, rcurve):
    eq0 = 100_000.0
    yrs = (curve.index[-1] - curve.index[0]).days / 365.25
    g, ls = t.pnl[t.pnl > 0].sum(), -t.pnl[t.pnl <= 0].sum()
    dd_mtm = (curve / curve.cummax() - 1).min()
    dd_real = (pd.concat([pd.Series([eq0]), rcurve.reset_index(drop=True)]).pipe(lambda s: s / s.cummax() - 1)).min()
    under = (curve < curve.cummax()).astype(int)
    grp = (under.diff() != 0).cumsum()
    longest = max((g_.index[-1] - g_.index[0]).days
                  for _, g_ in curve[under == 1].groupby(grp[under == 1])) if under.any() else 0
    s = sorted(t.pnl, reverse=True)
    yearly = curve.resample("YE").last()
    yearly = (yearly / yearly.shift(1).fillna(eq0) - 1)
    return {"final": float(curve.iloc[-1]), "ret": float(curve.iloc[-1] / eq0 - 1),
            "cagr": float((curve.iloc[-1] / eq0) ** (1 / yrs) - 1), "pf": float(g / ls) if ls else float("inf"),
            "n": int(len(t)), "win": float((t.pnl > 0).mean()), "dd_realised": float(dd_real), "dd_mtm": float(dd_mtm),
            "longest_underwater_days": int(longest), "worst_R": float(t.r.min()),
            "top10_share": float(sum(s[:10]) / t.pnl.sum()) if t.pnl.sum() > 0 else float("nan"),
            "long_pnl": float(t.pnl[t.dir == 1].sum()), "short_pnl": float(t.pnl[t.dir == -1].sum()),
            "by_year": {int(k.year): round(float(v), 4) for k, v in yearly.items()}}


def main() -> None:
    df = S.load_csv(str(HERE / "btc_4h.csv"))
    df = df[df.time >= "2019-08-01"].reset_index(drop=True)   # same 7-year window as the claim
    fund = load_funding("BTCUSDT", start="2020-01", end="2026-08-31")
    fund.index = fund.index.floor("4h")
    fund = fund.groupby(level=0).sum()

    orig = S.backtest(df)
    t, cu, rc = run(df)
    op = np.array([x.pnl for x in orig["trades"]])
    assert len(t) == orig["n"] and np.allclose(t.pnl.to_numpy(), op, rtol=1e-9, atol=1e-6), (len(t), orig["n"])
    print(f"port matches original trade-by-trade: {len(t)} trades. Original final {orig['equity_final']:,.0f} "
          f"omits the last open trade ({op[-1]:+,.0f}); with it: {cu.iloc[-1]:,.0f}")

    tests = {
        "原版（按它自己的算法）": {},
        "原版 + 资金费": {"funding": True},
        "去掉空单 1R 止盈": {"funding": True, "tp_short": None},
        "加 1R 初始止损": {"funding": True, "init_stop": 1.0},
        "只做多": {"funding": True, "sides": (1,)},
        "只做空": {"funding": True, "sides": (-1,)},
        "空单 0.5R 止盈": {"funding": True, "tp_short": 0.5},
        "空单 1.5R 止盈": {"funding": True, "tp_short": 1.5},
        "空单 2R 止盈": {"funding": True, "tp_short": 2.0},
        "均线 5/20/90": {"funding": True, "ma": (5, 20, 90)},
        "均线 10/30/120": {"funding": True, "ma": (10, 30, 120)},
        "均线 7/25/60": {"funding": True, "ma": (7, 25, 60)},
        "均线 7/25/150": {"funding": True, "ma": (7, 25, 150)},
        "均线 12/26/99": {"funding": True, "ma": (12, 26, 99)},
        "风险 1% 上限 3x": {"funding": True, "risk": 0.01},
        "风险 2% 上限 3x": {"funding": True, "risk": 0.02},
        "风险 2% 上限 5x": {"funding": True, "risk": 0.02, "cap": 5},
        "风险 3% 上限 5x": {"funding": True, "risk": 0.03, "cap": 5},
        "风险 5% 上限 10x": {"funding": True, "risk": 0.05, "cap": 10},
    }
    out = {}
    for name, kw in tests.items():
        st = summary(*run(df, fund, **kw))
        out[name] = st
        print(f"{name:<22} final {st['final']:>9,.0f} CAGR {st['cagr']:+6.1%} PF {st['pf']:.2f} n {st['n']:>3} "
              f"win {st['win']:.0%} DD(已实现) {st['dd_realised']:.1%} DD(含浮亏) {st['dd_mtm']:.1%} "
              f"水下 {st['longest_underwater_days']}天 最差 {st['worst_R']:.1f}R 多 {st['long_pnl']:+,.0f} 空 {st['short_pnl']:+,.0f}")
        print(" " * 24, {y: f"{v:+.1%}" for y, v in st["by_year"].items()})
    # split halves
    for a, b in (("2019-08-01", "2023-01-01"), ("2023-01-01", "2026-09-01")):
        dd = S.load_csv(str(HERE / "btc_4h.csv"))
        warm_start = dd.index[dd.time >= a][0] - S.WARMUP
        dd = dd.iloc[max(warm_start, 0):][lambda x: x.time < b].reset_index(drop=True)
        st = summary(*run(dd, fund, funding=True))
        out[f"分段 {a[:4]}-{b[:4]}"] = st
        print(f"分段 {a[:7]}→{b[:7]}: CAGR {st['cagr']:+.1%} PF {st['pf']:.2f} n {st['n']} DD(含浮亏) {st['dd_mtm']:.1%}")
    (HERE / "audit_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
