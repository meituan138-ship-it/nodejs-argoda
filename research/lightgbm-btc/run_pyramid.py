"""Pyramiding trend strategy requested by the user, on 1h Binance data.

Rules (long; the short version mirrors them):
  entry   : on a trend signal, margin 20% of equity at 10x  -> notional 2x equity
  add #1  : price +step from entry          -> another 20% margin (2x)
  add #2  : price +step from add #1         -> another 20% margin (2x)
  add #3  : price +step from add #2         -> 40% margin (4x)        total 10x initial equity
  stop    : before any add, entry*(1-stop). After adds:
            mode "trail"   -> highest price since entry * (1-stop)
            mode "lastadd" -> last add price * (1-stop)
  cross margin; liquidation if equity < 0.5% maintenance of notional.
Costs: 2 bp fee + 3 bp slippage per side, real funding every 8h (0.01% assumed before 2020).
Intrabar: if a bar touches both the stop and an add level, the stop is assumed first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lgbm_btc.data import load_funding, load_klines  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
COST = 0.0005
MAINT = 0.005
SIZES = (0.2, 0.2, 0.2, 0.4)
LEV = 10


def signals(daily: pd.DataFrame, kind: str, side: int, seed: int = 0) -> pd.Series:
    """Boolean per day: a fresh signal at that day's close (trade from the next bar)."""
    c, h, l = daily["close"], daily["high"], daily["low"]
    if kind.startswith("don"):
        n = int(kind[3:])
        s = c > h.rolling(n).max().shift(1) if side > 0 else c < l.rolling(n).min().shift(1)
    elif kind == "ma":
        m50, m200 = c.rolling(50).mean(), c.rolling(200).mean()
        s = (m50 > m200) & (c > m50) if side > 0 else (m50 < m200) & (c < m50)
    elif kind == "random":
        rng = np.random.default_rng(seed)
        s = pd.Series(rng.random(len(c)) < 0.05, index=c.index)
    elif kind == "always":
        s = pd.Series(True, index=c.index)
    else:
        raise ValueError(kind)
    return s.fillna(False)


def simulate(px: pd.DataFrame, fund: pd.Series, sig: pd.Series, side: int, step: float, stop: float,
             mode: str, start: str, capital: float = 1000.0):
    px = px[px.index >= start]
    o, hi, lo, cl = (px[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = px.index
    # a signal at day d's close is actionable from the bar opening at d+1 00:00 UTC
    sig_bar = pd.Series(False, index=idx)
    sig_bar.loc[sig_bar.index.intersection(sig[sig].index + pd.Timedelta(days=1))] = True
    sig_arr = sig_bar.to_numpy()
    f = fund.reindex(idx).fillna(0.0).to_numpy().copy()
    pre = (idx < pd.Timestamp("2020-01-01", tz=idx.tz)) & (idx.hour % 8 == 0)
    f[pre] = 0.0001
    cash, units, avg, k = capital, 0.0, 0.0, 0
    base = last_add = peak = 0.0
    eq = np.zeros(len(idx))
    trades, liq = [], 0
    t_open, eq_open = None, 0.0
    end = len(idx)
    for i in range(len(idx)):
        if units == 0 and sig_arr[i] and cash > 10:
            p = o[i]
            base = cash
            units = SIZES[0] * LEV * base / p
            avg, k, last_add, peak = p, 1, p, p
            cash -= units * p * COST
            t_open, eq_open = idx[i], base
        if units > 0:
            if f[i]:
                cash -= side * f[i] * units * o[i]          # long pays positive funding
            fav = hi[i] if side > 0 else lo[i]
            adv = lo[i] if side > 0 else hi[i]
            ref = peak if (mode == "trail" and k > 1) else last_add
            stop_px = ref * (1 - side * stop)
            hit = (adv <= stop_px) if side > 0 else (adv >= stop_px)
            # liquidation: equity at the worst point of the bar (or at the stop, if it comes first)
            test_px = stop_px if hit else adv
            if hit and ((o[i] < stop_px) if side > 0 else (o[i] > stop_px)):
                test_px = o[i]                                # gapped through the stop
            eq_test = cash + side * units * (test_px - avg)
            if eq_test <= MAINT * units * test_px:
                trades.append((t_open, idx[i], k, -eq_open, "liquidated"))
                cash, units, k, liq = 0.0, 0.0, 0, liq + 1
                end = i + 1
                break
            if hit:
                cash += side * units * (test_px - avg) - units * test_px * COST
                trades.append((t_open, idx[i], k, cash - eq_open, "stop"))
                units, k = 0.0, 0
            else:
                while k < len(SIZES):
                    lvl = last_add * (1 + side * step)
                    if not ((fav >= lvl) if side > 0 else (fav <= lvl)):
                        break
                    add = SIZES[k] * LEV * base / lvl
                    avg = (avg * units + lvl * add) / (units + add)
                    units += add
                    cash -= add * lvl * COST
                    last_add = lvl
                    k += 1
                peak = max(peak, hi[i]) if side > 0 else min(peak, lo[i])
        eq[i] = cash + (side * units * (cl[i] - avg) if units else 0.0)
    eq = pd.Series(eq[:end], index=idx[:end])
    if units:
        trades.append((t_open, idx[end - 1], k, eq.iloc[-1] - eq_open, "still open"))
    return eq, pd.DataFrame(trades, columns=["open", "close", "tranches", "pnl", "exit"]), liq


def stats(eq: pd.Series, tr: pd.DataFrame, capital: float = 1000.0) -> dict:
    d = eq.resample("1D").last().dropna()
    yrs = max((d.index[-1] - d.index[0]).days / 365.25, 1e-9)
    final = float(d.iloc[-1])
    r = d.pct_change().dropna()
    dd = float((d / d.cummax() - 1).min())
    by_year = {int(y): float(g.iloc[-1] / g.iloc[0] - 1) if g.iloc[0] > 0 else -1.0
               for y, g in d.groupby(d.index.year)}
    wins = tr[tr["pnl"] > 0]
    loss = tr[tr["pnl"] <= 0]
    return {
        "final": final, "cagr": (final / capital) ** (1 / yrs) - 1 if final > 0 else -1.0,
        "sharpe": float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0, "max_dd": dd,
        "trades": int(len(tr)), "win_rate": float(len(wins) / len(tr)) if len(tr) else 0.0,
        "avg_win": float(wins["pnl"].mean()) if len(wins) else 0.0,
        "avg_loss": float(loss["pnl"].mean()) if len(loss) else 0.0,
        "full_pyramids": int((tr["tranches"] == 4).sum()), "by_year": by_year,
    }


def main() -> None:
    syms = {"BTC": "2018-01-01", "ETH": "2018-01-01", "SOL": "2021-01-01"}
    grid = [(0.10, 0.10), (0.05, 0.05), (0.05, 0.10), (0.10, 0.05), (0.03, 0.03), (0.01, 0.01), (0.20, 0.10)]
    rows, detail = [], {}
    for sym, start in syms.items():
        px = load_klines(f"{sym}USDT", "1h", "spot", start="2017-08" if sym != "SOL" else "2020-08", end="2026-08-31")
        fund = load_funding(f"{sym}USDT", start="2020-01" if sym != "SOL" else "2020-09", end="2026-08-31")
        fund.index = fund.index.floor("h")
        daily = px.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        for side in (1, -1):
            for kind in ("don20", "don55", "ma", "always", "random"):
                seeds = range(30) if kind == "random" else [0]
                for step, stop in grid:
                    for mode in ("trail", "lastadd"):
                        res = []
                        for sd in seeds:
                            sig = signals(daily, kind, side, sd)
                            eq, tr, liq = simulate(px, fund, sig, side, step, stop, mode, start)
                            st = stats(eq, tr)
                            st["liquidations"] = liq
                            res.append(st)
                        st = res[0] if len(res) == 1 else {
                            "final": float(np.median([r["final"] for r in res])),
                            "cagr": float(np.median([r["cagr"] for r in res])),
                            "sharpe": float(np.median([r["sharpe"] for r in res])),
                            "max_dd": float(np.median([r["max_dd"] for r in res])),
                            "trades": int(np.median([r["trades"] for r in res])),
                            "win_rate": float(np.median([r["win_rate"] for r in res])),
                            "avg_win": float(np.median([r["avg_win"] for r in res])),
                            "avg_loss": float(np.median([r["avg_loss"] for r in res])),
                            "full_pyramids": int(np.median([r["full_pyramids"] for r in res])),
                            "liquidations": int(np.sum([r["liquidations"] for r in res])),
                            "share_profitable": float(np.mean([r["final"] > 1000 for r in res])),
                            "by_year": {}}
                        key = f"{sym}|{'long' if side > 0 else 'short'}|{kind}|step{step:.0%}|stop{stop:.0%}|{mode}"
                        rows.append({"key": key, "sym": sym, "side": "long" if side > 0 else "short", "signal": kind,
                                     "step": step, "stop": stop, "mode": mode,
                                     **{k: v for k, v in st.items() if k != "by_year"}})
                        detail[key] = st
                        print(f"{key:<55} final={st['final']:>10.0f} cagr={st['cagr']:+.1%} dd={st['max_dd']:.0%} "
                              f"trades={st['trades']} win={st['win_rate']:.0%} full={st['full_pyramids']} liq={st['liquidations']}",
                              flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "pyramid_grid.csv", index=False)
    (OUT / "pyramid_detail.json").write_text(json.dumps(detail, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
