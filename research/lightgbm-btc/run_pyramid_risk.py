"""Risk-based version of the user's pyramid (turtle-style), with an honest protocol.

Changes vs run_pyramid.py (the user's fixed-% rules):
  * size by RISK, not by margin: each unit loses `risk` of equity if its stop is hit
        units = risk * equity / (m * ATR)          (ATR = 20-day average true range)
  * stop distance adapts to volatility: m * ATR below the entry / last add
  * pyramid: add one equal unit every +k*ATR, max 4 units (like the user's 4 tranches)
  * exits: the stop ratchets up to (last add - m*ATR) and to the lowest low of the last
    exit_n days (Donchian exit) — whichever is higher; never moves down
  * optional trend filter: only long above the 200-day average
  * leverage cap: total notional <= 3x equity (exchange leverage 10x is only margin)
Protocol: choose the configuration on BTC+ETH 2018-2023 only; report it untouched on
2024-01 -> 2026-08 and on SOL (never used for selection).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lgbm_btc.data import load_funding, load_klines  # noqa: E402
from run_pyramid import COST, MAINT, stats  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
MAX_UNITS = 4
LEV_CAP = 3.0


def daily_frame(px: pd.DataFrame, n: int, exit_n: int) -> pd.DataFrame:
    d = px.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    out = pd.DataFrame({
        "atr": tr.rolling(20).mean(),
        "entry": d.close > d.high.rolling(n).max().shift(1),
        "exit_low": d.low.rolling(exit_n).min(),
        "above200": d.close > d.close.rolling(200).mean(),
    })
    # everything above is known at the day's close -> usable from the next day 00:00
    out.index = out.index + pd.Timedelta(days=1)
    return out


def simulate(px, fund, cfg, start, end=None, capital=1000.0):
    n, m, risk, k, filt = cfg["n"], cfg["m"], cfg["risk"], cfg["k"], cfg["filter"]
    D = daily_frame(px, n, max(n // 2, 10))
    px = px[(px.index >= start) & ((px.index < end) if end else True)]
    idx = px.index
    Dh = D.reindex(idx, method="ffill")
    new_day = pd.Series(idx.hour == 0, index=idx).to_numpy()
    atr, sig, xlow, up = (Dh[c].to_numpy() for c in ("atr", "entry", "exit_low", "above200"))
    o, hi, lo, cl = (px[c].to_numpy() for c in ("open", "high", "low", "close"))
    f = fund.reindex(idx).fillna(0.0).to_numpy().copy()
    f[(idx < pd.Timestamp("2020-01-01", tz=idx.tz)) & (idx.hour % 8 == 0)] = 0.0001
    cash, units, avg, nu = capital, 0.0, 0.0, 0
    unit_size = last_add = stop = a0 = 0.0
    eq = np.zeros(len(idx))
    trades, t_open, eq_open = [], None, 0.0
    end_i = len(idx)
    for i in range(len(idx)):
        if units == 0 and new_day[i] and sig[i] and atr[i] > 0 and (up[i] or not filt) and cash > 10:
            p, a0 = o[i], atr[i]
            unit_size = min(risk * cash / (m * a0), LEV_CAP * cash / MAX_UNITS / p)
            units, avg, nu, last_add = unit_size, p, 1, p
            stop = p - m * a0
            cash -= units * p * COST
            t_open, eq_open = idx[i], cash + units * p * COST
        if units > 0:
            if f[i]:
                cash -= f[i] * units * o[i]
            if new_day[i] and not np.isnan(xlow[i]):
                stop = max(stop, xlow[i])                       # Donchian exit, ratchets up only
            if lo[i] <= stop:
                p = min(o[i], stop)                             # gap through the stop fills at the open
                if cash + units * (p - avg) <= MAINT * units * p:
                    trades.append((t_open, idx[i], nu, -eq_open, "liquidated"))
                    cash, units = 0.0, 0.0
                    end_i = i + 1
                    break
                cash += units * (p - avg) - units * p * COST
                trades.append((t_open, idx[i], nu, cash - eq_open, "stop"))
                units, nu = 0.0, 0
            else:
                while nu < MAX_UNITS and hi[i] >= last_add + k * a0:
                    lvl = last_add + k * a0
                    add = unit_size
                    avg = (avg * units + lvl * add) / (units + add)
                    units += add
                    cash -= add * lvl * COST
                    last_add, nu = lvl, nu + 1
                    stop = max(stop, lvl - m * a0)
        eq[i] = cash + (units * (cl[i] - avg) if units else 0.0)
    eq = pd.Series(eq[:end_i], index=idx[:end_i])
    if units:
        trades.append((t_open, idx[end_i - 1], nu, eq.iloc[-1] - eq_open, "still open"))
    return eq, pd.DataFrame(trades, columns=["open", "close", "tranches", "pnl", "exit"])


def main() -> None:
    data = {}
    for sym in ("BTC", "ETH", "SOL"):
        px = load_klines(f"{sym}USDT", "1h", "spot", start="2017-08" if sym != "SOL" else "2020-08", end="2026-08-31")
        fu = load_funding(f"{sym}USDT", start="2020-01" if sym != "SOL" else "2020-09", end="2026-08-31")
        fu.index = fu.index.floor("h")
        data[sym] = (px, fu)
    grid = [dict(n=n, m=m, risk=r, k=k, filter=fl) for n, m, r, k, fl in
            itertools.product((20, 55), (2.0, 3.0), (0.01, 0.02), (0.5, 1.0), (False, True))]
    rows = []
    for cfg in grid:
        name = f"N{cfg['n']}_m{cfg['m']:g}_r{cfg['risk']:.0%}_k{cfg['k']:g}_{'ma200' if cfg['filter'] else 'nofilt'}"
        row = {"config": name, **cfg}
        for sym, (px, fu) in data.items():
            for per, (s, e) in {"train": ("2018-01-01", "2024-01-01"), "test": ("2024-01-01", None)}.items():
                if sym == "SOL":
                    s = "2021-01-01" if per == "train" else s
                eq, tr = simulate(px, fu, cfg, s, e)
                st = stats(eq, tr)
                row[f"{sym}_{per}_final"] = st["final"]
                row[f"{sym}_{per}_cagr"] = st["cagr"]
                row[f"{sym}_{per}_dd"] = st["max_dd"]
                row[f"{sym}_{per}_sharpe"] = st["sharpe"]
                row[f"{sym}_{per}_trades"] = st["trades"]
                row[f"{sym}_{per}_by_year"] = json.dumps({y: round(v, 3) for y, v in st["by_year"].items()})
        row["train_score"] = (row["BTC_train_sharpe"] + row["ETH_train_sharpe"]) / 2
        rows.append(row)
        print(f"{name:<32} trainSR={row['train_score']:.2f}  BTC test {row['BTC_test_final']:.0f}  "
              f"ETH test {row['ETH_test_final']:.0f}  SOL test {row['SOL_test_final']:.0f}", flush=True)
    df = pd.DataFrame(rows).sort_values("train_score", ascending=False)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "pyramid_risk_grid.csv", index=False)
    best = df.iloc[0]
    print("\nSELECTED on 2018-2023 BTC+ETH:", best["config"])
    for sym in ("BTC", "ETH", "SOL"):
        for per in ("train", "test"):
            print(f"  {sym} {per}: 1000 -> {best[f'{sym}_{per}_final']:.0f}  CAGR {best[f'{sym}_{per}_cagr']:+.1%}  "
                  f"maxDD {best[f'{sym}_{per}_dd']:.0%}  Sharpe {best[f'{sym}_{per}_sharpe']:.2f}  "
                  f"trades {best[f'{sym}_{per}_trades']}  {best[f'{sym}_{per}_by_year']}")
    print("\nall 32 configs, test period: share with profit —",
          {s: float((df[f'{s}_test_final'] > 1000).mean()) for s in ("BTC", "ETH", "SOL")},
          " median final —", {s: float(df[f'{s}_test_final'].median()) for s in ("BTC", "ETH", "SOL")})


if __name__ == "__main__":
    main()
