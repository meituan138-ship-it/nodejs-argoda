"""Funding-rate carry (cash-and-carry): long spot + short USD-M perp in the
same coin, so price moves cancel and the short collects funding.

    python run_carry.py

Rules (fixed before looking at the results):
- decision once a day at 00:00 UTC from information up to then;
- candidates: coins in that month's point-in-time liquidity top 30
  (lgbm_btc/universe.py), or only BTC/ETH for the conservative version;
- signal: average funding over the last 7 days (21 settlements);
- enter a coin when it ranks in the top K and its 7-day average > ENTER
  per 8h; exit when it leaves the top 2K or the average drops below EXIT
  (hysteresis keeps trading rare);
- PnL per unit of notional: funding received on the short, minus costs on
  every entry / exit of BOTH legs (spot + perp). Basis moves between perp
  and spot are not modelled (usually a few bp, can be larger in squeezes).
- capital: 1 unit spot + 1/3 unit perp margin (3x on the short) per unit of
  notional, so return on capital = PnL / 1.33.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from lgbm_btc.universe import load_candidates, monthly_universe
from run_xs import RESULTS

ENTER, EXIT = 1e-4, 0.0  # per 8h: 0.01% ≈ 11%/yr
CAPITAL_PER_NOTIONAL = 1 + 1 / 3
COST_SCENARIOS = {  # one-way cost per leg in bp, (spot, perp)
    "MEXC 挂单 (0/0 + 2bp 价差)": (2.0, 2.0),
    "MEXC 吃单 (5/2 + 3bp 滑点)": (8.0, 5.0),
    "Binance 吃单 (10/5 + 3bp)": (13.0, 8.0),
}


def daily_funding(funding: dict[str, pd.Series], days: pd.DatetimeIndex) -> pd.DataFrame:
    """Funding received by a short during (d, d+1 day] for each coin."""
    out = {}
    for c, f in funding.items():
        cum = f.cumsum()
        a = cum.reindex(days, method="ffill").fillna(0.0).to_numpy()
        b = cum.reindex(days + pd.Timedelta(days=1), method="ffill").fillna(0.0).to_numpy()
        out[c] = b - a
    return pd.DataFrame(out, index=days)


def trailing_avg(funding: dict[str, pd.Series], days: pd.DatetimeIndex, n: int = 21) -> pd.DataFrame:
    out = {}
    for c, f in funding.items():
        m = f.rolling(n, min_periods=n).mean()
        out[c] = m.reindex(days, method="ffill")  # settlements up to and including 00:00
    return pd.DataFrame(out, index=days)


def backtest(eligible: pd.DataFrame, fwd: pd.DataFrame, sig: pd.DataFrame, k: int,
             cost_bp: tuple[float, float]) -> pd.DataFrame:
    held: set[str] = set()
    rows = []
    per_leg = (cost_bp[0] + cost_bp[1]) * 1e-4  # both legs, one way
    for d in fwd.index:
        s = sig.loc[d].where(eligible.loc[d]).dropna().sort_values(ascending=False)
        top_k, top_2k = set(s.index[:k]), set(s.index[:2 * k])
        keep = {c for c in held if c in top_2k and s.get(c, -1) > EXIT}
        new = {c for c in top_k if s[c] > ENTER} - keep
        room = max(0, k - len(keep))
        new = set(sorted(new, key=lambda c: -s[c])[:room])
        now = keep | new
        w_old = {c: 1 / len(held) for c in held} if held else {}
        w_new = {c: 1 / len(now) for c in now} if now else {}
        turnover = sum(abs(w_new.get(c, 0) - w_old.get(c, 0)) for c in set(w_old) | set(w_new))
        fund = sum(w * fwd.loc[d].get(c, 0.0) for c, w in w_new.items())
        rows.append({"date": d, "funding": fund, "cost": turnover * per_leg,
                     "net": fund - turnover * per_leg, "n": len(now), "invested": float(bool(now))})
        held = now
    return pd.DataFrame(rows).set_index("date")


def summarize(bt: pd.DataFrame) -> dict:
    r = bt["net"] / CAPITAL_PER_NOTIONAL  # return on capital
    eq = (1 + r).cumprod()
    yrs = len(r) / 365
    by_year = {int(y): float((1 + r[r.index.year == y]).prod() - 1) for y in sorted(set(r.index.year))}
    m26 = r[r.index >= "2026-01-01"]
    months_2026 = {k: float(v) for k, v in ((1 + m26).groupby(m26.index.strftime("%Y-%m")).prod() - 1).items()}
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "max_dd": float((eq / eq.cummax() - 1).min()),
            "sharpe": float(r.mean() / r.std() * np.sqrt(365)), "time_invested": float(bt["invested"].mean()),
            "cost_share_of_funding": float(bt["cost"].sum() / max(bt["funding"].sum(), 1e-12)),
            "by_year": by_year, "months_2026": months_2026}


def main() -> None:
    klines, funding = load_candidates()
    start = min(d.index[0] for d in klines.values())
    grid = pd.date_range(start, pd.Timestamp("2026-08-31 23:00").tz_localize(start.tz), freq="h")
    member = monthly_universe(klines, funding, grid)
    days = pd.date_range(pd.Timestamp("2021-01-01").tz_localize(start.tz),
                         pd.Timestamp("2026-08-30").tz_localize(start.tz), freq="D")
    coins = [c for c in member.columns if c in funding]
    elig_all = member.reindex(days)[coins].fillna(False).astype(bool)
    elig_major = elig_all.copy()
    elig_major.loc[:, [c for c in coins if c not in ("BTC", "ETH")]] = False
    fwd = daily_funding({c: funding[c] for c in coins}, days)
    sig = trailing_avg({c: funding[c] for c in coins}, days)

    out = {}
    for name, elig, k in (("BTC+ETH", elig_major, 2), ("Top30 选前5", elig_all, 5), ("Top30 选前10", elig_all, 10)):
        for cname, cost in COST_SCENARIOS.items():
            s = summarize(backtest(elig, fwd, sig, k, cost))
            out[f"{name} | {cname}"] = s
            print(f"{name:12s} {cname:28s} CAGR {s['cagr']:+.1%} Sharpe {s['sharpe']:.2f} MDD {s['max_dd']:.1%} "
                  f"invested {s['time_invested']:.0%} cost/funding {s['cost_share_of_funding']:.0%} "
                  f"2026 {s['by_year'].get(2026, float('nan')):+.1%}", flush=True)
    (RESULTS / "carry_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
