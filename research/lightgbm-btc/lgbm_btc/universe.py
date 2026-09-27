"""Point-in-time coin universe for the cross-sectional study.

The first multi-coin run used 20 coins that are still big today, which
flatters the result (survivorship bias). Here the candidate list also
contains coins that later collapsed, were delisted or renamed (LUNA, FTT,
SRM, WAVES, XMR, MATIC, FTM, RNDR, EOS, ...). Each month the universe is
the top-N candidates by trailing 30-day USDT volume *as known at that
time*, restricted to coins that had a USD-M perpetual (so they could be
shorted) — no knowledge of which coins survive.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .data import load_funding, load_klines

CANDIDATES = [
    # still-large coins
    "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "LTC", "LINK", "DOT", "AVAX", "TRX", "BCH",
    "ETC", "XLM", "ATOM", "UNI", "FIL", "NEAR", "AAVE", "SHIB", "SAND", "MANA", "AXS", "APE",
    "GALA", "ICP", "THETA", "VET", "ALGO", "XTZ", "NEO", "IOTA", "ZEC", "DASH", "EGLD", "HBAR",
    "CHZ", "CRV", "SUSHI", "1INCH", "GRT", "KSM", "RUNE", "APT", "ARB", "OP", "SUI", "PEPE",
    "WIF", "INJ", "TIA", "SEI", "LDO", "STX", "ORDI", "WLD", "FET", "DYDX", "COMP",
    # collapsed / delisted / renamed — the ones a today-picked list silently drops
    "LUNA", "FTT", "SRM", "WAVES", "XMR", "MATIC", "FTM", "RNDR", "EOS", "ANKR", "OMG", "ZIL",
]
PERP_ALIAS = {"SHIB": "1000SHIBUSDT", "PEPE": "1000PEPEUSDT"}
MAX_GAP_BARS = 72  # a longer zero-volume run means delisting (or a relaunch under the same ticker)


def _first_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Cut the series at the first long gap: keeps old LUNA, drops the new chain relisted as LUNAUSDT."""
    dead = (df["quote_volume"] == 0).astype(int)
    run = dead.groupby((dead == 0).cumsum()).cumsum()
    long_gap = np.where(run.to_numpy() >= MAX_GAP_BARS)[0]
    if len(long_gap):
        end = long_gap[0] - MAX_GAP_BARS + 1
        df = df.iloc[:end]
    # trailing zero-volume bars after delisting
    last = np.where(df["quote_volume"].to_numpy() > 0)[0]
    return df.iloc[: last[-1] + 1] if len(last) else df.iloc[:0]


def _load_one(coin: str, start: str, end: str):
    try:
        k = _first_segment(load_klines(f"{coin}USDT", "1h", "spot", start, end))
    except RuntimeError:
        return coin, None, None
    try:
        f = load_funding(PERP_ALIAS.get(coin, f"{coin}USDT"), "2019-09", end)
    except RuntimeError:
        f = None
    return coin, k, f


def load_candidates(start: str = "2019-01", end: str = "2026-08-31", workers: int = 8):
    klines, funding = {}, {}
    with ThreadPoolExecutor(workers) as ex:
        for coin, k, f in ex.map(lambda c: _load_one(c, start, end), CANDIDATES):
            if k is not None and len(k) > 24 * 90:
                klines[coin] = k
                if f is not None and len(f):
                    funding[coin] = f
            print(f"  {coin}: {'-' if k is None else len(k)} bars, funding={'yes' if coin in funding else 'no'}",
                  flush=True)
    return klines, funding


def monthly_universe(klines: dict[str, pd.DataFrame], funding: dict[str, pd.Series], grid: pd.DatetimeIndex,
                     top_n: int = 30, min_history_days: int = 60) -> pd.DataFrame:
    """Boolean (hour × coin): member of this month's top-N by trailing 30-day volume,
    decided at the start of the month with information available then."""
    qv = pd.DataFrame({c: d["quote_volume"] for c, d in klines.items()}).reindex(grid)
    first = {c: d.index[0] for c, d in klines.items()}
    last = {c: d.index[-1] for c, d in klines.items()}
    perp_start = {c: f.index[0] for c, f in funding.items()}
    member = pd.DataFrame(False, index=grid, columns=qv.columns)
    for m in pd.date_range(grid[0].ceil("D") + pd.offsets.MonthBegin(0), grid[-1], freq="MS"):
        hist = qv.loc[m - pd.Timedelta(days=30): m - pd.Timedelta(hours=1)]
        vol30 = hist.sum()
        ok = [c for c in qv.columns
              if first[c] <= m - pd.Timedelta(days=min_history_days) and last[c] >= m
              and c in perp_start and perp_start[c] <= m]
        top = vol30[ok].sort_values(ascending=False).index[:top_n]
        m_end = m + pd.offsets.MonthBegin(1)
        member.loc[(member.index >= m) & (member.index < m_end), list(top)] = True
    return member
