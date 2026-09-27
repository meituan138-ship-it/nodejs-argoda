"""A-share daily bars from Tencent's public quote API for every code in the
A-share code ranges — including delisted stocks, which the API still serves,
so the backtest has no survivorship bias.

Back-adjusted (后复权) OHLC is used for returns. Forward-adjusted prices are
avoided: with 3-decimal rounding they destroy the precision of early,
heavily-adjusted prices (e.g. 浦发 2012 shows as ~0.4 CNY). Raw (不复权)
open/close are kept for traded amount, price level and limit checks.
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,{a},{b},640,{fq}"
START = "2012-01-01"
END = "2026-09-26"


def code_universe() -> list[str]:
    rng = [("sh", 600000, 602000), ("sh", 603000, 604000), ("sh", 605000, 605600), ("sh", 688000, 689000),
           ("sz", 0, 2000), ("sz", 2000, 3000), ("sz", 3000, 3100), ("sz", 300000, 302000)]
    return [f"{ex}{n:06d}" for ex, a, b in rng for n in range(a, b)]


class FetchError(RuntimeError):
    pass


def _get(code: str, a: str, b: str, fq: str) -> list:
    """Empty list = no data; FetchError = rate-limited / network failure (never cached)."""
    url = URL.format(code=code, a=a, b=b, fq=fq)
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
            d = json.loads(body).get("data")  # the WAF answers with an HTML page -> ValueError
            if not isinstance(d, dict) or not d:
                return []
            k = list(d.values())[0]
            return (k.get(f"{fq}day") or k.get("day") or []) if isinstance(k, dict) else []
        except Exception:
            time.sleep(min(60, 3 * 2 ** attempt))
    raise FetchError(code)


def _fetch_rows(code: str, fq: str) -> list:
    rows: list = []
    b = END
    while True:  # the API returns the last 640 rows before `b`; page backwards
        chunk = _get(code, START, b, fq)
        if not chunk:
            break
        rows = chunk + rows
        first = chunk[0][0]
        if len(chunk) < 640 or first <= START:
            break
        b = (pd.Timestamp(first) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return rows


def _frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame([r[:6] for r in rows], columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    return df.drop_duplicates("date").set_index("date").astype(float).sort_index()


def fetch_stock(code: str, index: bool = False) -> pd.DataFrame | None:
    path = DATA / ("index" if index else "daily") / f"{code}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df if len(df) else None
    df = pd.DataFrame()
    if index:
        rows = _fetch_rows(code, "")
        if rows:
            df = _frame(rows)
    else:
        rows = _fetch_rows(code, "hfq")
        if rows:
            df = _frame(rows)
            raw = _frame(_fetch_rows(code, ""))
            df["raw_close"] = raw["close"].reindex(df.index)
            df["raw_open"] = raw["open"].reindex(df.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df if len(df) else None


def _safe_fetch(code: str) -> pd.DataFrame | None:
    try:
        return fetch_stock(code)
    except FetchError:
        return None


def load_all(workers: int = 24) -> dict[str, pd.DataFrame]:
    codes = code_universe()
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, (c, df) in enumerate(zip(codes, ex.map(_safe_fetch, codes))):
            if df is not None and len(df) > 60:
                out[c] = df
            if i % 1000 == 0:
                print(f"  {i}/{len(codes)} codes checked, {len(out)} with data", flush=True)
    return out


def load_index(code: str) -> pd.DataFrame | None:
    return fetch_stock(code, index=True)
