"""Daily money-flow history from Sina (aggregated from tick-by-tick / Level-2
trades): turnover rate, main-force net inflow, extra-large-order net inflow.
Covers delisted stocks too (unlike Eastmoney valuation data, which only has
listed stocks and would leak survivorship into the model)."""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

DIR = Path(__file__).resolve().parent.parent / "data" / "moneyflow"
URL = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
       "MoneyFlow.ssl_qsfx_zjlrqs?page=1&num=5000&sort=opendate&asc=0&daima={code}")
NUM = ["trade", "changeratio", "turnover", "netamount", "ratioamount", "r0_net", "r0_ratio", "r0x_ratio"]


class FetchError(RuntimeError):
    pass


def fetch(code: str) -> pd.DataFrame | None:
    path = DIR / f"{code}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df if len(df) else None
    for attempt in range(8):
        try:
            req = urllib.request.Request(URL.format(code=code), headers={
                "User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn"})
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
            rows = json.loads(body) if body.strip() not in (b"", b"null") else []
            break
        except Exception:
            time.sleep(min(60, 3 * 2 ** attempt))
    else:
        raise FetchError(code)
    df = pd.DataFrame(rows)
    if len(df):
        df["date"] = pd.to_datetime(df["opendate"])
        df = df.set_index("date")[[c for c in NUM if c in df.columns]].apply(pd.to_numeric, errors="coerce").sort_index()
        df = df[~df.index.duplicated()]
    DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df if len(df) else None


def _safe(code):
    try:
        return fetch(code)
    except FetchError:
        return None


def load(codes: list[str], workers: int = 6) -> dict[str, pd.DataFrame]:
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for i, (c, df) in enumerate(zip(codes, ex.map(_safe, codes))):
            if df is not None:
                out[c] = df
            if i % 500 == 0:
                print(f"  moneyflow {i}/{len(codes)}, {len(out)} with data", flush=True)
    return out
