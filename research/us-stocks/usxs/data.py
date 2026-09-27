"""US equity data: point-in-time S&P 500 membership (github.com/fja05680/sp500)
and daily split/dividend-adjusted bars from Yahoo Finance's chart endpoint."""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
UA = {"User-Agent": "Mozilla/5.0"}


def membership(start: str = "2005-01-01") -> pd.DataFrame:
    """Boolean (date × ticker): was the ticker in the S&P 500 on that date."""
    raw = pd.read_csv(DATA / "sp500_hist.csv", parse_dates=["date"])
    raw = raw[raw["date"] >= pd.Timestamp(start) - pd.Timedelta(days=400)]
    sets = {d: set(t.split(",")) for d, t in zip(raw["date"], raw["tickers"])}
    tickers = sorted(set().union(*sets.values()))
    m = pd.DataFrame(False, index=pd.DatetimeIndex(list(sets)), columns=tickers)
    for d, s in sets.items():
        m.loc[d, list(s)] = True
    return m.sort_index()


def _yahoo(ticker: str, start: str = "2004-01-01") -> pd.DataFrame | None:
    path = DATA / "yahoo" / f"{ticker}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df if len(df) else None
    sym = ticker.replace(".", "-")
    p1 = int(pd.Timestamp(start).timestamp())
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2=9999999999"
           f"&interval=1d&events=div%2Csplits&includeAdjustedClose=true")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                js = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                js = None
                break
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    else:
        js = None
    df = pd.DataFrame()
    try:
        res = js["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        df = pd.DataFrame({"open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"],
                           "volume": q["volume"], "adjclose": adj},
                          index=pd.to_datetime(res["timestamp"], unit="s").normalize())
        df = df[~df.index.duplicated(keep="last")].dropna(subset=["adjclose"])
    except Exception:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df if len(df) else None


def load_prices(tickers: list[str], workers: int = 8) -> dict[str, pd.DataFrame]:
    out = {}
    with ThreadPoolExecutor(workers) as ex:
        for t, df in zip(tickers, ex.map(_yahoo, tickers)):
            if df is not None and len(df) > 300:
                out[t] = df
    return out


def sectors() -> dict[str, str]:
    c = pd.read_csv(DATA / "constituents.csv")
    return dict(zip(c["Symbol"], c["GICS Sector"]))
