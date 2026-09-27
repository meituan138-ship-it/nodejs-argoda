"""Loader for Binance's free public archive (https://data.binance.vision).

No API key is needed, so anyone can download exactly the same files and
reproduce every number in the report.

Quirks handled here:
- spot kline files have no header, futures files do;
- spot timestamps switched from milliseconds to microseconds in 2025;
- the exchange had a few outages, so bars are re-indexed onto a complete
  grid (``shift(k)`` must always mean "k bars ago").
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision/data"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
KLINE_NUMERIC = [
    "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
]
_MARKET_PATH = {"spot": "spot", "um": "futures/um"}


def _fetch(url: str, dest: Path, retries: int = 4) -> bool:
    """Download ``url`` to ``dest``; return False when the file does not exist."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = resp.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {err}")


def _read_zip_csv(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        raw = zf.read(zf.namelist()[0])
    first_char = raw[:1].decode(errors="ignore")
    has_header = not (first_char.isdigit() or first_char == "-")
    df = pd.read_csv(io.BytesIO(raw), header=0 if has_header else None)
    if columns is not None:
        df.columns = columns[: df.shape[1]]
    return df


def _to_utc(ts: pd.Series) -> pd.DatetimeIndex:
    """Epoch timestamps -> UTC datetimes (spot files use microseconds since 2025)."""
    v = ts.astype("int64").to_numpy()
    v = np.where(v > 10**14, v // 1000, v)
    return pd.DatetimeIndex(pd.to_datetime(v, unit="ms", utc=True))


def _monthly_or_daily(
    kind_path: str, fname_prefix: str, start: str, end: pd.Timestamp,
    cache_dir: Path, columns: list[str] | None, daily_fallback: bool = True,
) -> pd.DataFrame:
    """Fetch monthly archives from ``start`` to ``end``; fall back to daily files
    for months whose monthly archive is not published yet."""
    frames = []
    for month in pd.period_range(start, end.strftime("%Y-%m"), freq="M"):
        ym = month.strftime("%Y-%m")
        url = f"{BASE_URL}/{kind_path.format(freq='monthly')}/{fname_prefix}-{ym}.zip"
        dest = cache_dir / kind_path.format(freq="monthly") / f"{fname_prefix}-{ym}.zip"
        month_complete = month.end_time.normalize().tz_localize("UTC") <= end.normalize()
        if month_complete and _fetch(url, dest):
            frames.append(_read_zip_csv(dest, columns))
            continue
        if not daily_fallback:
            continue
        last_day = min(month.end_time.tz_localize("UTC"), end)
        for day in pd.date_range(month.start_time.tz_localize("UTC"), last_day, freq="D"):
            d = day.strftime("%Y-%m-%d")
            url = f"{BASE_URL}/{kind_path.format(freq='daily')}/{fname_prefix}-{d}.zip"
            dest = cache_dir / kind_path.format(freq="daily") / f"{fname_prefix}-{d}.zip"
            if _fetch(url, dest):
                frames.append(_read_zip_csv(dest, columns))
    if not frames:
        raise RuntimeError(f"no data found for {fname_prefix} between {start} and {end:%Y-%m-%d}")
    return pd.concat(frames, ignore_index=True)


def _resolve_end(end: str | None) -> pd.Timestamp:
    if end is None:
        return pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(seconds=1)
    return pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def load_klines(
    symbol: str = "BTCUSDT", interval: str = "1h", market: str = "spot",
    start: str = "2017-08", end: str | None = None, cache_dir: Path = DEFAULT_CACHE,
) -> pd.DataFrame:
    """OHLCV + taker-flow bars indexed by bar *open* time (UTC).

    A row's values are only known at open_time + interval, which is why every
    feature is computed from rows <= t and every label from rows > t.
    """
    end_ts = _resolve_end(end)
    kind = f"{_MARKET_PATH[market]}/{{freq}}/klines/{symbol}/{interval}"
    raw = _monthly_or_daily(kind, f"{symbol}-{interval}", start, end_ts, cache_dir, KLINE_COLUMNS)
    raw.index = _to_utc(raw["open_time"])
    df = raw[KLINE_NUMERIC].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df.index <= end_ts]

    freq = pd.Timedelta(interval.replace("m", "min") if interval.endswith("m") else interval)
    grid = pd.date_range(df.index[0], df.index[-1], freq=freq)
    n_missing = len(grid.difference(df.index))
    df = df.reindex(grid)
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    vol_cols = ["volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    df[vol_cols] = df[vol_cols].fillna(0.0)
    df.index.name = "open_time"
    df.attrs["n_missing_bars"] = n_missing
    return df


def load_funding(
    symbol: str = "BTCUSDT", start: str = "2020-01", end: str | None = None,
    cache_dir: Path = DEFAULT_CACHE,
) -> pd.Series:
    """Settled perpetual funding rates (USD-M), indexed by settlement time."""
    end_ts = _resolve_end(end)
    kind = f"futures/um/{{freq}}/fundingRate/{symbol}"
    raw = _monthly_or_daily(kind, f"{symbol}-fundingRate", start, end_ts, cache_dir, None,
                            daily_fallback=False)
    s = pd.Series(raw["last_funding_rate"].astype(float).to_numpy(), index=_to_utc(raw["calc_time"]))
    # calc_time is sometimes 1 ms past the hour; snap to the settlement minute
    s.index = s.index.floor("min")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "funding_rate"
    return s[s.index <= end_ts]


def load_premium_index(
    symbol: str = "BTCUSDT", interval: str = "1h", start: str = "2020-01",
    end: str | None = None, cache_dir: Path = DEFAULT_CACHE,
) -> pd.Series:
    """Perp premium index (≈ basis between perpetual and spot index) close per bar."""
    end_ts = _resolve_end(end)
    kind = f"futures/um/{{freq}}/premiumIndexKlines/{symbol}/{interval}"
    raw = _monthly_or_daily(kind, f"{symbol}-{interval}", start, end_ts, cache_dir, KLINE_COLUMNS)
    s = pd.Series(raw["close"].astype(float).to_numpy(), index=_to_utc(raw["open_time"]))
    s = s[~s.index.duplicated(keep="last")].sort_index()
    s.name = "premium"
    return s[s.index <= end_ts]


def estimate_funding_from_premium(premium: pd.Series, after: pd.Timestamp) -> pd.Series:
    """Approximate 8h settlements after ``after`` from hourly premium-index closes.

    Binance: F = P + clamp(I - P, -0.05%, +0.05%) with I = 0.01% per 8h and P the
    average premium over the interval. The archive only publishes funding
    monthly, so live use needs this (or the exchange API) for the current month.
    """
    p = premium[premium.index > after - pd.Timedelta(hours=8)]
    settle = pd.date_range(after.ceil("8h"), p.index[-1] + pd.Timedelta(hours=1), freq="8h")
    settle = settle[settle > after]
    vals = []
    for t in settle:
        avg = p[(p.index >= t - pd.Timedelta(hours=8)) & (p.index < t)].mean()
        vals.append(avg + np.clip(1e-4 - avg, -5e-4, 5e-4))
    return pd.Series(vals, index=settle, name="funding_rate").dropna()


def load_market_data(
    start: str = "2017-08", end: str | None = None, interval: str = "1h",
    cache_dir: Path = DEFAULT_CACHE, with_derivatives: bool = True,
    extend_funding: bool = False,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Everything the feature builder needs, loaded from the public archive."""
    out: dict[str, pd.DataFrame | pd.Series] = {
        "btc": load_klines("BTCUSDT", interval, "spot", start, end, cache_dir),
        "eth": load_klines("ETHUSDT", interval, "spot", start, end, cache_dir),
    }
    if with_derivatives:
        funding = load_funding("BTCUSDT", "2020-01", end, cache_dir)
        premium = load_premium_index("BTCUSDT", interval, "2020-01", end, cache_dir)
        if extend_funding and premium.index[-1] > funding.index[-1] + pd.Timedelta(hours=8):
            funding = pd.concat([funding, estimate_funding_from_premium(premium, funding.index[-1])])
        out["funding"] = funding
        out["premium"] = premium
    return out
