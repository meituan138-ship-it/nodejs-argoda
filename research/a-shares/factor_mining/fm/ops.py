"""Causal building blocks for factor formulas (Alpha101 style).

Every input is a wide DataFrame: rows = trading days (ascending), columns =
stock codes. Every operator here only looks at rows <= t, so a factor made
only from these operators cannot peek into the future.

Do NOT use .shift(-k), .iloc[i + k], rolling(..., center=True), bfill(),
or anything that reads later rows — the leak test will reject the factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def delay(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    """Value n days ago."""
    if n < 0:
        raise ValueError("delay() only looks back; n must be >= 0")
    return x.shift(n)


def delta(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return x - delay(x, n)


def pct(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return x / (delay(x, n) + EPS) - 1


def log_ret(x: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return np.log(x / delay(x, n))


def ts_mean(x, n):
    return x.rolling(n, min_periods=max(1, n // 2)).mean()


def ts_sum(x, n):
    return x.rolling(n, min_periods=max(1, n // 2)).sum()


def ts_std(x, n):
    return x.rolling(n, min_periods=max(2, n // 2)).std()


def ts_max(x, n):
    return x.rolling(n, min_periods=max(1, n // 2)).max()


def ts_min(x, n):
    return x.rolling(n, min_periods=max(1, n // 2)).min()


def ts_skew(x, n):
    return x.rolling(n, min_periods=max(3, n // 2)).skew()


def ts_zscore(x, n):
    return (x - ts_mean(x, n)) / (ts_std(x, n) + EPS)


def ts_rank(x, n):
    """Percentile of today's value within the last n days (slow for big n)."""
    return x.rolling(n, min_periods=max(2, n // 2)).rank(pct=True)


def ts_corr(x, y, n):
    return x.rolling(n, min_periods=max(3, n // 2)).corr(y)


def ts_argmax(x, n):
    """Days since the n-day maximum (0 = today)."""
    return x.rolling(n, min_periods=max(1, n // 2)).apply(lambda a: len(a) - 1 - np.nanargmax(a), raw=True)


def decay_linear(x, n):
    """Linearly weighted moving average, most recent day heaviest."""
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return x.rolling(n, min_periods=n).apply(lambda a: np.dot(a, w), raw=True)


def ema(x, span):
    return x.ewm(span=span, adjust=False, min_periods=span).mean()


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank on each day (uses only same-day values)."""
    return x.rank(axis=1, pct=True)


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + EPS, axis=0)


def cs_demean(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0)


def where(cond, a, b):
    return a.where(cond, b)


def safe_div(a, b):
    return a / (b + EPS) if not isinstance(b, (int, float)) else a / (b if b else EPS)
