"""Targets. Never the price level: a model trained on prices only learns
"tomorrow ≈ today" and produces the famous lagged-copy charts.

- ``regression_target``: forward log return scaled by current volatility,
  the G-Research / Numerai style continuous target.
- ``triple_barrier``: López de Prado's path-dependent label (which of the
  ±k·σ·√h barriers is touched first within h bars).
- ``meta_label``: did a primary rule's trade make money? (meta-labeling)
- ``realized_vol_target``: log realized volatility of the next h bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """log(C[t+h] / C[t]): the return earned by a position opened at the close of bar t."""
    return np.log(close.shift(-horizon) / close)


def regression_target(close: pd.Series, sig: pd.Series, horizon: int) -> pd.Series:
    return forward_log_return(close, horizon) / (sig * np.sqrt(horizon))


def triple_barrier(
    close: pd.Series, high: pd.Series, low: pd.Series, sig: pd.Series,
    horizon: int, pt: float = 1.0, sl: float = 1.0,
) -> pd.DataFrame:
    """Barriers at C[t]·exp(±k·σ[t]·√h); the vertical barrier is h bars ahead.

    Returns columns ``side`` (+1 upper first, -1 lower first, 0 vertical),
    ``ret`` (log return at exit), ``bars`` (holding period) and ``y`` —
    the binary target: 1 if upper first, 0 if lower first, sign of the
    return at the vertical barrier otherwise. A bar that touches both
    barriers counts as a stop-out (we cannot know the intrabar order, so
    we assume the unfavourable one).
    """
    c = close.to_numpy(float)
    hi = high.to_numpy(float)
    lo = low.to_numpy(float)
    width = sig.to_numpy(float) * np.sqrt(horizon)
    upper = c * np.exp(pt * width)
    lower = c * np.exp(-sl * width)
    n = len(c)

    side = np.zeros(n)
    ret = np.full(n, np.nan)
    bars = np.full(n, np.nan)
    done = np.zeros(n, dtype=bool)
    for j in range(1, horizon + 1):
        hj = np.full(n, np.nan)
        lj = np.full(n, np.nan)
        hj[:-j] = hi[j:]
        lj[:-j] = lo[j:]
        up = ~done & (hj >= upper)
        dn = ~done & (lj <= lower)
        stop = dn  # includes the ambiguous "both touched" bars
        take = up & ~dn
        side[take], ret[take], bars[take] = 1, np.log(upper[take] / c[take]), j
        side[stop], ret[stop], bars[stop] = -1, np.log(lower[stop] / c[stop]), j
        done |= take | stop

    fwd = forward_log_return(close, horizon).to_numpy()
    vertical = ~done
    ret[vertical] = fwd[vertical]
    bars[vertical] = horizon
    y = np.where(side == 1, 1.0, np.where(side == -1, 0.0, (fwd > 0).astype(float)))
    valid = np.isfinite(width) & np.isfinite(fwd)
    y[~valid] = np.nan
    ret[~valid] = np.nan
    return pd.DataFrame({"side": side, "ret": ret, "bars": bars, "y": y}, index=close.index)


def tsmom_side(close: pd.Series, lookback: int = 168) -> pd.Series:
    """Primary model for meta-labeling: time-series momentum (Liu & Tsyvinski 2021)."""
    return np.sign(np.log(close / close.shift(lookback))).replace(0, 1.0)


def meta_label(close: pd.Series, side: pd.Series, horizon: int, min_ret: float = 0.0) -> pd.Series:
    """1 if the primary rule's trade over the next ``horizon`` bars earns > ``min_ret``."""
    fwd = forward_log_return(close, horizon)
    y = (side * fwd > min_ret).astype(float)
    return y.where(fwd.notna() & side.notna())


def realized_vol_target(close: pd.Series, horizon: int) -> pd.Series:
    """log sqrt(sum of squared one-bar log returns over bars t+1..t+h)."""
    r2 = np.log(close).diff() ** 2
    fwd_sum = r2.rolling(horizon).sum().shift(-horizon)
    return 0.5 * np.log(fwd_sum + 1e-12)
