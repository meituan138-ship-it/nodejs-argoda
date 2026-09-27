"""Purged walk-forward splits.

Random K-fold on overlapping financial labels leaks the future (López de
Prado, pitfall #8). Here every test block is strictly later than its
training data, and training rows whose label window (t, t+h] would reach
into the next block are purged, plus an embargo for serial correlation.
The same purge is applied between the inner train and the early-stopping
validation slice.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Fold:
    train: np.ndarray  # integer row positions
    valid: np.ndarray
    test: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_folds(
    index: pd.DatetimeIndex,
    test_start: str,
    test_end: str,
    horizon: int,
    freq: str = "MS",
    train_start: str | None = None,
    train_bars: int | None = None,
    valid_frac: float = 0.15,
    embargo: int = 24,
    min_train_bars: int = 24 * 365,
) -> list[Fold]:
    """One fold per ``freq`` period in [test_start, test_end).

    ``train_bars=None`` gives an expanding window, otherwise a rolling one.
    """
    idx = pd.DatetimeIndex(index)
    tz = idx.tz
    starts = pd.date_range(pd.Timestamp(test_start, tz=tz), pd.Timestamp(test_end, tz=tz), freq=freq)
    bounds = list(starts)
    end_ts = pd.Timestamp(test_end, tz=tz)
    if not bounds or bounds[-1] < end_ts:
        bounds.append(end_ts)
    first_row = 0 if train_start is None else int(idx.searchsorted(pd.Timestamp(train_start, tz=tz)))

    folds = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        lo, hi = idx.searchsorted(a), idx.searchsorted(b)
        if hi <= lo:
            continue
        # a training label must be fully observed before the first test decision
        train_end = lo - horizon - embargo
        begin = first_row if train_bars is None else max(first_row, train_end - train_bars)
        if train_end - begin < min_train_bars:
            continue
        n_valid = int((train_end - begin) * valid_frac)
        valid = np.arange(train_end - n_valid, train_end)
        train = np.arange(begin, train_end - n_valid - horizon)
        folds.append(Fold(train, valid, np.arange(lo, hi), a, b))
    return folds
