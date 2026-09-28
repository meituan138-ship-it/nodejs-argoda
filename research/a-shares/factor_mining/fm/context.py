"""Data handed to factor functions: `d.close`, `d.turnover`, ... (wide daily frames)."""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "cache"

FIELDS = {
    "open": "开盘价（后复权）", "high": "最高价（后复权）", "low": "最低价（后复权）", "close": "收盘价（后复权）",
    "raw_close": "收盘价（不复权，真实价格）", "raw_open": "开盘价（不复权）",
    "volume": "成交量（股）", "amount": "成交额（元）", "turnover": "换手率（占流通股比例，0.01 = 1%）",
    "mf_net": "主力净流入（元，新浪逐笔汇总）", "mf_xl_net": "超大单净流入（元）",
    "mf_net_ratio": "主力净流入 / 成交额", "traded": "当天是否有成交（1/0）",
    "limit_pct": "涨跌停幅度（0.1 或 0.2；ST 的 5% 未标注）",
}


class Context:
    """Lazy loader of the cached daily frames. `truncate(t)` gives a copy that
    only contains data up to day t (used by the look-ahead test)."""

    def __init__(self, cache: Path = CACHE, until: pd.Timestamp | None = None, _frames: dict | None = None):
        self._cache = Path(cache)
        self._until = until
        self._frames = _frames if _frames is not None else {}

    def _get(self, name: str) -> pd.DataFrame:
        if name not in self._frames:
            self._frames[name] = pd.read_parquet(self._cache / "daily" / f"{name}.parquet")
        df = self._frames[name]
        return df.loc[: self._until] if self._until is not None else df

    def __getattr__(self, name):
        if name in FIELDS:
            return self._get(name)
        raise AttributeError(f"unknown field {name!r}; available: {', '.join(FIELDS)}")

    @cached_property
    def ret(self) -> pd.DataFrame:
        """Daily log return (NaN on days without trading)."""
        import numpy as np
        return np.log(self.close / self.close.shift(1)).where(self.traded > 0)

    @cached_property
    def mkt_ret(self) -> pd.Series:
        """Equal-weight market daily log return."""
        return self.ret.mean(axis=1)

    def truncate(self, t: pd.Timestamp) -> "Context":
        return Context(self._cache, t, self._frames)
