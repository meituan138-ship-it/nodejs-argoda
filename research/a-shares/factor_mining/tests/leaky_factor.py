"""Deliberately broken factors: the evaluator must reject both."""
from fm.registry import factor


@factor("leak_tomorrow_return", rationale="uses tomorrow's close — must be caught", source="leak test")
def leak_tomorrow_return(d):
    c = d.close
    return c.shift(-5) / c - 1


@factor("leak_hidden_rolling", rationale="future data via reversed index, bypasses regex", source="leak test")
def leak_hidden_rolling(d):
    c = d.close.iloc[::-1]
    return (c.rolling(5).mean().iloc[::-1] / d.close - 1)


@factor("leak_survivorship", rationale="simulates a data source without delisted stocks (like Eastmoney valuation)",
        source="leak test")
def leak_survivorship(d):
    import pandas as pd
    from fm.context import CACHE
    tr = pd.read_parquet(CACHE / "daily" / "traded.parquet")       # outside data, not truncated
    alive = tr.iloc[-60:].sum() > 0                                 # still listed today
    val = d.turnover.rolling(20, min_periods=10).mean()
    return val.loc[:, alive[alive].index.intersection(val.columns)].reindex(columns=val.columns)
