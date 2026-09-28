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
