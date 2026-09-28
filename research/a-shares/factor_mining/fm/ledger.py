"""Append-only record of every factor ever evaluated.

The number of distinct formulas tried (plus every formula a GP run scored)
sets the significance bar: the more you try, the higher |t| must be.
Deleting or editing the ledger to lower the bar defeats the whole point —
final_check.py prints the trial count next to the result.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
LEDGER = RESULTS / "ledger.csv"
ACCEPTED = RESULTS / "accepted"
COLS = ["time", "kind", "name", "code_hash", "n", "kept", "reason", "t_mine", "t_required", "ic_mine",
        "same_sign_years", "t_select", "ic_select", "max_corr", "most_similar", "t_missing", "coverage",
        "leak_mismatch", "source", "rationale", "file"]


def read() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(LEDGER, dtype={"code_hash": str})


def n_trials(extra: int = 0) -> int:
    """Distinct factor formulas evaluated + formulas scored inside GP runs."""
    df = read()
    fac = df[df["kind"] == "factor"]["code_hash"].nunique() if len(df) else 0
    gp = int(df.loc[df["kind"] == "gp_batch", "n"].sum()) if len(df) else 0
    return int(fac + gp + extra)


def seen(code_hash: str) -> pd.Series | None:
    df = read()
    hit = df[(df["kind"] == "factor") & (df["code_hash"] == code_hash)]
    return None if hit.empty else hit.iloc[-1]


def append(row: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    row = {"time": dt.datetime.now().isoformat(timespec="seconds"), "kind": "factor", "n": 1, **row}
    out = pd.DataFrame([{c: row.get(c, "") for c in COLS}])
    out.to_csv(LEDGER, mode="a", header=not LEDGER.exists(), index=False)


def accepted_samples() -> dict[str, pd.Series]:
    ACCEPTED.mkdir(parents=True, exist_ok=True)
    return {p.stem: pd.read_parquet(p)["v"] for p in sorted(ACCEPTED.glob("*.parquet"))}


def save_accepted(name: str, values: pd.Series, source_code: str) -> None:
    ACCEPTED.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"v": values.astype("float32")}).to_parquet(ACCEPTED / f"{name}.parquet")
    (ACCEPTED / f"{name}.py.txt").write_text(source_code)
