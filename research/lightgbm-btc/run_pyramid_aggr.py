"""How much more aggressive can the risk-based pyramid be? Fixed rules
(N20, 2 ATR stop, add every 0.5 ATR, 4 units); vary only risk per unit and
the leverage cap. Train 2018-2023, test 2024-01 -> 2026-08."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lgbm_btc.data import load_funding, load_klines  # noqa: E402
from run_pyramid import stats  # noqa: E402
from run_pyramid_risk import simulate  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"


def main() -> None:
    data = {}
    for sym in ("BTC", "ETH", "SOL"):
        px = load_klines(f"{sym}USDT", "1h", "spot", start="2017-08" if sym != "SOL" else "2020-08", end="2026-08-31")
        fu = load_funding(f"{sym}USDT", start="2020-01" if sym != "SOL" else "2020-09", end="2026-08-31")
        fu.index = fu.index.floor("h")
        data[sym] = (px, fu)
    rows = []
    for risk in (0.02, 0.03, 0.05, 0.08, 0.10, 0.15):
        for cap in (3, 5, 10, 20):
            cfg = dict(n=20, m=2.0, risk=risk, k=0.5, filter=False, lev_cap=cap)
            row = {"risk": risk, "lev_cap": cap}
            for sym, (px, fu) in data.items():
                for per, (s, e) in {"train": ("2018-01-01" if sym != "SOL" else "2021-01-01", "2024-01-01"),
                                    "test": ("2024-01-01", None), "all": ("2018-01-01" if sym != "SOL" else "2021-01-01", None)}.items():
                    eq, tr = simulate(px, fu, cfg, s, e)
                    st = stats(eq, tr)
                    d = eq.resample("1D").last()
                    # average and peak account leverage actually used
                    row[f"{sym}_{per}_final"] = st["final"]
                    row[f"{sym}_{per}_cagr"] = st["cagr"]
                    row[f"{sym}_{per}_dd"] = st["max_dd"]
                    row[f"{sym}_{per}_liq"] = int((tr["exit"] == "liquidated").sum())
                    row[f"{sym}_{per}_worst_trade"] = float((tr["pnl"] / 1).min()) if len(tr) else 0.0
                    row[f"{sym}_{per}_by_year"] = json.dumps({y: round(v, 3) for y, v in st["by_year"].items()})
            rows.append(row)
            print(f"risk {risk:>4.0%} cap {cap:>2}x | " + " | ".join(
                f"{s} test {row[f'{s}_test_final']:>7.0f} dd {row[f'{s}_test_dd']:.0%} liq {row[f'{s}_test_liq']}"
                f" / all {row[f'{s}_all_final']:>9.0f} dd {row[f'{s}_all_dd']:.0%}" for s in data), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "pyramid_aggr_grid.csv", index=False)


if __name__ == "__main__":
    main()
