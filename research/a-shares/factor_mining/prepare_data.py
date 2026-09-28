"""One-off: download / load the data and build the cache the miner uses.

    python prepare_data.py            # first run downloads ~2-4 hours (free APIs, rate-limited)

Creates cache/:
  daily/<field>.parquet   wide daily frames (dates × stock codes), float32
  weekly.parquet          (date, code, target) for every point-in-time universe member
  base_sample.parquet     existing 73 model features on every 4th week (redundancy check)
  panel.parquet           full weekly panel + baseline predictions (used only by final_check.py)
  meta.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # research/a-shares
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "lightgbm-btc"))
from ash.data import load_all  # noqa: E402
from ash.moneyflow import load as load_mf  # noqa: E402
from ash.panel import build, limit_pct, wide  # noqa: E402

CACHE = HERE / "cache"


def main() -> None:
    print("loading daily bars (downloads on first run)...", flush=True)
    stocks = load_all()
    print(f"{len(stocks)} stocks; building weekly panel...", flush=True)
    X, feats, ex = build(stocks)
    days = ex["days"]
    print("loading money flow (downloads on first run)...", flush=True)
    mf = load_mf(list(stocks))

    (CACHE / "daily").mkdir(parents=True, exist_ok=True)
    vol = wide(stocks, "volume").reindex(days)
    raw_close = wide(stocks, "raw_close").reindex(days)
    traded = vol.fillna(0) > 0

    def mfw(col):
        return pd.DataFrame({k: d[col] for k, d in mf.items() if col in d}).reindex(index=days, columns=ex["close"].columns).where(traded)

    fields = {
        "open": ex["open"], "high": ex["high"], "low": wide(stocks, "low").reindex(days), "close": ex["close"],
        "raw_close": raw_close, "raw_open": wide(stocks, "raw_open").reindex(days),
        "volume": vol * 100,                                  # shares
        "amount": (vol * 100 * raw_close).where(traded),      # CNY
        "turnover": mfw("turnover") / 1e4,                    # fraction of free float
        "mf_net": mfw("netamount"),                           # main-force net inflow, CNY
        "mf_xl_net": mfw("r0_net"),                           # extra-large orders net inflow, CNY
        "mf_net_ratio": mfw("ratioamount"),                   # main-force net / traded amount
        "traded": traded.astype(float),
        "limit_pct": limit_pct(ex["close"].columns, days),
    }
    for name, frame in fields.items():
        frame.astype("float32").to_parquet(CACHE / "daily" / f"{name}.parquet")
        print(f"  saved {name} {frame.shape}", flush=True)

    X[["date", "code", "target"]].to_parquet(CACHE / "weekly.parquet")
    base = [f for f in feats if not f.startswith("xs_") and f not in ("mkt_ret_5", "mkt_ret_20", "mkt_vol_20", "breadth_20")]
    sample_dates = sorted(X["date"].unique())[::4]
    X.loc[X["date"].isin(sample_dates), ["date", "code"] + base].to_parquet(CACHE / "base_sample.parquet")
    preds = ROOT / "results" / "ashare_preds.parquet"
    if preds.exists():
        X["lgbm_base"] = pd.read_parquet(preds)["lgbm"].to_numpy()
    num = X.select_dtypes("float64").columns
    X[num] = X[num].astype("float32")
    X.to_parquet(CACHE / "panel.parquet")
    (CACHE / "meta.json").write_text(json.dumps({
        "features": feats, "base_features": base, "n_stocks": len(stocks),
        "first_day": str(days[0].date()), "last_day": str(days[-1].date()),
    }, indent=2))
    print("done:", CACHE, flush=True)


if __name__ == "__main__":
    main()
