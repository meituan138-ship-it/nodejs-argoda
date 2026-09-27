"""Train the main (E1-style) model on all labelled history and print the
signal for the most recent closed bar.

    python predict_latest.py                  # 24h horizon
    python predict_latest.py --horizon 168    # one-week horizon

The public archive lags real time by about a day, and funding for the
current month is estimated from the premium index. For live trading, feed
``build_features`` with exchange-API klines instead; the logic is identical.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from lgbm_btc.data import load_market_data
from lgbm_btc.labels import regression_target
from lgbm_btc.model import fit_predict
from lgbm_btc.pipeline import MarketDataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--end", default=None, help="last UTC day to load (YYYY-MM-DD), default yesterday")
    args = ap.parse_args()
    h = args.horizon

    data = load_market_data(start="2017-08", end=args.end, extend_funding=True)
    ds = MarketDataset.from_raw(data)
    X = ds.X
    y = regression_target(ds.close, ds.sig, h).clip(-4, 4).to_numpy()

    labelled = np.where(np.isfinite(y) & (X.index >= pd.Timestamp("2018-01-01", tz="UTC")))[0]
    n_valid = int(len(labelled) * 0.15)
    valid = labelled[-n_valid:]
    train = labelled[: len(labelled) - n_valid - h]  # purge h bars before the validation slice
    recent = np.arange(len(X) - h, len(X))           # the last h bars have no label yet
    pred, info = fit_predict(X.iloc[train], y[train], X.iloc[valid], y[valid], X.iloc[recent])

    ls_pos = float(np.mean(np.sign(pred)))
    lo_pos = float(np.mean(np.sign(pred) > 0))
    last = X.index[-1]
    print(f"last closed bar : {last:%Y-%m-%d %H:%M} UTC (+1h)   close = {ds.close.iloc[-1]:,.2f}")
    print(f"trees (3 seeds) : {info['best_iters']}")
    print(f"score (vol-normalised {h}h return forecast): {pred[-1]:+.3f}")
    print(f"sign of the last {h} scores: +{int((pred > 0).sum())} / -{int((pred < 0).sum())}")
    print(f"position, long/short perp : {ls_pos:+.2f}   (mean of the last {h} signs)")
    print(f"position, long-only spot  : {lo_pos:.2f}")
    print("The edge is small and regime-dependent (see README §4); use it as one input, not an oracle.")


if __name__ == "__main__":
    main()
