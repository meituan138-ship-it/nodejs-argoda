"""Second half of the factor lab (split out to fit in memory): retrain with the
13 factors that passed run_factor_lab.py's mining + selection gates, then open
the sealed 2023-01 → 2026-09 hold-out once and compare with the baseline."""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lightgbm-btc"))
from ash.data import load_all  # noqa: E402
from ash.factors_new import build_candidates  # noqa: E402
from ash.moneyflow import load as load_mf  # noqa: E402
from ash.panel import build, wide  # noqa: E402
from lgbm_btc.model import fit_predict  # noqa: E402
import run_ashare as RA  # noqa: E402
import run_small_capital as SC  # noqa: E402
from run_factor_lab import HOLD, weekly_ic, t_of  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
KEPT = ['mf_net_ratio_1', 'mf_net_ratio_5', 'mf_net_ratio_20', 'mf_xl_ratio_5', 'mf_pos_days_10', 'turnover_5',
        'turnover_20', 'turnover_std_20', 'gap_up_count_20', 'up_volume_share_20', 'pv_corr_60', 'limit_up_recent_5',
        'max_ret_5']


def main():
    stocks = load_all()
    X, feats, ex = build(stocks)
    X["lgbm_base"] = pd.read_parquet(RES / "ashare_preds.parquet")["lgbm"].to_numpy()
    num = X.select_dtypes("float64").columns
    X[num] = X[num].astype("float32")
    days = ex["days"]
    mf = load_mf(list(stocks))
    C = build_candidates(ex["open"], ex["high"], wide(stocks, "low").reindex(days), ex["close"],
                         wide(stocks, "volume").reindex(days), wide(stocks, "raw_close").reindex(days), mf)
    del mf
    dec = pd.DatetimeIndex(X["date"].unique())
    keys = pd.MultiIndex.from_frame(X[["date", "code"]])
    for n in KEPT:
        X[n] = C[n][0].reindex(dec).stack(future_stack=True).reindex(keys).to_numpy().astype("float32")
        X[f"xs_{n}"] = X.groupby("date")[n].rank(pct=True).astype("float32")
    del C
    gc.collect()
    new_feats = feats + KEPT + [f"xs_{n}" for n in KEPT]
    pred = np.full(len(X), np.nan)
    d = X["date"]
    ok = X["target"].notna().to_numpy()
    qs = pd.date_range(HOLD[0], d.max() + pd.offsets.QuarterBegin(1), freq="QS")
    for a, b in zip(qs[:-1], qs[1:]):
        cut = a - pd.Timedelta(days=14)
        tr = np.where(ok & (d >= "2013-01-01").to_numpy() & (d < cut).to_numpy())[0]
        te = np.where(((d >= a) & (d < b)).to_numpy())[0]
        if not len(te):
            continue
        Xtr, ytr = X.iloc[tr][new_feats], X["target"].to_numpy()[tr]
        p, info = fit_predict(Xtr, ytr, Xtr.iloc[:0], ytr[:0], X.iloc[te][new_feats],
                              params={"min_data_in_leaf": 2000}, seeds=(0, 1), n_trees=300)
        pred[te] = p
        del Xtr
        gc.collect()
        print(f"  retrain {a:%Y-%m} train={len(tr)}", flush=True)
    X["lgbm_new"] = pred
    imp = info["gain"] / info["gain"].sum()
    # ---- hold-out opened once
    H = X[(X["date"] >= HOLD[0]) & X["lgbm_new"].notna()].copy()
    raw_open = wide(stocks, "raw_open").reindex(days)
    out = {"kept": KEPT, "new_factor_gain_share_last_model": float(imp[[f for f in imp.index if f in KEPT or f[3:] in KEPT]].sum())}
    for col in ("lgbm_base", "lgbm_new"):
        ic = weekly_ic(H, col)
        bt = RA.backtest(H, col, ex, 50)
        T = H[["date", "code", col]].rename(columns={col: "lgbm"})
        small = SC.summary(SC.simulate(T, ex, raw_open, n=8, min_comm=0.0))
        out[col] = {"ic": float(ic.mean()), "ic_t": t_of(ic),
                    "ic_by_year": {int(y): float(ic[ic.index.year == y].mean()) for y in sorted(set(ic.index.year))},
                    "top50": RA.stats(bt["net"]),
                    "top50_by_year": {int(y): float((1 + bt["net"][bt.index.year == y]).prod() - 1) for y in sorted(set(bt.index.year))},
                    "small8": {k: v for k, v in small.items() if k != "by_year"}, "small8_by_year": small["by_year"]}
        print(col, json.dumps(out[col], ensure_ascii=False, default=float), flush=True)
    (RES / "factor_lab_holdout.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
