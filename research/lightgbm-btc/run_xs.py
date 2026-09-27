"""Multi-coin cross-sectional experiment (see lgbm_btc/xs.py).

    python run_xs.py        # ~20 coins, 2021-01 → 2026-08, monthly retrain

Pre-declared: fixed 300 trees (the E9 setting, no early stopping), training
rows every 4 hours, one decision per day at 00:00 UTC, 24h holding,
long top 20% / short bottom 20%, 5 bp per side. Baselines: cross-sectional
momentum (7d) and short-term reversal (1d), plus a ridge model on the same
features. Results go to results/xs_*.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lgbm_btc.model import fit_predict, fit_predict_ridge
from lgbm_btc.xs import H, build_panel, load_panel

RESULTS = Path(__file__).resolve().parent / "results"
TRAIN_START = pd.Timestamp("2019-04-01", tz="UTC")
TEST_START, TEST_END = "2021-01-01", "2026-09-01"
FEE_BPS = 5.0
PARAMS = {"min_data_in_leaf": 2000}
TRAIN_EVERY_H = 4  # sample training rows every N hours


def walk_forward(X: pd.DataFrame, features: list[str], model: str) -> pd.Series:
    t = X.index.get_level_values(0) if isinstance(X.index, pd.MultiIndex) else X.index
    ok = X["target"].notna().to_numpy()
    preds = pd.Series(np.nan, index=range(len(X)))
    months = pd.date_range(TEST_START, TEST_END, freq="MS", tz="UTC")
    for a, b in zip(months[:-1], months[1:]):
        cutoff = a - pd.Timedelta(hours=H + 24)  # purge label window + embargo
        tr = np.where(ok & (t >= TRAIN_START) & (t < cutoff) & (t.hour % TRAIN_EVERY_H == 0))[0]
        te = np.where((t >= a) & (t < b) & (t.hour == 0))[0]
        if len(te) == 0:
            continue
        Xtr, ytr = X.iloc[tr][features], X["target"].to_numpy()[tr]
        empty = Xtr.iloc[:0]
        if model == "ridge":
            p, _ = fit_predict_ridge(Xtr, ytr, empty, ytr[:0], X.iloc[te][features])
        else:
            p, _ = fit_predict(Xtr, ytr, empty, ytr[:0], X.iloc[te][features],
                               params=PARAMS, seeds=(0, 1), n_trees=300)
        preds.iloc[te] = p
        print(f"  [{model}] {a:%Y-%m} train={len(tr)} test={len(te)}", flush=True)
    return pd.Series(preds.to_numpy(), index=X.index)


def daily_ic(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby(level=0).apply(lambda g: g[col].corr(g["target"], method="spearman")).dropna()


def long_short(df: pd.DataFrame, col: str, fee_bps: float = FEE_BPS, q: float = 0.2,
               long_only: bool = False) -> pd.DataFrame:
    rows, prev = [], pd.Series(dtype=float)
    for t, g in df.groupby(level=0):
        g = g.dropna(subset=[col, "fwd_ret"]).set_index("coin")
        if len(g) < 5:
            continue
        k = max(1, int(round(q * len(g))))
        order = g[col].sort_values()
        w = pd.Series(0.0, index=g.index)
        if long_only:
            w[order.index[-k:]] = 1.0 / k
        else:
            w[order.index[-k:]] = 0.5 / k
            w[order.index[:k]] = -0.5 / k
        simple = np.exp(g["fwd_ret"]) - 1
        gross = float((w * simple).sum())
        turnover = float(w.sub(prev, fill_value=0.0).abs().sum())
        prev = w
        rows.append({"time": t, "gross": gross, "net": gross - turnover * fee_bps * 1e-4,
                     "turnover": turnover, "n": len(g)})
    return pd.DataFrame(rows).set_index("time")


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / 365
    return {"cagr": float(eq.iloc[-1] ** (1 / years) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(365)),
            "max_dd": float((eq / eq.cummax() - 1).min())}


def main() -> None:
    panel = load_panel()
    X, features = build_panel(panel)
    X = X.set_index("coin", append=True)
    print(f"rows={len(X)} features={len(features)} coins={X.index.get_level_values(1).nunique()}")
    cache = RESULTS / "cache" / "xs_preds.parquet"
    if cache.exists():
        P = pd.read_parquet(cache)
    else:
        Xr = X.reset_index(level=1)
        P = pd.DataFrame({"lgbm": walk_forward(Xr, features, "lgbm").to_numpy(),
                          "ridge": walk_forward(Xr, features, "ridge").to_numpy()}, index=X.index)
        cache.parent.mkdir(parents=True, exist_ok=True)
        P.to_parquet(cache)
    D = X[["target", "fwd_ret", "rel_ret_168", "rel_ret_24"]].join(P)
    D = D[D["lgbm"].notna()].copy()
    D["mom_7d"] = D["rel_ret_168"]
    D["rev_1d"] = -D["rel_ret_24"]
    D = D.reset_index(level=1)

    out: dict = {"n_days": int(D.index.nunique()), "coins_per_day": float(D.groupby(level=0).size().mean())}
    ic_tab, strat_tab, yearly = {}, {}, {}
    for col in ("lgbm", "ridge", "mom_7d", "rev_1d"):
        ic = daily_ic(D, col)
        ic_tab[col] = {"ic_mean": float(ic.mean()), "ic_tstat": float(ic.mean() / ic.std() * np.sqrt(len(ic))),
                       "ic_positive_days": float((ic > 0).mean())}
        bt = long_short(D, col)
        s = {**stats(bt["net"]), "sharpe_gross": float(bt["gross"].mean() / bt["gross"].std() * np.sqrt(365)),
             "turnover_per_day": float(bt["turnover"].mean())}
        for fee in (2.0, 10.0):
            s[f"sharpe@{fee:g}bps"] = stats(long_short(D, col, fee_bps=fee)["net"])["sharpe"]
        strat_tab[col] = s
        yearly[col] = {int(y): {"ic": float(ic[ic.index.year == y].mean()),
                                "sharpe": stats(bt.loc[bt.index.year == y, "net"])["sharpe"]}
                       for y in sorted(set(bt.index.year))}
    # long-only top quintile (spot, 10 bp) vs equal-weight market
    lo = long_short(D, "lgbm", fee_bps=10.0, long_only=True)
    ew = D.groupby(level=0)["fwd_ret"].apply(lambda s: float((np.exp(s) - 1).mean()))
    out["long_only_top20"] = stats(lo["net"])
    out["equal_weight_market"] = stats(ew)
    # BTC angle: does the model know whether BTC will beat the average coin tomorrow?
    btc = D[D["coin"] == "BTC"]
    beat = (btc["target"] > 0).astype(int)
    from sklearn.metrics import roc_auc_score
    out["btc_vs_market"] = {"auc": float(roc_auc_score(beat, btc["lgbm"])),
                            "hit_rate": float(((btc["lgbm"] > 0) == (btc["target"] > 0)).mean()),
                            "days": int(len(btc))}
    out.update({"ic": ic_tab, "long_short": strat_tab, "yearly": yearly})
    (RESULTS / "xs_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
