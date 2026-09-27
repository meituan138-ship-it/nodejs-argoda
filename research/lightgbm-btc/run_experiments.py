"""Run the pre-declared experiment set on real Binance data and write results/.

    python run_experiments.py            # everything (~1h on 4 cores)
    python run_experiments.py --only E1_reg_h24 C2_leak_h24

The list below was fixed before looking at any test-period result; all of
them are reported, including the ones that fail, and the deflated Sharpe
ratio accounts for the number of strategy variants tried.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lgbm_btc.backtest import deflated_sharpe, perf_stats, positions_from_scores, simulate
from lgbm_btc.cv import walk_forward_folds
from lgbm_btc.data import load_market_data
from lgbm_btc.features import PRICE_ONLY_GROUPS
from lgbm_btc.labels import realized_vol_target
from lgbm_btc.pipeline import (
    Experiment, MarketDataset, Result, confidence_hit_rates, evaluate_strategy,
    run_experiment, strategy_positions, summarize, yearly_table,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
END = "2026-08-31"  # last fully published month in the archive at the time of the run

EXPERIMENTS = [
    Experiment("E1_reg_h24"),                                   # main model
    Experiment("E2_reg_h4", horizon=4),                         # shorter horizon
    Experiment("E3_tb_h24", task="tb"),                         # triple-barrier labels
    Experiment("E4_meta_tsmom_h24", task="meta"),               # meta-labeling on 7d momentum
    Experiment("E5_reg_h24_price_only", groups=PRICE_ONLY_GROUPS),  # no ETH / funding / basis
    Experiment("E6_reg_h24_rolling2y", train_bars=2 * 365 * 24),    # rolling instead of expanding
    Experiment("E7_vol_h24", task="vol"),                       # volatility, not direction
    Experiment("E8_reg_h168", horizon=168),                     # one-week horizon
    # added after the E1 training log showed early stopping often picking 1-10 trees
    # (before any test-period result was looked at): fixed tree count instead
    Experiment("E9_reg_h24_fixed300", n_trees=300),
    Experiment("C1_shuffle_h24", control="shuffle", seeds=(0,)),  # negative control
    Experiment("C2_leak_h24", control="leak", seeds=(0,)),        # positive control (corr≈0.1 leak)
]
PERP_FEE_BPS = 5.0   # Binance USD-M taker ≈ 4.5–5 bps
SPOT_FEE_BPS = 10.0  # Binance spot taker


def _load_or_run(exp: Experiment, ds: MarketDataset, force: bool) -> Result:
    path = CACHE / f"{exp.name}.parquet"
    if path.exists() and not force:
        oos = pd.read_parquet(path)
        imp = pd.read_parquet(CACHE / f"{exp.name}_importance.parquet")["gain"]
        meta = json.loads((CACHE / f"{exp.name}.json").read_text())
        return Result(exp, oos, imp, meta["best_iters"], meta["seconds"])
    print(f"running {exp.name} ...", flush=True)
    res = run_experiment(exp, ds)
    CACHE.mkdir(parents=True, exist_ok=True)
    res.oos.to_parquet(path)
    res.importance.rename("gain").to_frame().to_parquet(CACHE / f"{exp.name}_importance.parquet")
    (CACHE / f"{exp.name}.json").write_text(json.dumps({"best_iters": res.best_iters, "seconds": res.seconds}))
    return res


def _baselines(ds: MarketDataset, index: pd.DatetimeIndex) -> dict[str, tuple[pd.Series, bool, float]]:
    """name -> (position, is_perp, fee_bps)."""
    close = ds.close.reindex(index)
    mom = np.log(ds.close / ds.close.shift(168)).reindex(index)
    return {
        "BH_spot": (pd.Series(1.0, index=index), False, SPOT_FEE_BPS),
        "Long_perp": (pd.Series(1.0, index=index), True, PERP_FEE_BPS),
        "TSMOM168_LS_perp": (positions_from_scores(mom, 24), True, PERP_FEE_BPS),
        "TSMOM168_LO_spot": (positions_from_scores(mom, 24, long_only=True), False, SPOT_FEE_BPS),
    } if len(close) else {}


def _vol_study(res: Result, ds: MarketDataset) -> dict:
    """LightGBM vs HAR vs naive for next-24h realized vol, plus a vol-targeting use case."""
    exp = res.exp
    h = exp.horizon
    r2 = np.log(ds.close).diff() ** 2
    har = pd.DataFrame({
        "d": 0.5 * np.log(r2.rolling(24).sum() + 1e-12),
        "w": 0.5 * np.log(r2.rolling(168).sum() / 7 + 1e-12),
        "m": 0.5 * np.log(r2.rolling(720).sum() / 30 + 1e-12),
    })
    y = realized_vol_target(ds.close, h)
    folds = walk_forward_folds(har.index, exp.test_start, exp.test_end, h, exp.retrain,
                               train_start=exp.train_start)
    har_pred = pd.Series(np.nan, index=har.index)
    for f in folds:
        tr = np.concatenate([f.train, f.valid])
        A = np.c_[np.ones(len(tr)), har.iloc[tr].to_numpy()]
        ok = np.isfinite(A).all(1) & np.isfinite(y.iloc[tr].to_numpy())
        beta = np.linalg.lstsq(A[ok], y.iloc[tr].to_numpy()[ok], rcond=None)[0]
        har_pred.iloc[f.test] = np.c_[np.ones(len(f.test)), har.iloc[f.test].to_numpy()] @ beta
    df = pd.DataFrame({"y": res.oos["target"], "lgbm": res.oos["pred"],
                       "har": har_pred.reindex(res.oos.index), "naive": har["d"].reindex(res.oos.index)}).dropna()
    mse = {k: float(((df[k] - df["y"]) ** 2).mean()) for k in ("lgbm", "har", "naive")}
    out = {
        "mse": mse,
        "r2_vs_naive": {k: float(1 - mse[k] / mse["naive"]) for k in ("lgbm", "har")},
        "corr": {k: float(df[k].corr(df["y"])) for k in ("lgbm", "har", "naive")},
    }
    # vol targeting of a long-only spot position, rebalanced once a day at 00:00 UTC
    target_hourly = 0.5 / np.sqrt(24 * 365)  # 50% annualised
    idx = res.oos.index
    close = ds.close.reindex(idx)
    strategies = {"BH_spot": pd.Series(1.0, index=idx)}
    for k in ("naive", "har", "lgbm"):
        sig_hat = np.exp(df[k].reindex(idx)) / np.sqrt(h)
        pos = (target_hourly / sig_hat).clip(upper=1.0)
        daily = pos.where(idx.hour == 0).ffill().fillna(0.0)
        strategies[f"VolTarget_{k}"] = daily
    out["vol_targeting"] = {k: perf_stats(simulate(p, close, SPOT_FEE_BPS)) for k, p in strategies.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="experiment names to run")
    ap.add_argument("--force", action="store_true", help="ignore cached predictions")
    args = ap.parse_args()

    data = load_market_data(start="2017-08", end=END)
    ds = MarketDataset.from_raw(data)
    print(f"bars={len(ds.X)} features={ds.X.shape[1]} missing_bars_filled={data['btc'].attrs['n_missing_bars']}")

    exps = [e for e in EXPERIMENTS if not args.only or e.name in args.only]
    results = {e.name: _load_or_run(e, ds, args.force) for e in exps}

    RESULTS.mkdir(exist_ok=True)
    summary_rows, yearly, strat_rows, daily_returns = [], {}, [], {}
    test_index = next(iter(results.values())).oos.index

    for name, (pos, perp, fee) in _baselines(ds, test_index).items():
        stats, bt = evaluate_strategy(pos, ds, fee, perp)
        strat_rows.append({"strategy": name, "family": "baseline", **stats})
        daily_returns[name] = bt["net"].groupby(bt.index.floor("D")).sum()

    vol_study = None
    for name, res in results.items():
        row = {"experiment": name, "task": res.exp.task, "horizon": res.exp.horizon,
               "median_trees": float(np.median(res.best_iters)), "minutes": round(res.seconds / 60, 1)}
        row.update(summarize(res))
        if res.exp.task == "vol":
            vol_study = _vol_study(res, ds)
            summary_rows.append(row)
            continue
        row.update(confidence_hit_rates(res))
        summary_rows.append(row)
        for variant, long_only, perp, fee in (("LS_perp", False, True, PERP_FEE_BPS),
                                              ("LO_spot", True, False, SPOT_FEE_BPS)):
            pos = strategy_positions(res, long_only=long_only)
            stats, bt = evaluate_strategy(pos, ds, fee, perp)
            sname = f"{name}_{variant}"
            fam = "control" if res.exp.control else "ml"
            strat_rows.append({"strategy": sname, "family": fam, **stats})
            daily_returns[sname] = bt["net"].groupby(bt.index.floor("D")).sum()
            if variant == "LS_perp":
                yearly[name] = yearly_table(res, bt)
                for fee_alt in (0.0, 2.0, 10.0):
                    st, _ = evaluate_strategy(pos, ds, fee_alt, perp)
                    strat_rows[-1][f"sharpe@{fee_alt:g}bps"] = st["sharpe"]

    strat = pd.DataFrame(strat_rows).set_index("strategy")
    # deflated Sharpe over all ML strategy variants that were tried
    ml = [s for s in strat.index if strat.loc[s, "family"] == "ml"]
    trial_sr = [float(daily_returns[s].mean() / daily_returns[s].std()) for s in ml]
    strat["dsr"] = np.nan
    for s in ml:
        strat.loc[s, "dsr"] = deflated_sharpe(daily_returns[s], trial_sr)

    summary = pd.DataFrame(summary_rows).set_index("experiment")
    summary.to_csv(RESULTS / "signal_summary.csv", float_format="%.4f")
    strat.to_csv(RESULTS / "strategy_summary.csv", float_format="%.4f")
    pd.concat(yearly, names=["experiment"]).to_csv(RESULTS / "yearly.csv", float_format="%.4f")
    pd.DataFrame(daily_returns).to_csv(RESULTS / "daily_returns.csv", float_format="%.6f")
    imp = pd.DataFrame({n: r.importance for n, r in results.items() if not r.exp.control})
    imp.to_csv(RESULTS / "feature_importance.csv", float_format="%.5f")
    if vol_study:
        (RESULTS / "vol_study.json").write_text(json.dumps(vol_study, indent=2))

    pd.set_option("display.width", 200)
    print(summary.round(4).to_string())
    print(strat.round(3).to_string())
    if vol_study:
        print(json.dumps(vol_study, indent=2))


if __name__ == "__main__":
    main()
