"""Walk-forward experiment runner: features -> target -> purged folds ->
LightGBM -> out-of-sample predictions -> signal and trading statistics."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import (
    funding_per_bar, perf_stats, positions_from_scores, signal_stats, simulate,
)
from .cv import walk_forward_folds
from .features import ALL_GROUPS, bar_volatility, build_features
from .labels import (
    forward_log_return, meta_label, realized_vol_target, regression_target,
    triple_barrier, tsmom_side,
)
from .model import fit_predict


@dataclass
class Experiment:
    name: str
    task: str = "reg"  # reg | tb | meta | vol
    horizon: int = 24
    groups: tuple[str, ...] = ALL_GROUPS
    train_start: str = "2018-01-01"
    test_start: str = "2021-01-01"
    test_end: str = "2026-09-01"
    retrain: str = "MS"  # monthly
    train_bars: int | None = None  # None = expanding window
    seeds: tuple[int, ...] = (0, 1, 2)
    params: dict = field(default_factory=dict)
    barrier: float = 1.0
    target_clip: float = 4.0
    meta_lookback: int = 168
    min_train_bars: int = 24 * 365
    n_trees: int | None = None  # fixed tree count instead of early stopping
    control: str | None = None  # None | "shuffle" | "leak"
    leak_noise: float = 10.0  # corr(leak, target) ≈ 1/sqrt(1 + leak_noise²) ≈ 0.1


@dataclass
class MarketDataset:
    X: pd.DataFrame
    close: pd.Series
    high: pd.Series
    low: pd.Series
    sig: pd.Series
    funding_bar: pd.Series | None

    @classmethod
    def from_raw(cls, data: dict) -> "MarketDataset":
        btc = data["btc"]
        X = build_features(btc, data.get("eth"), data.get("funding"), data.get("premium"))
        fb = funding_per_bar(data["funding"], btc.index) if "funding" in data else None
        return cls(X, btc["close"], btc["high"], btc["low"], bar_volatility(btc["close"]), fb)

    def columns_for(self, groups: tuple[str, ...]) -> list[str]:
        group_of = self.X.attrs["group_of"]
        return [c for c in self.X.columns if group_of[c] in groups]


@dataclass
class Result:
    exp: Experiment
    oos: pd.DataFrame  # pred, score, target, fwd_ret (+ side for meta)
    importance: pd.Series
    best_iters: list[int]
    seconds: float


def _targets(exp: Experiment, ds: MarketDataset) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    """(training target, evaluation target, primary side or None)."""
    h = exp.horizon
    if exp.task == "reg":
        y = regression_target(ds.close, ds.sig, h)
        return y.clip(-exp.target_clip, exp.target_clip), y, None
    if exp.task == "tb":
        y = triple_barrier(ds.close, ds.high, ds.low, ds.sig, h, exp.barrier, exp.barrier)["y"]
        return y, y, None
    if exp.task == "meta":
        side = tsmom_side(ds.close, exp.meta_lookback)
        y = meta_label(ds.close, side, h)
        return y, y, side
    if exp.task == "vol":
        y = realized_vol_target(ds.close, h)
        return y, y, None
    raise ValueError(exp.task)


def run_experiment(exp: Experiment, ds: MarketDataset, verbose: bool = True) -> Result:
    t0 = time.time()
    X = ds.X[ds.columns_for(exp.groups)].copy()
    y_train, y_eval, side = _targets(exp, ds)
    if side is not None:
        X["primary_side"] = side.astype("float32")
    rng = np.random.default_rng(12345)
    if exp.control == "leak":
        # positive control: a feature weakly correlated with the (future!) target
        z = (y_eval - y_eval.mean()) / y_eval.std()
        X["leak"] = (z + rng.normal(0, exp.leak_noise, len(z))).astype("float32")

    folds = walk_forward_folds(
        X.index, exp.test_start, exp.test_end, exp.horizon, exp.retrain,
        train_start=exp.train_start, train_bars=exp.train_bars,
        min_train_bars=exp.min_train_bars,
    )
    ok = y_train.notna().to_numpy()
    task = "binary" if exp.task in ("tb", "meta") else "regression"
    preds = pd.Series(np.nan, index=X.index)
    gain = pd.Series(0.0, index=X.columns)
    best_iters: list[int] = []
    for i, f in enumerate(folds):
        tr, va = f.train[ok[f.train]], f.valid[ok[f.valid]]
        ytr, yva = y_train.to_numpy()[tr], y_train.to_numpy()[va]
        if exp.control == "shuffle":
            ytr, yva = rng.permutation(ytr), rng.permutation(yva)
        p, info = fit_predict(X.iloc[tr], ytr, X.iloc[va], yva, X.iloc[f.test],
                              task=task, params=exp.params, seeds=exp.seeds, n_trees=exp.n_trees)
        preds.iloc[f.test] = p
        gain += info["gain"]
        best_iters.extend(info["best_iters"])
        if verbose and (i % 12 == 0 or i == len(folds) - 1):
            print(f"  [{exp.name}] fold {i + 1}/{len(folds)} {f.test_start:%Y-%m} "
                  f"train={len(tr)} best_iter={info['best_iters']} {time.time() - t0:.0f}s", flush=True)

    oos = pd.DataFrame({
        "pred": preds,
        "target": y_eval,
        "fwd_ret": forward_log_return(ds.close, exp.horizon),
    }).loc[preds.notna()]
    oos["score"] = oos["pred"] - 0.5 if task == "binary" else oos["pred"]
    if side is not None:
        oos["side"] = side.reindex(oos.index)
    imp = (gain / gain.sum()).sort_values(ascending=False)
    return Result(exp, oos, imp, best_iters, time.time() - t0)


# ------------------------------------------------------------- evaluation
def strategy_positions(res: Result, long_only: bool = False) -> pd.Series:
    """Map out-of-sample predictions to positions with the pre-declared rule."""
    h = res.exp.horizon
    o = res.oos
    if res.exp.task == "meta":
        raw = o["side"] * (o["pred"] > 0.5)
        if long_only:
            raw = raw.clip(lower=0)
        return raw.rolling(h, min_periods=1).mean()
    return positions_from_scores(o["score"], h, threshold=0.0, long_only=long_only)


def evaluate_strategy(pos: pd.Series, ds: MarketDataset, fee_bps: float, perp: bool = True) -> tuple[dict, pd.DataFrame]:
    close = ds.close.loc[pos.index[0]:pos.index[-1]]
    bt = simulate(pos, close, fee_bps, ds.funding_bar if perp else None)
    return perf_stats(bt), bt


def confidence_hit_rates(res: Result, quantiles=(0.0, 0.5, 0.8, 0.9)) -> dict:
    """Direction hit rate when |score| exceeds a *trailing* 30-day quantile
    (no look-ahead in the threshold). Shows whether confidence filtering helps."""
    o = res.oos
    s = o["score"] if res.exp.task != "meta" else o["side"] * (o["pred"] - 0.5)
    a = s.abs()
    out = {}
    for q in quantiles:
        thr = a.rolling(720, min_periods=240).quantile(q).shift(1) if q > 0 else pd.Series(-1.0, index=a.index)
        m = (a > thr) & o["fwd_ret"].notna() & (s != 0)
        out[f"hit@q{int(q * 100)}"] = float((np.sign(s[m]) == np.sign(o["fwd_ret"][m])).mean())
        out[f"n@q{int(q * 100)}"] = int(m.sum())
    return out


def yearly_table(res: Result, bt: pd.DataFrame) -> pd.DataFrame:
    o = res.oos
    rows = []
    for yr, g in o.groupby(o.index.year):
        b = bt.loc[bt.index.year == yr, "net"]
        rows.append({
            "year": yr,
            "ic_spearman": g["pred"].corr(g["target"], method="spearman"),
            "hit_rate": float((np.sign(g["score"]) == np.sign(g["fwd_ret"])).mean()) if res.exp.task != "meta" else np.nan,
            "net_return": float((1 + b).prod() - 1),
            "sharpe": float(b.mean() / b.std() * np.sqrt(24 * 365)) if b.std() > 0 else 0.0,
        })
    return pd.DataFrame(rows).set_index("year")


def summarize(res: Result) -> dict:
    o = res.oos
    if res.exp.task == "vol":
        return {"ic_pearson": float(o["pred"].corr(o["target"])), "n": len(o)}
    if res.exp.task == "meta":
        from sklearn.metrics import roc_auc_score

        d = o.dropna(subset=["target"])
        take = d["pred"] > 0.5
        return {
            "meta_auc": float(roc_auc_score(d["target"], d["pred"])),
            "primary_precision": float(d["target"].mean()),
            "filtered_precision": float(d.loc[take, "target"].mean()),
            "coverage": float(take.mean()),
            "n": int(len(d)),
        }
    return signal_stats(o["score"], o["target"], o["fwd_ret"])
