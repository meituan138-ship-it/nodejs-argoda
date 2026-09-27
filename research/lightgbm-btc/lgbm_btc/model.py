"""LightGBM training for low signal-to-noise targets.

Parameters lean conservative, as in the Numerai/G-Research/Qlib examples:
small learning rate, shallow trees, large ``min_data_in_leaf``, row and
column subsampling, L2 penalty. The number of trees is chosen by early
stopping on a purged validation slice, then the model is refit on
train+valid (the most recent data matters most in a drifting market) with
the tree count scaled up proportionally. Several seeds are averaged.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_PARAMS: dict = {
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "verbosity": -1,
    "num_threads": 4,
}
TASK_OBJECTIVE = {
    "regression": {"objective": "regression", "metric": "None"},
    "binary": {"objective": "binary", "metric": "auc"},
}


def _pearson_eval(preds: np.ndarray, data: lgb.Dataset):
    y = data.get_label()
    if np.std(preds) < 1e-12:
        return "corr", 0.0, True
    return "corr", float(np.corrcoef(preds, y)[0, 1]), True


def fit_predict(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_valid: pd.DataFrame, y_valid: np.ndarray,
    X_test: pd.DataFrame,
    task: str = "regression",
    params: dict | None = None,
    seeds: tuple[int, ...] = (0, 1, 2),
    max_rounds: int = 3000,
    early_stopping: int = 200,
    refit: bool = True,
    n_trees: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Return the seed-averaged test prediction and a small info dict.

    ``n_trees`` skips early stopping and fits a fixed number of trees on
    train+valid (more stable when the validation IC is too noisy to pick one).
    """
    p = {**BASE_PARAMS, **TASK_OBJECTIVE[task], **(params or {})}
    feval = _pearson_eval if task == "regression" else None
    preds, best_iters = [], []
    gain = pd.Series(0.0, index=X_train.columns)
    for seed in seeds:
        ps = {**p, "seed": seed}
        if n_trees is not None:
            full = lgb.Dataset(pd.concat([X_train, X_valid]), np.concatenate([y_train, y_valid]))
            booster = lgb.train(ps, full, num_boost_round=n_trees)
            preds.append(booster.predict(X_test))
            best_iters.append(n_trees)
            gain += pd.Series(booster.feature_importance("gain"), index=X_train.columns)
            continue
        dtr = lgb.Dataset(X_train, y_train, free_raw_data=False)
        dva = lgb.Dataset(X_valid, y_valid, reference=dtr, free_raw_data=False)
        booster = lgb.train(
            ps, dtr, num_boost_round=max_rounds, valid_sets=[dva], feval=feval,
            callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
        )
        best = max(booster.best_iteration, 1)
        best_iters.append(best)
        if refit:
            n_full = int(round(best * (len(X_train) + len(X_valid)) / len(X_train)))
            full = lgb.Dataset(pd.concat([X_train, X_valid]), np.concatenate([y_train, y_valid]))
            booster = lgb.train(ps, full, num_boost_round=max(n_full, 1))
            preds.append(booster.predict(X_test))
        else:
            preds.append(booster.predict(X_test, num_iteration=best))
        gain += pd.Series(booster.feature_importance("gain"), index=X_train.columns)
    info = {"best_iters": best_iters, "gain": gain / len(seeds)}
    return np.mean(preds, axis=0), info
