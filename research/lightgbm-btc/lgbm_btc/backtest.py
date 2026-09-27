"""Cost-aware backtest and statistics.

Conventions: ``pos[t]`` is the position held from the close of bar t to the
close of bar t+1, so it earns ``close[t+1]/close[t] - 1``. Fees are charged
on every change of position (``fee_bps`` per unit of notional traded, one
side). For perpetual futures the position also pays/receives funding at
each settlement it is held through (longs pay positive funding).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

BARS_PER_YEAR = 24 * 365
EULER_GAMMA = 0.5772156649015329


def positions_from_scores(
    score: pd.Series, horizon: int, threshold: float = 0.0, long_only: bool = False,
) -> pd.Series:
    """Sign of the score (0 inside the dead band), averaged over the last
    ``horizon`` bars: h overlapping sub-portfolios, each held for h bars
    (Jegadeesh-Titman). This matches the label horizon and cuts turnover."""
    raw = np.sign(score) * (score.abs() > threshold)
    if long_only:
        raw = raw.clip(lower=0)
    return raw.rolling(horizon, min_periods=1).mean().fillna(0.0)


def funding_per_bar(funding: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Map each settlement at time T onto the bar whose position is held through T.

    The position chosen at the close of bar t (time t+1bar) is held until
    t+2bar, so a settlement at T is paid by ``pos`` indexed at T - 2 bars.
    """
    bar = index[1] - index[0]
    f = funding.copy()
    f.index = f.index - 2 * bar
    return f.groupby(level=0).sum().reindex(index).fillna(0.0)


def simulate(
    pos: pd.Series, close: pd.Series, fee_bps: float = 5.0, funding_bar: pd.Series | None = None,
) -> pd.DataFrame:
    pos = pos.reindex(close.index).fillna(0.0)
    ret_next = (close.shift(-1) / close - 1).fillna(0.0)
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    gross = pos * ret_next
    cost = turnover * fee_bps * 1e-4
    fund = pos * funding_bar.reindex(close.index).fillna(0.0) if funding_bar is not None else 0.0 * pos
    net = gross - cost - fund
    return pd.DataFrame({"pos": pos, "gross": gross, "cost": cost, "funding": fund,
                         "net": net, "turnover": turnover})


def perf_stats(bt: pd.DataFrame, col: str = "net") -> dict:
    r = bt[col]
    years = len(r) / BARS_PER_YEAR
    equity = (1 + r).cumprod()
    daily = r.groupby(r.index.floor("D")).sum()
    ann_ret = r.mean() * BARS_PER_YEAR
    ann_vol = r.std() * np.sqrt(BARS_PER_YEAR)
    active = bt["pos"].abs() > 1e-9
    return {
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1) if equity.iloc[-1] > 0 else -1.0,
        "ann_ret": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else 0.0,
        "sharpe_daily": float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0,
        "max_dd": float((equity / equity.cummax() - 1).min()),
        "turnover_per_year": float(bt["turnover"].sum() / years),
        "exposure": float(bt["pos"].abs().mean()),
        "cost_per_year": float(bt["cost"].sum() / years),
        "funding_per_year": float(bt["funding"].sum() / years),
        "hit_rate": float((bt.loc[active, "gross"] > 0).mean()) if active.any() else float("nan"),
    }


def probabilistic_sharpe(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR (Bailey & López de Prado 2012) on non-annualised per-period returns."""
    r = returns.dropna()
    n = len(r)
    sr = r.mean() / r.std()
    skew, kurt = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def deflated_sharpe(returns: pd.Series, trial_sharpes: list[float]) -> float:
    """DSR (Bailey & López de Prado 2014): PSR against the Sharpe ratio you
    would expect from the best of ``len(trial_sharpes)`` unskilled trials.
    Sharpe ratios must be per-period (same frequency as ``returns``)."""
    n_trials = len(trial_sharpes)
    if n_trials < 2:
        return probabilistic_sharpe(returns)
    var_sr = float(np.var(trial_sharpes, ddof=1))
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    sr0 = np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    return probabilistic_sharpe(returns, sr0)


def signal_stats(pred: pd.Series, target: pd.Series, fwd_ret: pd.Series) -> dict:
    """Prediction quality independent of any trading rule."""
    from sklearn.metrics import roc_auc_score

    df = pd.DataFrame({"p": pred, "y": target, "r": fwd_ret}).dropna()
    monthly = df.groupby(df.index.strftime("%Y-%m")).apply(
        lambda g: g["p"].corr(g["y"], method="spearman"), include_groups=False
    ).dropna()
    up = (df["r"] > 0).astype(int)
    return {
        "ic_pearson": float(df["p"].corr(df["y"])),
        "ic_spearman": float(df["p"].corr(df["y"], method="spearman")),
        "ic_monthly_mean": float(monthly.mean()),
        "ic_monthly_tstat": float(monthly.mean() / monthly.std() * np.sqrt(len(monthly))),
        "ic_months_positive": float((monthly > 0).mean()),
        "auc_direction": float(roc_auc_score(up, df["p"])) if up.nunique() == 2 else float("nan"),
        "hit_rate_direction": float((np.sign(df["p"]) == np.sign(df["r"])).mean()),
        "n": int(len(df)),
    }
