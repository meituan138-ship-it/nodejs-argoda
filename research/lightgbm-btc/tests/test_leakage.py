"""Guards against the failure modes that invalidate most published BTC
"prediction" results: look-ahead features, labels that peek past their
horizon, CV folds that overlap, and a harness that cannot tell signal from
noise. Runs on synthetic data, no network needed."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lgbm_btc.backtest import funding_per_bar, positions_from_scores, simulate
from lgbm_btc.cv import walk_forward_folds
from lgbm_btc.features import bar_volatility, build_features
from lgbm_btc.labels import forward_log_return, regression_target, triple_barrier
from lgbm_btc.pipeline import Experiment, MarketDataset, run_experiment, summarize


def _synthetic_bars(n: int, seed: int, start: str = "2019-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    vol = 0.006 * np.exp(np.cumsum(rng.normal(0, 0.02, n)) * 0.2)
    r = rng.standard_t(4, n) * vol / np.sqrt(2)
    close = 10000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, vol, (2, n)))
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    qv = np.exp(rng.normal(17, 0.5, n))
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": qv / close, "quote_volume": qv, "trades": np.round(qv / 2000),
        "taker_buy_base": qv / close * rng.uniform(0.3, 0.7, n),
        "taker_buy_quote": qv * rng.uniform(0.3, 0.7, n),
    }, index=idx)


@pytest.fixture(scope="module")
def raw():
    n = 24 * 400
    btc = _synthetic_bars(n, 1)
    eth = _synthetic_bars(n, 2)
    idx = btc.index
    funding = pd.Series(np.random.default_rng(3).normal(1e-4, 1e-4, n // 8),
                        index=idx[::8][: n // 8], name="funding_rate")
    premium = pd.Series(np.random.default_rng(4).normal(0, 3e-4, n), index=idx, name="premium")
    return {"btc": btc, "eth": eth, "funding": funding, "premium": premium}


def test_features_are_point_in_time(raw):
    full = build_features(raw["btc"], raw["eth"], raw["funding"], raw["premium"])
    for cut in (24 * 200, 24 * 300 + 7):
        t = raw["btc"].index[cut]
        # bars are indexed by open time; the decision is made at the close (t + 1h),
        # when a funding settlement stamped t + 1h is already public
        part = build_features(
            raw["btc"].loc[:t], raw["eth"].loc[:t],
            raw["funding"].loc[:t + pd.Timedelta(hours=1)], raw["premium"].loc[:t],
        )
        a, b = full.loc[:t], part
        assert list(a.columns) == list(b.columns)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=1e-4, atol=1e-6, equal_nan=True,
                                   err_msg=f"feature values up to {t} changed when future bars were removed")


def test_funding_only_known_after_settlement(raw):
    X = build_features(raw["btc"], raw["eth"], raw["funding"], raw["premium"])
    f = raw["funding"]
    settle = f.index[50]
    # the bar that closes exactly at the settlement sees it, the bar before does not
    assert X.loc[settle - pd.Timedelta(hours=1), "funding_last"] == pytest.approx(f.iloc[50], rel=1e-5)
    assert X.loc[settle - pd.Timedelta(hours=2), "funding_last"] == pytest.approx(f.iloc[49], rel=1e-5)


def test_labels_only_use_their_horizon(raw):
    close, h = raw["btc"]["close"].copy(), 24
    sig = bar_volatility(close)
    base = regression_target(close, sig, h)
    t = 5000
    bumped = close.copy()
    bumped.iloc[t + h + 1:] *= 1.5  # beyond the horizon: must not matter
    assert regression_target(bumped, sig, h).iloc[t] == pytest.approx(base.iloc[t])
    bumped.iloc[t + h] *= 1.5  # at the horizon: must matter
    assert regression_target(bumped, sig, h).iloc[t] != pytest.approx(base.iloc[t])

    tb = triple_barrier(close, raw["btc"]["high"], raw["btc"]["low"], sig, h)
    assert tb["bars"].dropna().between(1, h).all()
    assert set(tb["y"].dropna().unique()) <= {0.0, 1.0}


def test_walk_forward_folds_are_purged(raw):
    idx = raw["btc"].index
    h = 24
    folds = walk_forward_folds(idx, "2019-08-01", "2020-02-01", h, "MS",
                               train_start="2019-01-10", min_train_bars=24 * 60)
    assert len(folds) >= 5
    for f in folds:
        assert f.train.max() + h < f.valid.min(), "train labels overlap the validation slice"
        assert f.valid.max() + h < f.test.min(), "validation labels overlap the test block"
        assert idx[f.test].min() >= f.test_start and idx[f.test].max() < f.test_end


def test_backtest_conventions():
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    close = pd.Series([100, 110, 99, 99, 100, 100.0], index=idx)
    pos = pd.Series([1, 1, 0, 0, -1, 0.0], index=idx)
    bt = simulate(pos, close, fee_bps=10)
    # pos[t] earns close[t+1]/close[t]-1
    assert bt["gross"].iloc[0] == pytest.approx(0.10)
    assert bt["gross"].iloc[1] == pytest.approx(-0.10)
    assert bt["turnover"].sum() == pytest.approx(4.0)
    assert bt["cost"].sum() == pytest.approx(4.0 * 1e-3)
    # funding at 08:00 is paid by the position chosen at the close of the 06:00 bar
    f = pd.Series([0.001], index=[pd.Timestamp("2024-01-01 08:00", tz="UTC")])
    idx2 = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    fb = funding_per_bar(f, idx2)
    assert fb[pd.Timestamp("2024-01-01 06:00", tz="UTC")] == pytest.approx(0.001)
    assert fb.sum() == pytest.approx(0.001)
    s = pd.Series([1.0, 1, 1, -1, -1, -1], index=idx)
    assert positions_from_scores(s, 3).tolist() == pytest.approx([1, 1, 1, 1 / 3, -1 / 3, -1])


def _small_ds(raw):
    return MarketDataset.from_raw(raw)


def test_real_features_carry_no_signal_on_random_walk(raw):
    """On a pure random walk every feature is noise, so an honest harness must report ~0 IC."""
    ds = _small_ds(raw)
    kw = dict(horizon=1, train_start="2019-02-01", test_start="2019-11-01", test_end="2020-02-01",
              seeds=(0,), min_train_bars=24 * 60)
    res = summarize(run_experiment(Experiment("rw", **kw), ds, verbose=False))
    assert abs(res["ic_spearman"]) < 0.07


def test_negative_and_positive_controls(raw):
    """Shuffled labels must show no skill; a weak leaked signal (corr ~0.1) must be found."""
    ds = _small_ds(raw)
    # h=1 keeps the test samples ~independent, so the IC noise is ~1/sqrt(n) ≈ 0.02
    kw = dict(horizon=1, train_start="2019-02-01", test_start="2019-11-01", test_end="2020-02-01",
              seeds=(0,), min_train_bars=24 * 60)
    neg = summarize(run_experiment(Experiment("neg", control="shuffle", **kw), ds, verbose=False))
    pos = summarize(run_experiment(Experiment("pos", control="leak", leak_noise=3.0, **kw), ds, verbose=False))
    assert abs(neg["ic_spearman"]) < 0.07
    assert pos["ic_spearman"] > 0.15


def test_forward_return_alignment(raw):
    close = raw["btc"]["close"]
    fr = forward_log_return(close, 3)
    assert fr.iloc[10] == pytest.approx(np.log(close.iloc[13] / close.iloc[10]))
    assert fr.iloc[-3:].isna().all()
