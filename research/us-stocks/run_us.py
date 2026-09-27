"""S&P 500 weekly cross-sectional LightGBM, walk-forward 2015-01 → latest.

    python run_us.py

Pre-declared (same recipe as the crypto multi-coin study):
- universe = point-in-time S&P 500 members with Yahoo price history;
- target = next-week market-residual return / (21d vol·√5);
- LightGBM 300 trees, lr 0.02, 31 leaves, min_data_in_leaf 2000, 2 seeds;
  retrained on the first week of every month on all data up to two weeks
  before (purge of the 5-day label + 1 week embargo); Ridge as baseline;
- portfolio: every week long the top 10% and short the bottom 10%
  (0.5 / 0.5 of capital), or long-only top 10%; cost 5 bp per side
  (large-cap commissions + half-spread), short borrow 0.5%/yr on the
  short notional (general collateral);
- baselines: 12-1 momentum, 1-week reversal, SPY buy & hold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lightgbm-btc"))
from lgbm_btc.model import fit_predict, fit_predict_ridge  # noqa: E402
from usxs.data import load_prices, membership, sectors  # noqa: E402
from usxs.panel import H, build  # noqa: E402

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
TEST_START = "2015-01-01"
FEE_BPS, BORROW = 5.0, 0.005
PARAMS = {"min_data_in_leaf": 2000}


def walk_forward(X: pd.DataFrame, feats: list[str], model: str) -> np.ndarray:
    pred = np.full(len(X), np.nan)
    dates = X["date"]
    ok = X["target"].notna().to_numpy()
    months = pd.date_range(TEST_START, dates.max() + pd.offsets.MonthBegin(1), freq="MS")
    for a, b in zip(months[:-1], months[1:]):
        cutoff = a - pd.Timedelta(days=14)
        tr = np.where(ok & (dates < cutoff).to_numpy() & (dates >= "2006-01-01").to_numpy())[0]
        te = np.where(((dates >= a) & (dates < b)).to_numpy())[0]
        if not len(te):
            continue
        Xtr, ytr = X.iloc[tr][feats], X["target"].to_numpy()[tr]
        if model == "ridge":
            p, _ = fit_predict_ridge(Xtr, ytr, Xtr.iloc[:0], ytr[:0], X.iloc[te][feats])
        else:
            p, _ = fit_predict(Xtr, ytr, Xtr.iloc[:0], ytr[:0], X.iloc[te][feats],
                               params=PARAMS, seeds=(0, 1), n_trees=300)
        pred[te] = p
        if a.month == 1:
            print(f"  [{model}] {a:%Y-%m} train={len(tr)} test={len(te)}", flush=True)
    return pred


def portfolio(D: pd.DataFrame, col: str, q: float = 0.1, long_only: bool = False, fee_bps: float = FEE_BPS):
    rows, prev = [], pd.Series(dtype=float)
    for t, g in D.groupby("date"):
        g = g.dropna(subset=[col, "fwd"]).set_index("ticker")
        k = max(1, int(round(q * len(g))))
        o = g[col].sort_values()
        w = pd.Series(0.0, index=g.index)
        if long_only:
            w[o.index[-k:]] = 1 / k
        else:
            w[o.index[-k:]] = 0.5 / k
            w[o.index[:k]] = -0.5 / k
        simple = np.exp(g["fwd"]) - 1
        gross = float((w * simple).sum())
        turn = float(w.sub(prev, fill_value=0).abs().sum())
        borrow = 0.0 if long_only else 0.5 * BORROW / 52
        prev = w
        rows.append({"date": t, "gross": gross, "net": gross - turn * fee_bps * 1e-4 - borrow, "turnover": turn})
    return pd.DataFrame(rows).set_index("date")


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 52
    return {"cagr": float(eq.iloc[-1] ** (1 / yrs) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(52)),
            "max_dd": float((eq / eq.cummax() - 1).min()), "win_weeks": float((r > 0).mean())}


def main() -> None:
    mem = membership("2006-01-01")
    tickers = list(mem.columns[mem.loc["2006":].any()])
    prices = load_prices(tickers + ["SPY"])
    spy = prices.pop("SPY")
    X, feats, member_weeks = build(prices, mem, sectors())
    print(f"rows={len(X)} features={len(feats)} tickers with data={len(prices)}/{len(tickers)}")

    cache = RESULTS / "us_preds.parquet"
    if cache.exists():
        P = pd.read_parquet(cache)
    else:
        P = pd.DataFrame({"lgbm": walk_forward(X, feats, "lgbm"), "ridge": walk_forward(X, feats, "ridge")})
        RESULTS.mkdir(exist_ok=True)
        P.to_parquet(cache)
    D = pd.concat([X[["date", "ticker", "fwd", "target", "mom_12_1", "ret_5"]].reset_index(drop=True), P], axis=1)
    D = D[D["lgbm"].notna()].copy()
    D["mom_12_1_sig"] = D["mom_12_1"]
    D["rev_1w"] = -D["ret_5"]

    # survivorship check: share of point-in-time member-weeks we actually have prices for
    mem_w = mem.reindex(sorted(D["date"].unique()), method="ffill")
    coverage = float(D.groupby("date").size().sum() / mem_w.sum().sum())

    out = {"test_weeks": int(D["date"].nunique()), "stocks_per_week": float(D.groupby("date").size().mean()),
           "member_coverage": coverage, "tickers_with_data": len(prices), "tickers_ever_member": len(tickers)}
    ic_all, ls_all, lo_all, yearly = {}, {}, {}, {}
    for col in ("lgbm", "ridge", "mom_12_1_sig", "rev_1w"):
        ic = D.groupby("date").apply(lambda g: g[col].corr(g["target"], method="spearman"), include_groups=False)
        ic_all[col] = {"mean": float(ic.mean()), "tstat": float(ic.mean() / ic.std() * np.sqrt(len(ic)))}
        ls = portfolio(D, col)
        s = stats(ls["net"])
        s["sharpe_gross"] = stats(ls["gross"])["sharpe"]
        s["sharpe@10bps"] = stats(portfolio(D, col, fee_bps=10)["net"])["sharpe"]
        s["turnover_per_week"] = float(ls["turnover"].mean())
        ls_all[col] = s
        yearly[col] = {int(y): {"ic": float(ic[ic.index.year == y].mean()),
                                "ls_return": float((1 + ls.loc[ls.index.year == y, "net"]).prod() - 1)}
                       for y in sorted(set(ls.index.year))}
        lo = portfolio(D, col, long_only=True)
        lo_all[col] = stats(lo["net"])
        if col == "lgbm":
            lo_net, ls_net = lo["net"], ls["net"]
    # benchmarks on the same weekly dates
    spy_c = spy["adjclose"]
    dts = sorted(D["date"].unique())
    idx = spy_c.index.get_indexer(dts)
    spy_w = pd.Series(spy_c.to_numpy()[np.minimum(idx + H, len(spy_c) - 1)] / spy_c.to_numpy()[idx] - 1, index=dts)
    ew_w = D.groupby("date")["fwd"].apply(lambda s: float((np.exp(s) - 1).mean()))
    out.update({"ic": ic_all, "long_short": ls_all, "long_only_top10": lo_all,
                "spy": stats(spy_w), "equal_weight_universe": stats(ew_w), "yearly": yearly})
    for name, r in (("lgbm_ls", ls_net), ("lgbm_lo", lo_net), ("spy", spy_w), ("ew", ew_w)):
        out.setdefault("by_year_return", {})[name] = {int(y): float((1 + r[r.index.year == y]).prod() - 1)
                                                     for y in sorted(set(r.index.year))}
    ex = (lo_net - spy_w.reindex(lo_net.index)).dropna()
    out["long_only_vs_spy_alpha"] = {"ann": float(ex.mean() * 52), "tstat": float(ex.mean() / ex.std() * np.sqrt(len(ex)))}
    (RESULTS / "us_summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
