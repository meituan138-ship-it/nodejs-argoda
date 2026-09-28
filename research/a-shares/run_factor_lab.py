"""Factor lab with a sealed hold-out — the protocol is fixed here, before any
candidate factor has been evaluated.

    python run_factor_lab.py

Periods:
  mining     2013-01 → 2020-12  factors are tested here
  selection  2021-01 → 2022-12  survivors must hold up here
  hold-out   2023-01 → 2026-09  opened ONCE, at the very end

A candidate is kept only if ALL of:
  1. mining: |t| of the weekly rank IC >= 3.5 (≈ Bonferroni for ~30 tests)
     and the IC has the same sign in >= 75% of the mining years;
  2. selection: same sign as in mining and t >= 2.0;
  3. not redundant: mean |cross-sectional rank correlation| with every
     existing base feature < 0.7;
  4. no survivorship leak: the "value is missing" indicator must not
     predict returns (|t| < 3 in mining) — Sina data coverage must not
     depend on a stock's future.
Then the LightGBM model is retrained (same settings as run_ashare.py) with
the kept factors added, only for the hold-out quarters, and compared with
the existing baseline predictions on the hold-out: rank IC, top-50 net
return, and the 10k CNY / 8-stock weekly account. Reported whatever it shows.
"""
from __future__ import annotations

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

RES = Path(__file__).resolve().parent / "results"
MINE = ("2013-01-01", "2021-01-01")
SELECT = ("2021-01-01", "2023-01-01")
HOLD = ("2023-01-01", "2027-01-01")


def weekly_ic(df: pd.DataFrame, col: str) -> pd.Series:
    d = df[["date", col, "target"]].dropna()
    return d.groupby("date").apply(lambda g: g[col].corr(g["target"], method="spearman"), include_groups=False).dropna()


def t_of(s: pd.Series) -> float:
    return float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else 0.0


def main() -> None:
    stocks = load_all()
    X, feats, ex = build(stocks)
    X["lgbm_base"] = pd.read_parquet(RES / "ashare_preds.parquet")["lgbm"].to_numpy()
    days = ex["days"]
    mf = load_mf(list(stocks))
    print(f"money-flow coverage: {len(mf)}/{len(stocks)} stocks", flush=True)
    o, c, h = ex["open"], ex["close"], ex["high"]
    l = wide(stocks, "low").reindex(days)
    vol = wide(stocks, "volume").reindex(days)
    rc = wide(stocks, "raw_close").reindex(days)
    C = build_candidates(o, h, l, c, vol, rc, mf)

    keys = pd.MultiIndex.from_frame(X[["date", "code"]])
    for name, (frame, _) in C.items():
        X[name] = frame.reindex(pd.DatetimeIndex(X["date"].unique())).stack(future_stack=True).reindex(keys).to_numpy()

    base_raw = [f for f in feats if not f.startswith("xs_") and f not in ("mkt_ret_5", "mkt_ret_20", "mkt_vol_20", "breadth_20")]
    report = {}
    in_mine = (X["date"] >= MINE[0]) & (X["date"] < MINE[1])
    in_sel = (X["date"] >= SELECT[0]) & (X["date"] < SELECT[1])
    M = X[in_mine]
    sample_dates = sorted(M["date"].unique())[::8]  # every 8th week is enough for correlation estimates
    Mc = M[M["date"].isin(sample_dates)]
    for name, (_, why) in C.items():
        ic_m = weekly_ic(M, name)
        yearly = ic_m.groupby(ic_m.index.year).mean()
        sign = np.sign(ic_m.mean())
        same_sign = float((np.sign(yearly) == sign).mean())
        ic_s = weekly_ic(X[in_sel], name)
        miss = M[[name, "target", "date"]].assign(miss=M[name].isna().astype(float))
        ic_miss = weekly_ic(miss, "miss") if miss["miss"].between(0.001, 0.999).mean() > 0 and miss["miss"].mean() > 0.001 else pd.Series(dtype=float)
        rank_corr = {b: Mc.groupby("date").apply(lambda g: g[name].corr(g[b], method="spearman"), include_groups=False).mean()
                     for b in base_raw}
        worst = max(rank_corr, key=lambda b: abs(rank_corr[b]) if np.isfinite(rank_corr[b]) else 0)
        r = {
            "why": why, "coverage_mining": float(M[name].notna().mean()),
            "ic_mine": float(ic_m.mean()), "t_mine": t_of(ic_m), "same_sign_years": same_sign,
            "ic_select": float(ic_s.mean()), "t_select": t_of(ic_s) * (1 if sign >= 0 else -1) * (1 if sign != 0 else 0),
            "max_corr_existing": float(abs(rank_corr[worst])), "most_similar": worst,
            "t_missing_indicator": t_of(ic_miss) if len(ic_miss) else 0.0,
        }
        r["pass_mine"] = abs(r["t_mine"]) >= 3.5 and same_sign >= 0.75
        r["pass_select"] = r["t_select"] >= 2.0
        r["pass_redundancy"] = r["max_corr_existing"] < 0.7
        r["pass_leak"] = abs(r["t_missing_indicator"]) < 3
        r["kept"] = all((r["pass_mine"], r["pass_select"], r["pass_redundancy"], r["pass_leak"]))
        report[name] = r
        print(f"{name:24s} IC挖掘 {r['ic_mine']:+.4f} (t {r['t_mine']:+.1f}, 同号年份 {same_sign:.0%}) | "
              f"筛选 {r['ic_select']:+.4f} (t {r['t_select']:+.1f}) | 相关 {r['max_corr_existing']:.2f}({worst}) | "
              f"缺失t {r['t_missing_indicator']:+.1f} | {'✅ 保留' if r['kept'] else '✗'}", flush=True)

    kept = [n for n, r in report.items() if r["kept"]]
    print("kept:", kept, flush=True)
    out = {"protocol": __doc__, "candidates": report, "kept": kept}
    if kept:
        for n in kept:
            X[f"xs_{n}"] = X.groupby("date")[n].rank(pct=True)
        new_feats = feats + kept + [f"xs_{n}" for n in kept]
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
            p, _ = fit_predict(Xtr, ytr, Xtr.iloc[:0], ytr[:0], X.iloc[te][new_feats],
                               params={"min_data_in_leaf": 2000}, seeds=(0, 1), n_trees=300)
            pred[te] = p
            print(f"  retrain {a:%Y-%m} train={len(tr)}", flush=True)
        X["lgbm_new"] = pred
        # ---- the hold-out is opened here, once
        H = X[(X["date"] >= HOLD[0]) & X["lgbm_new"].notna()].copy()
        cmp = {}
        raw_open = wide(stocks, "raw_open").reindex(days)
        for col in ("lgbm_base", "lgbm_new"):
            ic = weekly_ic(H, col)
            bt = RA.backtest(H, col, ex, 50)
            H2 = H.rename(columns={col: "lgbm"}) if col != "lgbm" else H
            small = SC.summary(SC.simulate(H2[["date", "code", "lgbm"]], ex, raw_open, n=8, min_comm=0.0))
            cmp[col] = {"ic": float(ic.mean()), "ic_t": t_of(ic), "top50": RA.stats(bt["net"]),
                        "small_8_weekly_mianwu": {k: v for k, v in small.items() if k != "by_year"},
                        "small_by_year": small["by_year"]}
            print(col, json.dumps(cmp[col], ensure_ascii=False), flush=True)
        out["holdout"] = cmp
    RES.mkdir(exist_ok=True)
    (RES / "factor_lab.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
