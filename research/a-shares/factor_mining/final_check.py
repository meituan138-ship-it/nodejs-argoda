"""The ONLY script that opens the sealed hold-out (2023-01 → latest).

    python final_check.py              # asks for confirmation, runs once, logs it
    python final_check.py --force      # run again (the log keeps every opening)

What it does:
  1. re-gates every accepted factor at the FINAL trial count (factors accepted
     early, when the bar was lower, must clear today's bar);
  2. verifies each factor's formula still matches the hash in the ledger;
  3. retrains the weekly LightGBM walk-forward (quarterly, 14-day embargo) with
     the base features + survivors, starting 2023-01;
  4. compares baseline vs +factors on the hold-out: rank IC, top-50 equal-weight
     net of costs, 10k-CNY 8-stock account (免五 commission).
Every opening is appended to results/holdout_openings.log. If you keep opening
the hold-out and tweaking factors, it stops being a hold-out — do it rarely.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "lightgbm-btc"))
from fm import ledger  # noqa: E402
from fm.context import CACHE, Context  # noqa: E402
from fm.evaluate import HOLDOUT_START, required_t  # noqa: E402
from fm.registry import load_factor_file  # noqa: E402

OPEN_LOG = ledger.RESULTS / "holdout_openings.log"


def weekly_ic(df, col):
    g = df.dropna(subset=[col, "target"]).groupby("date")
    return g.apply(lambda x: x[col].corr(x["target"], method="spearman"), include_groups=False).dropna()


def t_of(s):
    return float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 2 else 0.0


def survivors() -> list:
    df = ledger.read()
    n = ledger.n_trials()
    bar = required_t(n)
    kept = df[(df["kind"] == "factor") & (df["kept"].astype(str) == "True")].drop_duplicates("name", keep="last")
    out = []
    print(f"final trial count N={n} -> bar |t_mine| >= {bar:.2f}")
    for _, r in kept.iterrows():
        if abs(float(r["t_mine"])) < bar:
            print(f"  drop {r['name']}: t_mine {float(r['t_mine']):+.2f} < final bar {bar:.2f}")
            continue
        facs = {f.name: f for f in load_factor_file(r["file"])} if Path(r["file"]).exists() else {}
        f = facs.get(r["name"])
        if f is None or f.code_hash != r["code_hash"]:
            print(f"  drop {r['name']}: formula missing or changed since it was accepted")
            continue
        out.append(f)
    print(f"survivors: {[f.name for f in out]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    if OPEN_LOG.exists() and not a.force:
        sys.exit(f"hold-out already opened (see {OPEN_LOG}). Use --force to open again — every opening is logged.")
    facs = survivors()
    if not facs:
        sys.exit("no factor survives the final bar; nothing to test")
    if not a.yes and input("open the sealed hold-out now? type YES: ").strip() != "YES":
        sys.exit("aborted")

    from ash.data import load_all
    from ash.panel import build, wide
    from lgbm_btc.model import fit_predict
    import run_ashare as RA
    import run_small_capital as SC

    stocks = load_all()
    X, feats, ex = build(stocks)
    days = ex["days"]
    try:  # baseline walk-forward predictions saved by prepare_data.py (from run_ashare.py)
        X["lgbm_base"] = pd.read_parquet(CACHE / "panel.parquet", columns=["lgbm_base"])["lgbm_base"].to_numpy()
    except Exception:  # noqa: BLE001 — retrain the baseline below instead
        X["lgbm_base"] = np.nan
    num = X.select_dtypes("float64").columns
    X[num] = X[num].astype("float32")
    ctx = Context()
    dec = pd.DatetimeIndex(X["date"].unique())
    keys = pd.MultiIndex.from_frame(X[["date", "code"]])
    names = []
    for f in facs:
        F = f.func(ctx).replace([np.inf, -np.inf], np.nan)
        X[f.name] = F.reindex(dec).stack(future_stack=True).reindex(keys).to_numpy().astype("float32")
        X[f"xs_{f.name}"] = X.groupby("date")[f.name].rank(pct=True).astype("float32")
        names += [f.name, f"xs_{f.name}"]
        del F
        gc.collect()
    ok = X["target"].notna().to_numpy()
    d = X["date"]
    qs = pd.date_range(HOLDOUT_START, d.max() + pd.offsets.QuarterBegin(1), freq="QS")
    for col, fl in (("lgbm_base", feats), ("lgbm_new", feats + names)):
        if col == "lgbm_base" and X[col].notna().any():
            continue  # baseline predictions from run_ashare.py already cached
        pred = np.full(len(X), np.nan)
        for s, e in zip(qs[:-1], qs[1:]):
            tr = np.where(ok & (d >= "2013-01-01").to_numpy() & (d < s - pd.Timedelta(days=14)).to_numpy())[0]
            te = np.where(((d >= s) & (d < e)).to_numpy())[0]
            if not len(te):
                continue
            Xtr, ytr = X.iloc[tr][fl], X["target"].to_numpy()[tr]
            pred[te], _ = fit_predict(Xtr, ytr, Xtr.iloc[:0], ytr[:0], X.iloc[te][fl],
                                      params={"min_data_in_leaf": 2000}, seeds=(0, 1), n_trees=300)
            del Xtr
            gc.collect()
            print(f"  {col} retrain {s:%Y-%m}", flush=True)
        X[col] = pred
    H = X[(X["date"] >= HOLDOUT_START) & X["lgbm_new"].notna()].copy()
    raw_open = wide(stocks, "raw_open").reindex(days)
    out = {"time": dt.datetime.now().isoformat(timespec="seconds"), "n_trials": ledger.n_trials(),
           "factors": [f.name for f in facs]}
    for col in ("lgbm_base", "lgbm_new"):
        ic = weekly_ic(H, col)
        bt = RA.backtest(H, col, ex, 50)
        small = SC.summary(SC.simulate(H[["date", "code", col]].rename(columns={col: "lgbm"}), ex, raw_open, n=8, min_comm=0.0))
        out[col] = {"ic": float(ic.mean()), "ic_t": t_of(ic),
                    "ic_by_year": {int(y): float(ic[ic.index.year == y].mean()) for y in sorted(set(ic.index.year))},
                    "top50": RA.stats(bt["net"]),
                    "top50_by_year": {int(y): float((1 + bt["net"][bt.index.year == y]).prod() - 1) for y in sorted(set(bt.index.year))},
                    "small8": {k: v for k, v in small.items() if k != "by_year"}, "small8_by_year": small["by_year"]}
        print(col, json.dumps(out[col], ensure_ascii=False, default=float), flush=True)
    b, n = out["lgbm_base"], out["lgbm_new"]
    verdict = "IMPROVES" if (n["ic"] > b["ic"] and n["top50"].get("sharpe", 0) > b["top50"].get("sharpe", 0)) else "NO IMPROVEMENT"
    out["verdict"] = verdict
    print(f"\nverdict: {verdict}  (IC {b['ic']:.4f} -> {n['ic']:.4f})")
    ledger.RESULTS.mkdir(exist_ok=True)
    (ledger.RESULTS / f"final_check_{dt.datetime.now():%Y%m%d_%H%M}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False, default=float))
    with OPEN_LOG.open("a") as fh:
        fh.write(json.dumps({"time": out["time"], "n_trials": out["n_trials"], "factors": out["factors"],
                             "verdict": verdict}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
