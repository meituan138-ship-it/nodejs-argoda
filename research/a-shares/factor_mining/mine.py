"""Evaluate factor files against the acceptance gates and record every trial.

    python mine.py factors/my_batch.py            # evaluate every @factor in the file
    python mine.py factors/my_batch.py --only foo # just one
    python mine.py --status                       # ledger summary + current bar

Never reads data from 2023-01-01 on (sealed hold-out, see final_check.py).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fm import ledger  # noqa: E402
from fm.context import CACHE, Context  # noqa: E402
from fm.evaluate import RULES, Evaluator, required_t  # noqa: E402
from fm.registry import load_factor_file  # noqa: E402

SHOW = ["t_mine", "t_required", "ic_mine", "same_sign_years", "t_select", "max_corr", "most_similar", "t_missing",
        "coverage", "leak_mismatch"]


def status() -> None:
    df = ledger.read()
    n = ledger.n_trials()
    print(f"trials so far: {n}  ->  next factor needs |t_mine| >= {required_t(n + 1):.2f}")
    if df.empty:
        return
    f = df[df["kind"] == "factor"]
    print(f"factor evaluations: {len(f)}, distinct formulas: {f['code_hash'].nunique()}, "
          f"GP formulas scored: {int(df.loc[df['kind'] == 'gp_batch', 'n'].sum())}")
    kept = f[f["kept"].astype(str) == "True"].drop_duplicates("name", keep="last")
    print(f"accepted ({len(kept)}):")
    for _, r in kept.iterrows():
        print(f"  {r['name']:<32} t={float(r['t_mine']):+.2f} sel_t={float(r['t_select']):+.2f} "
              f"corr={float(r['max_corr']):.2f}  [{r['source']}]")
    fails = f[f["kept"].astype(str) != "True"]["reason"].str.replace(r"failed: ", "", regex=True).str.split(",").explode()
    print("most common failures:", fails.value_counts().head(6).to_dict())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--only", default=None)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status or not a.file:
        status()
        return
    facs = load_factor_file(a.file)
    if a.only:
        facs = [f for f in facs if f.name == a.only]
    if not facs:
        sys.exit("no @factor found")
    ctx = Context()
    ev = Evaluator(ctx, pd.read_parquet(CACHE / "weekly.parquet"), pd.read_parquet(CACHE / "base_sample.parquet"),
                   ledger.accepted_samples())
    taken = set(ledger.accepted_samples())
    for fac in facs:
        prev = ledger.seen(fac.code_hash)
        if prev is not None:
            print(f"\n[{fac.name}] identical formula already evaluated as {prev['name']!r} on {prev['time']}: "
                  f"{prev['reason']} (not re-counted, not re-tested)")
            continue
        if fac.name in taken:
            print(f"\n[{fac.name}] name already used by an accepted factor with a different formula — rename it")
            continue
        t0 = time.time()
        r = ev.evaluate(fac, ledger.n_trials(extra=1))
        sample = r.pop("_sample_values", None)
        ledger.append(r)
        print(f"\n[{fac.name}] {'KEPT' if r['kept'] else 'rejected'} — {r['reason']}  ({time.time() - t0:.0f}s)")
        for k in SHOW:
            if k in r:
                v = r[k]
                print(f"    {k:<16} {v:.4f}" if isinstance(v, float) else f"    {k:<16} {v}")
        if r["kept"]:
            ledger.save_accepted(fac.name, sample, fac.code)
            ev.accepted[fac.name] = sample
            taken.add(fac.name)
    print(f"\nrules: {RULES}")
    status()


if __name__ == "__main__":
    main()
