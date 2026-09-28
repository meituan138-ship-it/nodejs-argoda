"""Genetic-programming formula search (data-driven, no "imagination" needed).

    python gp_search.py --pop 60 --gens 8 --seed 1       # one run, ~1-3 h
    python mine.py factors/gp_<run>.py                   # then the normal gates

Formulas are random trees of fields and causal operators, evolved for mining-
period rank-IC t-stat. EVERY formula scored here is added to the ledger's trial
count, so a long GP run raises the |t| bar for everything after it — that is
the price of searching wide. Only data before 2023 is loaded. The selection
period (2021-22) is not used for fitness, so mine.py's selection gate stays an
honest out-of-sample check for GP output.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fm import ledger, ops  # noqa: E402
from fm.context import CACHE, Context  # noqa: E402
from fm.evaluate import HOLDOUT_START, MINE, Evaluator  # noqa: E402

LEAVES = ["d.close", "d.open", "d.high", "d.low", "d.volume", "d.amount", "d.turnover", "d.mf_net_ratio",
          "d.ret", "safe_div(d.amount, d.volume)", "safe_div(d.close, d.open)", "safe_div(d.high, d.low)"]
WINDOWS = [3, 5, 10, 20, 60]
UNARY_TS = ["ts_mean", "ts_std", "delta", "ts_zscore", "ts_max", "ts_min", "ts_sum", "pct"]
UNARY = ["cs_rank", "abs", "neg"]
BINARY = ["add", "sub", "mul", "safe_div"]
BINARY_TS = ["ts_corr"]


def rand_tree(depth: int, rng: random.Random):
    if depth <= 0 or rng.random() < 0.25:
        return ("leaf", rng.choice(LEAVES))
    k = rng.random()
    if k < 0.45:
        return ("uts", rng.choice(UNARY_TS), rand_tree(depth - 1, rng), rng.choice(WINDOWS))
    if k < 0.6:
        return ("un", rng.choice(UNARY), rand_tree(depth - 1, rng))
    if k < 0.9:
        return ("bin", rng.choice(BINARY), rand_tree(depth - 1, rng), rand_tree(depth - 1, rng))
    return ("bts", "ts_corr", rand_tree(depth - 1, rng), rand_tree(depth - 1, rng), rng.choice([5, 10, 20]))


def code(t) -> str:
    kind = t[0]
    if kind == "leaf":
        return t[1]
    if kind == "uts":
        return f"{t[1]}({code(t[2])}, {t[3]})"
    if kind == "un":
        return {"cs_rank": f"cs_rank({code(t[2])})", "abs": f"({code(t[2])}).abs()", "neg": f"(-{code(t[2])})"}[t[1]]
    if kind == "bin":
        a, b = code(t[2]), code(t[3])
        return {"add": f"({a} + {b})", "sub": f"({a} - {b})", "mul": f"({a} * {b})", "safe_div": f"safe_div({a}, {b})"}[t[1]]
    return f"ts_corr({code(t[2])}, {code(t[3])}, {t[4]})"


def size(t) -> int:
    return 1 + sum(size(c) for c in t[1:] if isinstance(c, tuple))


def nodes(t, path=()):
    yield path, t
    for i, c in enumerate(t):
        if isinstance(c, tuple):
            yield from nodes(c, path + (i,))


def replace(t, path, new):
    if not path:
        return new
    lst = list(t)
    lst[path[0]] = replace(t[path[0]], path[1:], new)
    return tuple(lst)


def mutate(t, rng):
    path, _ = rng.choice(list(nodes(t)))
    return replace(t, path, rand_tree(2, rng))


def crossover(a, b, rng):
    pa, _ = rng.choice(list(nodes(a)))
    _, sb = rng.choice(list(nodes(b)))
    return replace(a, pa, sb)


class Scorer:
    def __init__(self):
        ctx = Context().truncate(HOLDOUT_START - pd.Timedelta(days=1))
        self.ctx = ctx
        weekly = pd.read_parquet(CACHE / "weekly.parquet")
        self.target = weekly[weekly["date"] < MINE[1]].pivot(index="date", columns="code", values="target")
        self.env = {k: getattr(ops, k) for k in dir(ops) if not k.startswith("_")}
        self.env["d"] = ctx

    def __call__(self, t) -> tuple[float, pd.Series | None]:
        try:
            with np.errstate(all="ignore"):
                F = eval(code(t), {"__builtins__": {}}, self.env)  # noqa: S307 — formula built from our own grammar
            if not isinstance(F, pd.DataFrame):
                return 0.0, None
            W = F.replace([np.inf, -np.inf], np.nan).reindex(index=self.target.index, columns=self.target.columns)
            if W.notna().to_numpy().mean() < 0.4 * self.target.notna().to_numpy().mean():
                return 0.0, None
            ic = Evaluator._rank_ic(W, self.target)
            ic = ic[ic.index >= MINE[0]]
            return Evaluator._t(ic), ic
        except Exception:  # noqa: BLE001
            return 0.0, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--top", type=int, default=10, help="formulas written out for mine.py")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    run = f"{time.strftime('%Y%m%d_%H%M')}_s{a.seed}"
    score = Scorer()
    seen: dict[str, tuple[float, pd.Series | None]] = {}

    def fit(t):
        c = code(t)
        if c not in seen:
            seen[c] = score(t)
        t_stat, _ = seen[c]
        yearly = seen[c][1]
        stab = 1.0
        if yearly is not None and len(yearly):
            y = yearly.groupby(yearly.index.year).mean()
            stab = float((np.sign(y) == np.sign(t_stat)).mean())
        return abs(t_stat) * stab - 0.05 * size(t)   # parsimony pressure: short formulas generalise better

    pop = [rand_tree(a.depth, rng) for _ in range(a.pop)]
    try:
        for g in range(a.gens):
            t0 = time.time()
            scored = sorted(((fit(t), t) for t in pop), key=lambda x: -x[0])
            print(f"gen {g}: best {scored[0][0]:.2f}  {code(scored[0][1])}   scored={len(seen)} ({time.time() - t0:.0f}s)",
                  flush=True)
            elite = [t for _, t in scored[: max(2, a.pop // 10)]]

            def pick():
                return max(rng.sample(scored, 4), key=lambda x: x[0])[1]
            nxt = list(elite)
            while len(nxt) < a.pop:
                r = rng.random()
                child = crossover(pick(), pick(), rng) if r < 0.6 else mutate(pick(), rng) if r < 0.9 else rand_tree(a.depth, rng)
                if size(child) <= 15:
                    nxt.append(child)
            pop = nxt
    finally:
        # every formula scored is a trial, even if the run is interrupted
        ledger.append({"kind": "gp_batch", "name": f"gp:{run}", "n": len(seen), "code_hash": "", "kept": "",
                       "reason": f"GP run scored {len(seen)} formulas", "source": f"gp:{run}"})
    # write the best distinct formulas (low mutual correlation of their IC series)
    ranked = sorted(((abs(v[0]), c, v) for c, v in seen.items() if v[1] is not None), reverse=True)
    chosen: list[tuple[str, float, pd.Series]] = []
    for _, c, (t_stat, ic) in ranked:
        if all(abs(ic.corr(o)) < 0.7 for _, _, o in chosen):
            chosen.append((c, t_stat, ic))
        if len(chosen) >= a.top:
            break
    out = HERE / "factors" / f"gp_{run}.py"
    lines = [f'"""GP run {run}: pop={a.pop} gens={a.gens} seed={a.seed}, {len(seen)} formulas scored.',
             "Fitness used mining-period IC only. Before submitting, replace each rationale with a real",
             'economic explanation if you can find one; unexplained formulas are the most likely to be noise."""',
             "from fm.ops import *  # noqa: F401,F403", "from fm.registry import factor", ""]
    for i, (c, t_stat, _) in enumerate(chosen):
        name = f"gp_{run}_{i}".replace("-", "_")
        lines += ["", f'@factor("{name}",',
                  f'        rationale="GP 自动生成（mining t={t_stat:+.2f}），暂无经济学解释",',
                  f'        source="gp:{run}")', f"def {name}(d):", f"    return {c}", ""]
    out.write_text("\n".join(lines))
    print(f"\nwrote {len(chosen)} formulas to {out}\nnext: python mine.py {out.relative_to(HERE)}")


if __name__ == "__main__":
    main()
