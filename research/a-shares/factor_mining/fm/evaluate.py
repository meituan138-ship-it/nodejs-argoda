"""Acceptance tests for one factor. The rules live here and in RULES — the
mining agent must not edit this file.

Periods (the hold-out is never touched here):
  mining     2013-01-01 .. 2020-12-31
  selection  2021-01-01 .. 2022-12-31
  hold-out   2023-01-01 .. end          -> only final_check.py, once
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

from .context import Context

MINE = (pd.Timestamp("2013-01-01"), pd.Timestamp("2021-01-01"))
SELECT = (pd.Timestamp("2021-01-01"), pd.Timestamp("2023-01-01"))
HOLDOUT_START = pd.Timestamp("2023-01-01")

RULES = {
    "min_t_mine_floor": 3.5,       # the bar rises with the number of factors ever tried (Bonferroni)
    "min_same_sign_years": 0.75,   # IC sign stable in >= 75% of mining years
    "min_t_select": 2.0,           # same sign, t >= 2 in the selection years
    "max_corr_existing": 0.70,     # |rank corr| with existing model features / accepted factors
    "max_t_missing": 3.0,          # "value missing" must not predict returns (survivorship leak)
    "min_coverage": 0.50,          # share of universe rows with a value in the mining period
    "max_leak_mismatch": 0.001,    # look-ahead test tolerance
}
FORBIDDEN = [r"\.shift\(\s*-", r"shift\(\s*periods\s*=\s*-", r"center\s*=\s*True", r"\.bfill\(", r"backfill",
             r"\btarget\b", r"\bfwd\b", r"future"]


def required_t(n_trials: int) -> float:
    """Two-sided Bonferroni: with N factors tried, a lucky one reaches |t| ~ z(1 - 0.025/N)."""
    return max(RULES["min_t_mine_floor"], float(stats.norm.ppf(1 - 0.025 / max(n_trials, 1))))


class Evaluator:
    def __init__(self, ctx: Context, weekly: pd.DataFrame, base_sample: pd.DataFrame, accepted: dict[str, pd.Series]):
        self.ctx = ctx
        w = weekly[weekly["date"] < HOLDOUT_START]
        self.target = w.pivot(index="date", columns="code", values="target").astype("float32")
        self.dates = self.target.index
        self.base_sample = base_sample[base_sample["date"] < HOLDOUT_START]
        self.accepted = accepted  # name -> values at base_sample rows

    # ---------- helpers
    def _weekly(self, F: pd.DataFrame) -> pd.DataFrame:
        return F.reindex(index=self.dates, columns=self.target.columns)

    @staticmethod
    def _rank_ic(A: pd.DataFrame, B: pd.DataFrame) -> pd.Series:
        m = A.notna() & B.notna()
        a = A.where(m).rank(axis=1)
        b = B.where(m).rank(axis=1)
        a = a.sub(a.mean(axis=1), axis=0)
        b = b.sub(b.mean(axis=1), axis=0)
        num = (a * b).sum(axis=1)
        den = np.sqrt((a ** 2).sum(axis=1) * (b ** 2).sum(axis=1))
        ic = num / den.replace(0, np.nan)
        return ic[m.sum(axis=1) >= 50].dropna()

    @staticmethod
    def _t(s: pd.Series) -> float:
        return float(s.mean() / s.std() * np.sqrt(len(s))) if len(s) > 2 and s.std() > 0 else 0.0

    def static_check(self, source: str) -> list[str]:
        return [p for p in FORBIDDEN if re.search(p, source)]

    def leak_check(self, func, F_full: pd.DataFrame) -> float:
        """Recompute on data truncated at a few dates; values on those dates must not change."""
        worst = 0.0
        for t in (pd.Timestamp("2016-06-30"), pd.Timestamp("2019-11-29")):
            t = self.ctx.close.index[self.ctx.close.index.searchsorted(t)]
            part = func(self.ctx.truncate(t))
            a = F_full.loc[t].astype("float64")
            b = part.loc[t].reindex(a.index).astype("float64")
            both = a.notna() | b.notna()
            if both.sum() == 0:
                continue
            diff = ~np.isclose(a[both], b[both], rtol=1e-4, atol=1e-8, equal_nan=True)
            worst = max(worst, float(diff.mean()))
        return worst

    # ---------- main
    def evaluate(self, fac, n_trials: int) -> dict:
        r = {"name": fac.name, "rationale": fac.rationale, "source": fac.source, "file": fac.file, "code_hash": fac.code_hash}
        bad = self.static_check(fac.code)
        if bad:
            return {**r, "kept": False, "reason": f"forbidden pattern(s) in code: {bad}"}
        try:
            F = fac.func(self.ctx)
        except Exception as e:  # noqa: BLE001
            return {**r, "kept": False, "reason": f"error: {type(e).__name__}: {e}"}
        if not isinstance(F, pd.DataFrame):
            return {**r, "kept": False, "reason": "factor must return a wide DataFrame (dates × codes)"}
        F = F.replace([np.inf, -np.inf], np.nan)
        W = self._weekly(F)
        in_m = (self.dates >= MINE[0]) & (self.dates < MINE[1])
        in_s = (self.dates >= SELECT[0]) & (self.dates < SELECT[1])
        univ = self.target.notna()
        cov = float((W.notna() & univ)[in_m].sum().sum() / max(univ[in_m].sum().sum(), 1))
        ic = self._rank_ic(W, self.target)
        ic_m, ic_s = ic[ic.index < MINE[1]], ic[(ic.index >= SELECT[0]) & (ic.index < SELECT[1])]
        sign = np.sign(ic_m.mean()) if len(ic_m) else 0.0
        yearly = ic_m.groupby(ic_m.index.year).mean()
        same = float((np.sign(yearly) == sign).mean()) if len(yearly) else 0.0
        t_m, t_s = self._t(ic_m), self._t(ic_s) * (sign if sign else 1)
        miss = W.isna().where(univ).astype(float)
        frac_missing = float(miss[in_m].stack().mean()) if in_m.any() else 0.0
        t_miss = self._t(self._rank_ic(miss, self.target)[lambda s: s.index < MINE[1]]) if 0.001 < frac_missing < 0.999 else 0.0
        # redundancy with the model's existing features and with factors accepted so far
        bs = self.base_sample
        vals = pd.Series(F.stack(future_stack=True), name=fac.name)
        v = vals.reindex(pd.MultiIndex.from_frame(bs[["date", "code"]])).to_numpy()
        corr, most = 0.0, ""
        cmp = {c: bs[c].to_numpy() for c in bs.columns if c not in ("date", "code")}
        cmp.update({f"accepted:{k}": s.to_numpy() for k, s in self.accepted.items()})
        dates = bs["date"].to_numpy()
        df = pd.DataFrame({"date": dates, "_f": v})
        for c, arr in cmp.items():
            df["_c"] = arr
            per = df.dropna().groupby("date").apply(lambda g: g["_f"].corr(g["_c"], method="spearman") if len(g) > 50 else np.nan,
                                                   include_groups=False)
            m = float(np.nanmean(np.abs(per))) if len(per) else 0.0
            if m > corr:
                corr, most = m, c
        leak = self.leak_check(fac.func, F)
        need_t = required_t(n_trials)
        r.update({
            "coverage": cov, "ic_mine": float(ic_m.mean()) if len(ic_m) else 0.0, "t_mine": t_m, "t_required": need_t,
            "same_sign_years": same, "ic_select": float(ic_s.mean()) if len(ic_s) else 0.0, "t_select": t_s,
            "max_corr": corr, "most_similar": most, "t_missing": t_miss, "leak_mismatch": leak,
        })
        checks = {
            "coverage": cov >= RULES["min_coverage"],
            "mining_t": abs(t_m) >= need_t,
            "mining_stable": same >= RULES["min_same_sign_years"],
            "selection": t_s >= RULES["min_t_select"],
            "not_redundant": corr < RULES["max_corr_existing"],
            "no_missing_leak": abs(t_miss) < RULES["max_t_missing"],
            "no_lookahead": leak <= RULES["max_leak_mismatch"],
        }
        r["failed"] = ",".join(k for k, ok in checks.items() if not ok)
        r["kept"] = all(checks.values())
        r["reason"] = "passed" if r["kept"] else f"failed: {r['failed']}"
        r["_sample_values"] = pd.Series(v)  # stored for future redundancy checks if kept
        return r
