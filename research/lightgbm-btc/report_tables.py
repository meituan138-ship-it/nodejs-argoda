"""Render results/*.csv as the markdown tables used in the README (results/tables.md)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

NAMES = {
    "E1_reg_h24": "E1 回归 24h（主模型）",
    "E2_reg_h4": "E2 回归 4h",
    "E3_tb_h24": "E3 三重障碍 24h",
    "E4_meta_tsmom_h24": "E4 meta-label（7日动量）24h",
    "E5_reg_h24_price_only": "E5 仅价量特征 24h",
    "E6_reg_h24_rolling2y": "E6 滚动2年窗口 24h",
    "E7_vol_h24": "E7 波动率 24h",
    "E8_reg_h168": "E8 回归 168h（一周）",
    "E9_reg_h24_fixed300": "E9 固定300棵树 24h",
    "E10_ridge_h24": "E10 Ridge 线性模型 24h",
    "C1_shuffle_h24": "C1 阴性对照（打乱标签）",
    "C2_leak_h24": "C2 阳性对照（泄漏特征 ρ≈0.1）",
    "C3_leak_ridge_h24": "C3 阳性对照 + Ridge",
}
STRAT_NAMES = {
    "BH_spot": "买入持有（现货）",
    "Long_perp": "永续一直做多（含资金费）",
    "TSMOM168_LS_perp": "7日动量 多空（永续）",
    "TSMOM168_LO_spot": "7日动量 多/空仓（现货）",
}


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _strategy_label(s: str) -> str:
    if s in STRAT_NAMES:
        return STRAT_NAMES[s]
    for key, label in NAMES.items():
        if s.startswith(key):
            variant = "多空·永续" if s.endswith("LS_perp") else "只做多·现货"
            return f"{label.split('（')[0]} · {variant}"
    return s


def signal_table(sig: pd.DataFrame) -> str:
    rows = ["| 实验 | 月均 Rank IC | t 值 | IC>0 月份 | 方向 AUC | 方向命中率 | 高置信 10% 命中率 | 中位树数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in sig.iterrows():
        if r["task"] in ("vol", "meta"):
            continue
        rows.append(
            f"| {NAMES.get(name, name)} | {r['ic_monthly_mean']:+.4f} | {r['ic_monthly_tstat']:+.2f} | "
            f"{_pct(r['ic_months_positive'])} | {r['auc_direction']:.4f} | {_pct(r['hit_rate_direction'])} | "
            f"{_pct(r['hit@q90'])} | {r['median_trees']:.0f} |"
        )
    return "\n".join(rows)


def meta_table(sig: pd.DataFrame) -> str:
    m = sig[sig["task"] == "meta"]
    if m.empty:
        return ""
    r = m.iloc[0]
    return (
        "| meta 模型 AUC | 7日动量本身胜率 | LightGBM 过滤后胜率 | 保留交易比例 | 高置信 10% 方向命中率 |\n"
        "|---:|---:|---:|---:|---:|\n"
        f"| {r['meta_auc']:.4f} | {_pct(r['primary_precision'])} | {_pct(r['filtered_precision'])} | "
        f"{_pct(r['coverage'])} | {_pct(r['hit@q90'])} |"
    )


def strategy_table(st: pd.DataFrame) -> str:
    rows = ["| 策略 | 年化(CAGR) | 夏普 | 最大回撤 | 年换手(倍) | 年手续费 | 年资金费 | DSR | 夏普@0bp | 夏普@10bp |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in st.iterrows():
        if r["family"] == "control":
            continue
        dsr = "" if pd.isna(r.get("dsr")) else f"{r['dsr']:.2f}"
        s0 = "" if pd.isna(r.get("sharpe@0bps")) else f"{r['sharpe@0bps']:.2f}"
        s10 = "" if pd.isna(r.get("sharpe@10bps")) else f"{r['sharpe@10bps']:.2f}"
        rows.append(
            f"| {_strategy_label(name)} | {_pct(r['cagr'])} | {r['sharpe']:.2f} | {_pct(r['max_dd'])} | "
            f"{r['turnover_per_year']:.0f} | {_pct(r['cost_per_year'])} | {_pct(r['funding_per_year'])} | "
            f"{dsr} | {s0} | {s10} |"
        )
    return "\n".join(rows)


def control_table(sig: pd.DataFrame, st: pd.DataFrame) -> str:
    rows = ["| 对照 | 月均 Rank IC | t 值 | 方向 AUC | 多空夏普（扣费） |", "|---|---:|---:|---:|---:|"]
    for name in ("C1_shuffle_h24", "C2_leak_h24", "C3_leak_ridge_h24", "E1_reg_h24", "E10_ridge_h24"):
        if name not in sig.index:
            continue
        r = sig.loc[name]
        sh = st.loc[f"{name}_LS_perp", "sharpe"] if f"{name}_LS_perp" in st.index else float("nan")
        rows.append(f"| {NAMES[name]} | {r['ic_monthly_mean']:+.4f} | {r['ic_monthly_tstat']:+.2f} | "
                    f"{r['auc_direction']:.4f} | {sh:.2f} |")
    return "\n".join(rows)


def yearly_tables(yr: pd.DataFrame, st_daily: pd.DataFrame) -> str:
    order = [n for n in NAMES if n in set(yr["experiment"])]
    ic = yr.pivot(index="experiment", columns="year", values="ic_spearman").reindex(order)
    sh = yr.pivot(index="experiment", columns="year", values="sharpe").reindex(order)
    years = list(ic.columns)
    head = "| 实验 | " + " | ".join(str(y) for y in years) + " |\n|---|" + "---:|" * len(years)
    out = ["**分年 Rank IC（全样本逐小时）**", "", head]
    for name in ic.index:
        if name.startswith("C"):
            continue
        out.append(f"| {NAMES.get(name, name)} | " + " | ".join(f"{ic.loc[name, y]:+.3f}" for y in years) + " |")
    out += ["", "**分年夏普（多空·永续，扣费和资金费）**", "", head]
    for name in sh.index:
        if name.startswith("C"):
            continue
        out.append(f"| {NAMES.get(name, name)} | " + " | ".join(f"{sh.loc[name, y]:+.2f}" for y in years) + " |")
    # buy & hold and momentum per year from daily returns, for reference
    d = st_daily
    for col in ("BH_spot", "TSMOM168_LS_perp"):
        if col in d:
            g = d[col].groupby(d.index.year)
            vals = (g.mean() / g.std() * (365 ** 0.5)).reindex(years)
            out.append(f"| {STRAT_NAMES[col]}（日频计算） | " + " | ".join(f"{v:+.2f}" for v in vals) + " |")
    return "\n".join(out)


def vol_table(vs: dict) -> str:
    rows = ["| 24h 已实现波动率预测 | MSE（log 波动率） | 相对朴素模型 R² | 与真实值相关 |", "|---|---:|---:|---:|"]
    label = {"naive": "朴素（过去 24h 波动 = 未来）", "har": "HAR（日/周/月）", "lgbm": "LightGBM（100 个特征）"}
    for k in ("naive", "har", "lgbm"):
        r2 = "0" if k == "naive" else f"{vs['r2_vs_naive'][k]:+.3f}"
        rows.append(f"| {label[k]} | {vs['mse'][k]:.4f} | {r2} | {vs['corr'][k]:.3f} |")
    rows += ["", "| 波动率目标 50% 的现货持仓（每日 00:00 调仓） | 年化 | 夏普 | 最大回撤 | 平均仓位 |", "|---|---:|---:|---:|---:|"]
    vlabel = {"BH_spot": "买入持有", "VolTarget_naive": "用朴素波动预测定仓位",
              "VolTarget_har": "用 HAR 定仓位", "VolTarget_lgbm": "用 LightGBM 定仓位"}
    for k, s in vs["vol_targeting"].items():
        rows.append(f"| {vlabel.get(k, k)} | {_pct(s['cagr'])} | {s['sharpe']:.2f} | {_pct(s['max_dd'])} | {s['exposure']:.2f} |")
    return "\n".join(rows)


def importance_table(imp: pd.DataFrame, exp: str = "E1_reg_h24", top: int = 15) -> str:
    s = imp[exp].dropna().sort_values(ascending=False).head(top)
    rows = ["| 排名 | 特征 | 增益占比 |", "|---:|---|---:|"]
    rows += [f"| {i} | `{k}` | {v * 100:.1f}% |" for i, (k, v) in enumerate(s.items(), 1)]
    return "\n".join(rows)


def main() -> None:
    sig = pd.read_csv(RESULTS / "signal_summary.csv", index_col=0)
    st = pd.read_csv(RESULTS / "strategy_summary.csv", index_col=0)
    yr = pd.read_csv(RESULTS / "yearly.csv")
    daily = pd.read_csv(RESULTS / "daily_returns.csv", index_col=0, parse_dates=True)
    imp = pd.read_csv(RESULTS / "feature_importance.csv", index_col=0)
    parts = [
        "### 预测质量", signal_table(sig),
        "### Meta-labeling", meta_table(sig),
        "### 策略绩效（2021-01 → 2026-08，扣费）", strategy_table(st),
        "### 对照实验", control_table(sig, st),
        "### 分年", yearly_tables(yr, daily),
        "### 特征重要性（E1）", importance_table(imp),
    ]
    vs_path = RESULTS / "vol_study.json"
    if vs_path.exists():
        parts += ["### 波动率", vol_table(json.loads(vs_path.read_text()))]
    text = "\n\n".join(parts) + "\n"
    (RESULTS / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
