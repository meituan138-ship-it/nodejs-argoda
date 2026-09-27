"""README charts from results/*.csv, in a light and a dark variant.

The README tables carry the same numbers (the table view for every chart).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
    },
}
EQUITY_SERIES = [  # fixed before the run: the main model, the meta model and their baselines
    ("BH_spot", "Buy & hold (spot)"),
    ("TSMOM168_LO_spot", "7-day momentum, long/flat"),
    ("E1_reg_h24_LO_spot", "E1 LightGBM 24h, long/flat"),
    ("E4_meta_tsmom_h24_LO_spot", "E4 meta-label, long/flat"),
]


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0)
    ax.grid(True, color=t["grid"], linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)


def _fig(t, w=9.0, h=4.8):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    _style(ax, t)
    return fig, ax


def equity(t, name):
    daily = pd.read_csv(RESULTS / "daily_returns.csv", index_col=0, parse_dates=True)
    fig, ax = _fig(t)
    ax.grid(axis="x", visible=False)
    for (col, label), color in zip(EQUITY_SERIES, t["series"]):
        if col not in daily:
            continue
        eq = (1 + daily[col].fillna(0)).cumprod()
        ax.plot(eq.index, eq.to_numpy(), color=color, linewidth=2, label=label)
        ax.plot(eq.index[-1], eq.iloc[-1], "o", color=color, markersize=6,
                markeredgecolor=t["surface"], markeredgewidth=2)
        ax.annotate(f"{eq.iloc[-1]:.2f}×", (eq.index[-1], eq.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=t["ink2"])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_title("Growth of 1 USDT, walk-forward out-of-sample, net of fees (2021-01 → 2026-08)",
                 loc="left", fontsize=11, color=t["ink"])
    leg = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_{'dark' if t is THEMES['dark'] else 'light'}.png", facecolor=t["surface"])
    plt.close(fig)


def ic_chart(t, name):
    s = pd.read_csv(RESULTS / "signal_summary.csv", index_col=0)
    s = s[s["task"].isin(["reg", "tb"])]
    mean = s["ic_monthly_mean"]
    se = (mean / s["ic_monthly_tstat"]).abs()
    order = mean.index[::-1]
    fig, ax = _fig(t, 8.0, 4.2)
    ax.grid(axis="y", visible=False)
    colors = [t["muted"] if i.startswith("C") else t["series"][0] for i in order]
    y = np.arange(len(order))
    ax.barh(y, mean[order], height=0.55, color=colors)
    ax.errorbar(mean[order], y, xerr=2 * se[order], fmt="none", ecolor=t["ink2"], elinewidth=1, capsize=3)
    ax.axvline(0, color=t["axis"], linewidth=1)
    ax.set_yticks(y, [i.replace("_", " ") for i in order], fontsize=9, color=t["ink2"])
    ax.set_xlabel("mean monthly rank IC (±2 s.e.)", fontsize=9, color=t["ink2"])
    ax.set_title("Out-of-sample rank IC per experiment (controls in gray)", loc="left", fontsize=11, color=t["ink"])
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_{'dark' if t is THEMES['dark'] else 'light'}.png", facecolor=t["surface"])
    plt.close(fig)


def importance(t, name, exp="E1_reg_h24", top=20):
    imp = pd.read_csv(RESULTS / "feature_importance.csv", index_col=0)[exp].dropna()
    imp = imp.sort_values(ascending=False).head(top)[::-1]
    fig, ax = _fig(t, 7.5, 5.6)
    ax.grid(axis="y", visible=False)
    ax.barh(np.arange(len(imp)), imp.to_numpy() * 100, height=0.6, color=t["series"][0])
    ax.set_yticks(np.arange(len(imp)), imp.index, fontsize=9, color=t["ink2"])
    ax.set_xlabel("share of total split gain, %", fontsize=9, color=t["ink2"])
    ax.set_title(f"Top {top} features by gain ({exp}, summed over 68 monthly models)",
                 loc="left", fontsize=11, color=t["ink"])
    fig.tight_layout()
    fig.savefig(RESULTS / f"{name}_{'dark' if t is THEMES['dark'] else 'light'}.png", facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    for theme in THEMES.values():
        equity(theme, "equity")
        ic_chart(theme, "ic")
        importance(theme, "importance")
    print("charts written to", RESULTS)
