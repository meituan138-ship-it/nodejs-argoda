"""Candidate factors for the factor lab. Each one states *why* it might work;
factors without an economic story are not allowed in.

All inputs are known at the close of day t (daily bars + Sina money flow,
which is aggregated from tick data and published after the close).
Units: Sina `turnover` is in 1/10000 of free-float shares; `netamount` /
`r0_net` in CNY; `ratioamount` / `r0_ratio` as fractions of traded amount.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _z(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return (x - x.rolling(n, min_periods=n // 2).mean()) / (x.rolling(n, min_periods=n // 2).std() + EPS)


def build_candidates(o, h, l, c, vol, rc, mf: dict[str, pd.DataFrame]) -> dict[str, tuple[pd.DataFrame, str]]:
    """Return {name: (wide frame dates×codes, rationale)}."""
    days = c.index
    traded = vol.fillna(0) > 0
    amount = (vol * 100 * rc).where(traded)
    r1 = np.log(c).diff().where(traded)
    mkt = r1.mean(axis=1)

    def mfw(col):
        return pd.DataFrame({k: d[col] for k, d in mf.items() if col in d}).reindex(index=days, columns=c.columns).where(traded)

    to = mfw("turnover") / 1e4          # turnover as a fraction of free float
    net = mfw("netamount")              # main-force net inflow, CNY
    r0 = mfw("r0_net")                  # extra-large orders net inflow, CNY
    free_cap = (amount / to.replace(0, np.nan))              # free-float market cap = amount / turnover

    F: dict[str, tuple[pd.DataFrame, str]] = {}
    # --- money flow (Level-2 derived)
    for n in (1, 5, 20):
        F[f"mf_net_ratio_{n}"] = (net.rolling(n, min_periods=max(1, n // 2)).sum() /
                                  (amount.rolling(n, min_periods=max(1, n // 2)).sum() + EPS),
                                  "主力净流入占成交额：大资金持续买入可能有信息优势；散户市场里也可能是'诱多'（方向由数据决定）")
    F["mf_xl_ratio_5"] = (r0.rolling(5, min_periods=3).sum() / (amount.rolling(5, min_periods=3).sum() + EPS),
                          "超大单净流入占比：最大一档订单的方向")
    F["mf_net_z_60"] = (_z(net / (amount + EPS), 60), "主力净流入占比相对自身历史的异常程度")
    F["mf_pos_days_10"] = ((net > 0).astype(float).where(traded).rolling(10, min_periods=5).mean(),
                           "近 10 天主力净流入为正的天数占比：持续性")
    F["mf_price_divergence_5"] = ((r1.rolling(5).sum()).rank(axis=1, pct=True) -
                                  (net.rolling(5, min_periods=3).sum() / (amount.rolling(5, min_periods=3).sum() + EPS)).rank(axis=1, pct=True),
                                  "价涨但主力流出（或价跌但主力流入）的背离")
    # --- turnover / size
    F["turnover_5"] = (to.rolling(5, min_periods=3).mean(), "换手率：A 股高换手 = 散户关注过度，后续收益偏低")
    F["turnover_20"] = (to.rolling(20, min_periods=10).mean(), "20 日平均换手率")
    F["turnover_ratio_5_60"] = (np.log((to.rolling(5, min_periods=3).mean() + EPS) / (to.rolling(60, min_periods=30).mean() + EPS)),
                                "异常换手：短期换手相对长期突然放大")
    F["turnover_std_20"] = (to.rolling(20, min_periods=10).std(), "换手率波动：关注度忽高忽低")
    F["log_free_cap"] = (np.log(free_cap.rolling(5, min_periods=1).median() + 1), "流通市值（由成交额/换手率反推）：规模效应")
    # --- candle structure
    rng = (h - l).where(traded)
    F["upper_shadow_20"] = (((h - np.maximum(o, c)) / (rng + EPS)).rolling(20, min_periods=10).mean(), "上影线：冲高回落 = 上方抛压")
    F["lower_shadow_20"] = (((np.minimum(o, c) - l) / (rng + EPS)).rolling(20, min_periods=10).mean(), "下影线：探底回升 = 下方承接")
    F["close_location_20"] = (((c - l) / (rng + EPS)).rolling(20, min_periods=10).mean(), "收盘在当日区间的位置：尾盘强弱")
    gap = np.log(o / c.shift(1))
    F["gap_up_count_20"] = ((gap > 0.02).astype(float).where(traded).rolling(20, min_periods=10).sum(), "大幅高开次数：情绪化追涨")
    F["up_days_60"] = ((r1 > 0).astype(float).where(traded).rolling(60, min_periods=30).mean(), "上涨天数占比：走势'质量'")
    F["up_volume_share_20"] = ((amount.where(r1 > 0, 0)).rolling(20, min_periods=10).sum() / (amount.rolling(20, min_periods=10).sum() + EPS),
                               "上涨日成交额占比：资金是涨时进还是跌时进")
    F["kurt_60"] = (r1.rolling(60, min_periods=30).kurt(), "收益尖峰：极端日多 = 彩票型")
    vol5 = r1.rolling(5).std()
    F["vol_of_vol_60"] = (vol5.rolling(60, min_periods=30).std() / (vol5.rolling(60, min_periods=30).mean() + EPS), "波动的波动：不确定性")
    F["dd_60"] = (np.log(c / c.rolling(60, min_periods=30).max()), "距 60 日高点回撤")
    F["dist_high_20"] = (np.log(c / h.rolling(20, min_periods=10).max()), "距 20 日最高价")
    vwap20 = amount.rolling(20, min_periods=10).sum() / ((vol * 100).where(traded).rolling(20, min_periods=10).sum() + EPS)
    F["close_vs_vwap_20"] = (np.log(rc / (vwap20 + EPS)), "价格相对 20 日成交均价：近期买入者整体盈亏")
    F["pv_corr_60"] = (r1.rolling(60, min_periods=30).corr(np.log1p(amount).diff()), "量价相关：放量涨/放量跌")
    beta = (r1.mul(mkt, axis=0).rolling(60, min_periods=30).mean() - r1.rolling(60, min_periods=30).mean().mul(mkt.rolling(60).mean(), axis=0)) \
        .div(mkt.rolling(60).var() + EPS, axis=0)
    F["idio_ret_20"] = ((r1.sub(beta.mul(mkt, axis=0))).rolling(20, min_periods=10).sum(), "剔除大盘后的特质收益：个股自身的过度反应")
    F["rev_x_turnover"] = ((-r1.rolling(5).sum()).rank(axis=1, pct=True) * to.rolling(5, min_periods=3).mean().rank(axis=1, pct=True),
                           "高换手下的反转：放量下跌的超卖更容易修复")
    lim_up = (c / c.shift(1) - 1) >= 0.095
    last_up = lim_up.where(lim_up).notna()
    F["limit_up_recent_5"] = (last_up.astype(float).where(traded).rolling(5, min_periods=1).sum(), "近 5 日涨停：打板情绪的延续或反转")
    F["amihud_change"] = (np.log(((r1.abs() / (amount + 1)).rolling(20, min_periods=10).mean() + EPS) /
                                 ((r1.abs() / (amount + 1)).rolling(120, min_periods=60).mean() + EPS)), "流动性变差/变好")
    F["ret_60_skip_5"] = (np.log(c.shift(5) / c.shift(60)), "跳过最近一周的中期动量")
    F["max_ret_5"] = (r1.rolling(5).max(), "近 5 日最大单日涨幅：短期彩票效应")
    return F
