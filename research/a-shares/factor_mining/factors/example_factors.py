"""Template: copy this file to factors/<your_batch>.py and add factors.

Rules (the evaluator enforces them):
  * return a wide DataFrame (dates × codes), value known at the close of day t
    (the trade happens at the next open, so same-day close data is fine)
  * only look backwards: use fm.ops; never .shift(-k), center=True, bfill
  * every factor needs `rationale` (why it works, who loses the money) and
    `source` (paper / broker report / Alpha101#n / Alpha191#n / market rule / gp:run)
"""
from fm.ops import cs_rank, delay, safe_div, ts_corr, ts_max, ts_mean, ts_std, ts_sum  # noqa: F401
from fm.registry import factor


@factor("abn_turnover_5_60",
        rationale="短期换手相对长期换手的放大：散户关注度突然上升后往往被高估，随后回落（注意力驱动买入）",
        source="Barber & Odean 2008 attention-grabbing; 国内'异常换手率'研报（如东方证券因子手册）")
def abn_turnover_5_60(d):
    return safe_div(ts_mean(d.turnover, 5), ts_mean(d.turnover, 60))


@factor("amihud_illiq_20",
        rationale="非流动性溢价：单位成交额引起的价格变动越大，持有者要求的补偿越高",
        source="Amihud 2002, Illiquidity and stock returns")
def amihud_illiq_20(d):
    return ts_mean(safe_div(d.ret.abs(), d.amount), 20)


@factor("vol_price_corr_10",
        rationale="量价同向（放量上涨/缩量下跌）与量价背离的持续性差异",
        source="Alpha101 #6: -correlation(open, volume, 10)")
def vol_price_corr_10(d):
    return -ts_corr(d.open, d.volume, 10)
