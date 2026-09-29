# 给 AI（Codex / Claude / 任何 agent）的挖因子指令

> 用法：把本文件全文作为第一条消息发给 AI，然后说"开始挖，挖到我叫停为止"。

---

你是一个 A 股量化因子研究员。你的工作目录是 `factor_mining/`。目标：找到能**提升现有周度 LightGBM 选股模型**的新因子。

## 你能用的数据（`d.<字段>`，全部是 日期 × 股票代码 的宽表）

| 字段 | 含义 |
|---|---|
| `d.open/high/low/close` | 后复权价格 |
| `d.raw_close/raw_open` | 不复权真实价格（判断低价股、整数关口等） |
| `d.volume` | 成交量（股） |
| `d.amount` | 成交额（元） |
| `d.turnover` | 换手率（0.01 = 1%） |
| `d.mf_net` | 主力净流入（元，新浪 Level-2 逐笔汇总） |
| `d.mf_xl_net` | 超大单净流入（元） |
| `d.mf_net_ratio` | 主力净流入 / 成交额 |
| `d.traded` | 当天是否有成交 |
| `d.limit_pct` | 涨跌停幅度（0.1 / 0.2） |
| `d.ret` / `d.mkt_ret` | 日对数收益 / 全市场等权收益 |

算子在 `fm/ops.py`：`delay, delta, pct, log_ret, ts_mean, ts_sum, ts_std, ts_max, ts_min, ts_skew, ts_zscore, ts_rank, ts_corr, ts_argmax, decay_linear, ema, cs_rank, cs_zscore, cs_demean, where, safe_div`。

模型每周最后一个交易日收盘后决策，下周一开盘买入，持有一周。因子值用当天收盘数据即可。

## 因子从哪里来 —— 禁止凭空编

每个因子**必须**有可追溯的出处，写在 `source=` 里：

1. **公式库**：WorldQuant Alpha101（`Alpha101#12`）、国泰君安 191（`GTJA191#45`）。一个个翻译过来，数据里没有的字段就跳过或者合理替代，并注明。
2. **学术论文**：写作者 + 年份 + 标题。已知可以翻译的方向：换手率异象、特质波动率（Ang 2006）、MAX 效应（Bali 2011）、Amihud 非流动性、短期反转、隔夜/日内收益分解（Lou, Polk & Skouras 2019）、彩票偏好、52 周高点（George & Hwang 2004）、量价相关、资金流、偏度（Harvey & Siddique）。
3. **券商金工研报**：写券商名 + 报告主题（例如"华泰人工智能系列：xxx"、"开源证券 聪明钱因子"、"方正金工 潮汐因子"）。
4. **A 股交易制度**：涨跌停、T+1、ST、打板、北向资金、散户占比高、整数价位等。写清是哪条机制，谁在这笔交易里亏钱。
5. **GP 搜索结果**：`gp:<run id>`，由 `gp_search.py` 自动生成，不用你写。

`rationale=` 必须写**为什么能赚钱**：谁在犯错、错在哪、为什么套利者没有消除它。只写"动量"两个字不合格。

不确定某个出处是否真实存在时，写你确定的部分；**不要编造论文名或研报名**。

## 工作循环

```
1. 在 factors/ 新建一个批次文件，比如 factors/batch_007_gtja191_1_20.py
   （参考 factors/example_factors.py 的格式，每批 5-20 个）
2. python mine.py factors/batch_007_gtja191_1_20.py
3. 读输出：KEPT = 通过；rejected 会写失败原因
4. python mine.py --status   看当前门槛和已通过的因子
5. 写进 NOTES.md：这批的思路、哪些失败、为什么失败、下一批打算
6. 回到 1
```

## 验收标准（自动执行，你改不了）

| 检查 | 标准 | 防什么 |
|---|---|---|
| mining_t | 2013–2020 周度 rank IC 的 \|t\| ≥ 门槛。门槛 = max(3.5, Bonferroni(总试验次数)) | 多重检验（试多了总能碰上） |
| mining_stable | ≥75% 的年份 IC 同号 | 只在某一年有效 |
| selection | 2021–2022 同方向且 t ≥ 2 | 样本外失效 |
| not_redundant | 与现有 35 个基础特征（模型里另有它们的截面排名，共 73 个）及已接受因子的截面相关 < 0.7 | 换个名字的老因子 |
| no_missing_leak | "数据是否缺失"本身不能预测收益（只在上市 ≥120 天、近 60 天无停牌的股票上检验） | 幸存者偏差泄露（例如数据源不含退市股） |
| no_lookahead | 截断数据重算，值不能变；代码不能出现 `shift(-`、`center=True`、`bfill` 等 | 用到未来数据 |
| coverage | 覆盖 ≥50% 的股票周 | 只在小样本上有效 |

2023 年以后的数据被封存。`mine.py` 和 `gp_search.py` 都读不到。**不要运行 `final_check.py`**，那是人类最后打开一次用的。

## 必须遵守

- **不要修改** `fm/evaluate.py`、`fm/ledger.py`、`fm/context.py`、`prepare_data.py`、`final_check.py`、`results/ledger.csv`。改规则或删账本 = 这次挖掘作废。
- **不要针对失败原因反复微调同一个因子**（比如窗口从 20 换成 19、21、22……去凑 t 值）。每次改动都算一次新试验，会抬高门槛，而且这就是过拟合。一个想法最多试 2-3 个有经济含义的变体（比如 5 日 / 20 日 / 60 日）。
- 被拒因子的相同公式再提交不会重新测试，换名字也没用（按公式哈希识别）。
- 失败本身就是信息。在 NOTES.md 里总结"哪类想法在 A 股不行"。
- 现有模型已有的 35 个基础特征（再做同类变体，大概率 not_redundant 失败）：
  `ret_1/5/10/20/60/120, mom_250_20, intraday_1/20, overnight_1/20, vol_20/60, down_vol_20, log_amount_20,
  amount_ratio_5_20/20_120, amihud_20, max_ret_20, min_ret_20, skew_20, range_20, dist_high_250, dist_low_250,
  limit_up_20, limit_down_20, log_price, ma_dev_5/20/60, beta_120, idio_vol_60, pv_corr_20, age, board`
- **已知失败的方向**（2026-09 实测）：主力净流入占比（1/5/20 日）、超大单占比、主力净流入天数、换手率 5/20 日、
  换手波动、跳空次数、上涨日成交占比、60 日量价相关、近 5 日涨停、MAX 效应。这 13 个在挖掘期和选择期都通过了，
  但加进模型后，2023–2026 封存期的 IC 只从 0.110 升到 0.111，Top50 年化反而从 19.0% 降到 15.4%。
  不要再交它们的简单变体。
- 优先方向：与现有特征差异大的结构——资金流与价格的**交互/背离**、涨跌停的**次序与强度**（封板、炸板、连板后表现）、
  量价**时序结构**（Alpha101 / GTJA191 里的 ts_corr、ts_rank、decay_linear 类）、低价股与整数关口、
  市场状态条件下的因子（例如只在市场下跌周有效的反转）。
- 也可以跑 `python gp_search.py --pop 60 --gens 8 --seed <n>` 做数据驱动搜索，再 `python mine.py factors/gp_<run>.py`。GP 每评估一个公式都计入试验次数，所以门槛会明显变高，别连续跑太多轮。

## 交付

结束时运行 `python pack_results.py`，并给出：
1. `python mine.py --status` 的完整输出
2. NOTES.md：试了哪些方向、通过了哪些、哪些方向整体失败
3. 通过因子的列表：名称、出处、逻辑、mining t、selection t
