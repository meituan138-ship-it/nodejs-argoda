# A 股 AI 因子挖掘工具包

让 AI（Codex、Claude 等）挖因子，程序自动验收。挖的次数越多，门槛越高；2023 年以后的数据封存，最后只打开一次。

## 目录

```
factor_mining/
  AGENT_GUIDE.md        给 AI 的指令：整篇发给 Codex
  prepare_data.py       一次性：下载数据并建缓存（免费接口，首次 2-4 小时）
  mine.py               验收因子文件，记账
  pack_results.py       挖完打包发回（<1MB，不含数据）
  gp_search.py          遗传规划自动搜索公式（数据驱动，不靠 AI 想象）
  final_check.py        封存期检验（2023-01 → 最新），只由你来跑
  fm/ops.py             因子算子（只回看，不看未来）
  fm/registry.py        @factor 装饰器（必须有 rationale + source）
  fm/evaluate.py        验收规则（AI 不许改）
  fm/ledger.py          试验账本
  factors/              因子文件放这里（example_factors.py 是模板）
  tests/leaky_factor.py 故意用未来数据的因子，用来确认检测有效
  results/ledger.csv    每一次试验的记录（挖完把整个 results/ 发回来）
```

## 步骤

```bash
# 0. 环境（Python 3.10+）
pip install pandas numpy scipy pyarrow lightgbm requests

# 1. 建数据缓存（在 research/a-shares/factor_mining 下）
python prepare_data.py

# 2. 自检：三个作弊因子都必须 rejected
python mine.py tests/leaky_factor.py
#   自检会让试验次数 +3，属于正常开销

# 3. 把 AGENT_GUIDE.md 发给 Codex，让它循环：写 factors/batch_xxx.py → python mine.py → 看结果 → 下一批
python mine.py --status      # 随时查看进度、当前门槛、已通过因子

# 4. 挖完之后（你自己跑，只跑一次）
python final_check.py
```

## 为什么要这么多规矩

- **多重检验**：随便试 1000 个随机公式，按 t>2 的标准会有约 50 个"显著"。所以门槛随试验次数上升：试 100 个要求 |t|≥3.5，试 1 万个约 4.6，试 10 万个约 5.0。
- **AI 不能凭空想因子**：每个因子必须写 `source`（论文、研报、Alpha101/国泰君安 191 编号、交易制度，或 GP 运行编号）。没有出处的故事是 AI 给噪声编出来的解释。
- **样本外两层**：2021–2022 是选择期，每次验收都要过；2023 年以后是封存期，只在最后打开。
- **泄露检测**：三种作弊都有自检因子（`tests/leaky_factor.py`），实测全部被拒绝：
  - 代码里写 `shift(-5)` → 静态检查直接拒绝；
  - 把数据倒序再滚动，绕过关键词 → 截断数据重算，结果 100% 不一致，拒绝；
  - 用不含退市股的数据源（幸存者偏差）→ "缺失本身预测收益" t=-3.8，拒绝。
- **去重**：与现有模型的 35 个基础特征（加上截面排名共 73 个）、已通过因子的截面相关 ≥0.7 的一律拒绝。换个名字的老因子没有增量。
- **最终复核**：`final_check.py` 用最终的试验次数重新卡一遍门槛。早期门槛低时通过的因子，如果达不到最终门槛也会被剔除。

## 挖完发回给我什么

```bash
python pack_results.py      # 生成 mining_results_<日期>.zip，通常不到 1MB
```
只打包因子代码、试验账本、通过的公式、NOTES.md，不含任何数据。`cache/`（约 1.4GB）和 `results/accepted/*.parquet` 不用发，
我这边按代码在同一份数据上重算。
我会在同一份数据上复核，然后把通过的因子加进周度 LightGBM，在封存期对比基线。

## 注意

- 行情用腾讯后复权 + 不复权两份，包含已退市股票（没有幸存者偏差）。资金流来自新浪，同样覆盖退市股。东方财富估值数据不含退市股，会造成泄露，所以没有使用。
- 首次下载会被限流，程序会自动退避重试。被截断的历史会重新抓，不会缓存失败结果。
- 速度（实测）：每个因子验收 10–30 秒；GP 每个公式约 3 秒，`--pop 60 --gens 8` 约 30 分钟。
- 内存：全市场日频宽表大约 14 个字段 × 70MB。8GB 内存够 `mine.py` 用；`final_check.py` 建议 16GB。
