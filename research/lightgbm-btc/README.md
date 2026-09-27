# LightGBM 预测 BTC：全网调研 → 共识方法 → 可复现算法 → 真实数据验证

> 这不是投资建议。所有结论都来自公开资料和本目录代码在 Binance 公开数据上的 walk-forward 回测，
> 回测不等于实盘。

<!-- RESULTS-TLDR -->

---

## 目录

1. [调研了什么（论文 / 竞赛 / 开源项目 / 教程）](#1-调研了什么)
2. [被反复验证的共识（11 条）](#2-被反复验证的共识)
3. [算法：从数据到仓位的完整流程](#3-算法)
4. [真实数据验证（2021-01 → 2026-08，68 次月度重训）](#4-真实数据验证)
5. [结论与使用建议](#5-结论与使用建议)
6. [复现](#6-复现)
7. [参考来源](#7-参考来源)

---

## 1. 调研了什么

### 1.1 学术论文

| 来源 | 做法 | 关键结果 | 可信度 / 备注 |
|---|---|---|---|
| Sun, Liu, Sima (2020), *Finance Research Letters* 32 [[1]](#ref1) | LightGBM 预测加密货币"跌 / 不跌"，42 个币的日线 + 宏观指标 | LightGBM 优于 SVM、RF；2 周周期比 2 天更准，最高准确率 0.905 / 0.952 | 最常被引用的 "LightGBM + 加密货币" 论文；准确率异常高，重叠窗口、趋势型标签和切分方式都会抬高数字，**其他研究没有复现出这个量级** |
| Jaquart, Dann, Weinhardt (2021), *J. Finance & Data Science* [[2]](#ref2) | BTC 1–60 分钟方向预测；技术 / 链上 / 情绪 / 资产特征；GBC、RF、LSTM、GRU、LR | 梯度提升和 RNN 最好；**周期越长越可预测**；技术面特征最重要；多空策略扣费前月收益可达 39%，**扣费后为负** | 设计规范，结论被后续项目反复印证 |
| Jaquart, Köpke, Weinhardt (2022), *JFDS* 8 [[3]](#ref3) | 100 个币的日度相对涨跌 | 平均准确率 **52.9%–54.1%**；只看每类置信度最高的 10% 时升到 **57.5%–59.5%** | 给出了真实的"天花板"量级 |
| Liu & Tsyvinski (2021), *Review of Financial Studies* [[4]](#ref4) | BTC/ETH/XRP 收益的风险因子与可预测性 | **时间序列动量**很强；关注度（Google 搜索）1 个标准差 → BTC 未来 2 周收益 +2.3% | 顶刊，动量是加密货币最稳健的信号 |
| Liu, Tsyvinski, Wu (2022), *Journal of Finance* [[5]](#ref5) | 横截面因子 | 市场、规模、**动量**三因子解释加密货币收益 | 顶刊 |
| Shen, Urquhart, Wang (2022), *Financial Review* [[6]](#ref6) | BTC 日内动量 | 以成交量定义"开盘"，首个半小时收益正向预测最后半小时；高成交量 / 高波动时段最强 | 支持"时间 + 成交量"类特征 |
| 集成学习 vs 深度学习比较 (2024), *Int. Review of Financial Analysis* [[7]](#ref7) | RF / GBM / XGBoost / LightGBM / LSTM / GRU 等 | **LightGBM 在 BTC、ETH、LTC 上排名第一**，XRP 上 GRU 第一；集成模型在不同市场状态下更稳定 | 支持"LightGBM 是强基线" |
| BTC 波动率预测 (2024–2025, *JIFMIM*、*Expert Systems with Applications*) [[8]](#ref8) | LightGBM / XGBoost / HAR / GARCH | LightGBM 在确定性和概率性波动率预测上都有效；加入 HAR 分量（日/周/月已实现波动）显著提升 | **波动率比方向好预测得多** |
| López de Prado (2018), *JPM* "The 10 Reasons Most ML Funds Fail" [[9]](#ref9) | 方法论 | 10 个坑：Sisyphus 范式、靠回测做研究、时间采样、整数差分、固定周期标签、方向与仓位一起学、非 IID 样本权重、**交叉验证泄漏**、walk-forward 单路径、**回测过拟合** | 对应解法：特征重要性、成交量时钟、分数差分、**三重障碍**、**meta-labeling**、唯一性权重、**purging + embargo**、CPCV、**Deflated Sharpe** |
| Joubert (2022), *JFDS* "Meta-Labeling: Theory and Framework" [[10]](#ref10) | 控制实验 | meta-labeling 可以过滤假阳性、用于仓位管理，改善夏普和最大回撤；多状态、非线性数据上集成更有利 | Hudson & Thames 开源代码 |
| Bisdoulis (2025), arXiv 2501.07580 [[11]](#ref11) | LightGBM 资产预测的目标变换 | 7 种目标变换中 **log 收益、收益、EMA 差比** 最好 | 支持"不要回归价格" |
| **反例**：Twitter + LGBM (arXiv 2409.15988) [[12]](#ref12) | 推文语义向量 + LightGBM | 小时 / 4 小时 / 日线"准确率" 78%–94% | 用**同期**推文解释同期涨跌，是"解释"不是"预测"，不能交易 |
| Turn-of-the-candle (2023, PMC) [[13]](#ref13) | 1 分钟 BTC | 收益集中在每 15 分钟 K 线切换的那一分钟，t 值 > 9 | 分钟级季节性真实存在，但只对分钟级交易有意义 |

### 1.2 竞赛（最接近"大规模反复验证"的证据）

| 竞赛 | 设定 | LightGBM 相关结论 |
|---|---|---|
| **Kaggle G-Research Crypto Forecasting**（2021-11 → 2022-05，1946 队，3 个月实盘数据评估）[[14]](#ref14) | 14 个币、分钟 K 线，预测未来 15 分钟**剔除市场后的残差收益**，指标为加权相关系数 | **前三名全部用 LightGBM**；主办方总结：*冠军们一致认为特征工程贡献最大，比模型开发影响大得多*。第 2 名：LightGBM 平方损失、除树数/叶子数/学习率外全默认，**6 折 walk-forward、每折 40 周、折间隔 1 周**；第 3 名：只用收盘价派生特征 `log(close / 滚动均价)`、`log(close / 滞后收盘价)` 及其**减去全市场均值**的版本，EmbargoCV；第 9 名：按上涨 / 下跌 / 震荡训练 3 个 LightGBM 专家模型，Hull MA（窗口 55/210/340/890/3750）是最重要特征 [[15]](#ref15) |
| **Kaggle DRW Crypto Market Prediction**（2025，1091 队）[[16]](#ref16) | 订单簿 / 成交流特征预测未来收益 | 有效特征集中在**订单不平衡、主动成交不平衡、流动性、多周期成交量**；冠军称"特征质量高时线性模型就很强" |
| **Numerai Crypto** [[17]](#ref17) | 自带特征预测代币收益排名 | 官方示例就是 LightGBM：`n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=32, colsample_bytree=0.1`——小学习率 + 强列采样 |

### 1.3 开源项目

| 项目 | 做法 | 结论 |
|---|---|---|
| **freqtrade / FreqAI** [[18]](#ref18) | `LightGBMRegressor` / `LightGBMClassifier`；特征 = RSI、MFI、ADX、SMA、EMA、布林带、ROC、相对成交量 × 多周期 × 多时间框架 × 相关币对 × 前几根 K 线平移；目标 = 未来 `label_period_candles` 根 K 线收盘均值 / 当前收盘 − 1；**滚动重训**（示例训练 15 天、回测 7 天），SVM 去离群点、DI 阈值 | 最成熟的实盘框架，"多周期技术特征 + 收益率目标 + 滚动重训"范式 |
| **microsoft/qlib** [[19]](#ref19) | LightGBM + Alpha158 基准 | 参数：`lr 0.2, num_leaves 210, max_depth 8, lambda_l1 205.7, lambda_l2 580.98, colsample 0.888, subsample 0.879`——**极强的 L1/L2 正则**；Alpha158 的 K 线形态 + 多窗口统计因子是很好的特征模板 |
| **Hassnat07/crypto-ml-backtest** [[20]](#ref20) | BTC/ETH/SOL/BNB 4 小时线 2020–2026，29 个特征，三重障碍标签，5 折 purged walk-forward，**预注册**判定标准 | BTC AUC **0.554**，精度比基准高 8.3pp，但 0.30% 往返成本下每笔净 **−0.021%**，结论 NO-GO；**2022 熊市 AUC 0.416，2024 年 0.625——"模型是牛市探测器"** |
| **intikhab49/edgeproof** [[21]](#ref21) | 15 分钟线，LightGBM / LSTM，三重障碍、purged k-fold、唯一性权重、Deflated Sharpe，**阳性对照** | 方向模型扣 taker 费后**无优势**（利润因子 0.65–0.98）；阳性对照能被检测出来；**波动率预测**比基准好 9.6% |
| **hudson-and-thames** mlfinlab / meta-labeling [[10]](#ref10) | López de Prado 方法的参考实现 | 三重障碍、purged CV、meta-labeling 的标准代码 |
| 中文社区（知乎 / CSDN / 腾讯云 / VeighNa / BigQuant）[[22]](#ref22) | 大多是 G-Research 或 Optiver 方案的复述，或"技术指标 + LightGBM + 滚动训练" | 很多教程**直接回归价格或随机切分**，图看起来很准，实际是滞后复制 |
| Flovik, *How (not) to use ML for time series forecasting* [[23]](#ref23) | 教程 | 演示了"预测价格"的模型只是把昨天的值平移一格，看似准确毫无预测力 |

---

## 2. 被反复验证的共识

按"支持它的独立来源数量"排序：

1. **目标用（波动率标准化的）收益率 / 方向 / 三重障碍，绝不用价格。** 价格回归只学到"明天 ≈ 今天"。G-Research 用残差收益、Numerai 用收益排名、FreqAI 用未来均价收益率、Bisdoulis 的对比实验都指向同一结论。[[11]](#ref11)[[14]](#ref14)[[18]](#ref18)[[23]](#ref23)
2. **LightGBM 是表格型金融特征的默认强基线，但特征工程比模型重要。** G-Research 前三全是 LightGBM 且都说特征最重要；IRFA 2024 中 LightGBM 在 BTC 上第一；Jaquart 2021 中 GBC 和 RNN 并列最好。[[2]](#ref2)[[7]](#ref7)[[14]](#ref14)
3. **有效特征族几乎固定**：多周期收益（动量 / 反转）、价格相对均线和区间位置、已实现波动率、成交量与**主动买卖不平衡**、时间（小时 / 星期）、跨资产（ETH、市场均值）、衍生品（资金费率、基差）。技术面最重要，链上 / 情绪次之。[[2]](#ref2)[[4]](#ref4)[[6]](#ref6)[[15]](#ref15)[[16]](#ref16)[[18]](#ref18)
4. **所有特征要平稳化**：用收益率、比值、z-score、除以近期波动率，而不是价格、成交量的绝对水平。[[9]](#ref9)[[15]](#ref15)[[19]](#ref19)
5. **验证只能用 walk-forward + purge + embargo。** 随机 K 折会泄漏未来。G-Research 第 2 名 1 周间隔、第 3 名 EmbargoCV、FreqAI 滚动重训、López de Prado 坑 #8。[[9]](#ref9)[[14]](#ref14)[[18]](#ref18)
6. **信号很弱是常态**：方向准确率 52%–56%、AUC 0.52–0.58；高置信子集更准。论文或教程里 70%–95% 的"准确率"几乎都来自泄漏、同期解释或随机切分。[[3]](#ref3)[[12]](#ref12)[[20]](#ref20)
7. **交易成本决定生死，周期越长越好做。** 分钟到 15 分钟级扣费后普遍为负；可预测性随周期上升。[[2]](#ref2)[[20]](#ref20)[[21]](#ref21)
8. **强烈依赖市场状态**，常常只是"牛市探测器"——必须滚动重训，并且**分年**报告结果。[[20]](#ref20)
9. **波动率远比方向好预测**，LightGBM + HAR 分量做波动率预测被多篇研究验证，最直接的用途是仓位管理 / 风控。[[8]](#ref8)[[21]](#ref21)
10. **Meta-labeling**：让简单规则（如 7 日动量）决定方向，LightGBM 只决定"做不做 / 做多大"，比直接预测涨跌更容易提升精度。[[9]](#ref9)[[10]](#ref10)
11. **LightGBM 参数要保守**：小学习率、浅树、大 `min_data_in_leaf`、强 L1/L2、行列采样、早停、多种子平均；并用 Deflated Sharpe 校正多次试验带来的虚高。[[9]](#ref9)[[17]](#ref17)[[19]](#ref19)

---

## 3. 算法

把上面的共识落成一套可以直接跑的流程（代码在 `lgbm_btc/`）：

```
输入  BTCUSDT 1h K 线（含主动买入量、成交笔数）、ETHUSDT 1h、BTC 永续资金费率、溢价指数（基差）
      全部来自 data.binance.vision 免费公开归档

0. 时间约定  第 t 根 K 线在收盘时（open_time + 1h）做决策；特征只用 ≤ t 的数据，标签只用 > t 的数据
1. 波动率尺度  σ_t = EWMA_std(1h 对数收益, span = 168)
2. 特征（100 个，全部因果、平稳化）
   动量      ret_k = log(C_t / C_{t-k}) / (σ_t·√k),  k ∈ {1,2,4,8,12,24,48,72,120,168,336,504,720}
   趋势      log(C / EMA_n) / (σ·√n)、EMA 斜率、Hull MA 偏离（n = 55, 210）、EMA24/168 交叉
   区间      (C − 最低) / (最高 − 最低)、距 n 小时最高 / 最低的标准化距离，n ∈ {24, 168, 720}
   振荡      RSI(14/48/168)、Stoch、MACD 柱 / (C·σ)、布林 %B、ADX、DI 差、MFI
   波动      已实现波动(6..720)、波动比、Parkinson、Garman-Klass、ATR/σ、偏度、峰度、下行方差占比、波动的波动
   量 / 流   成交额 z-score、相对成交额、成交笔数 z、平均单笔 z、主动买卖不平衡 OFI(1/6/24/168)、Amihud、量价相关
   K 线形态  实体、上下影线比例、振幅/σ
   时间      小时、星期
   跨资产    ETH 多周期收益、ETH/BTC 比值动量、BTC-ETH 相关、波动比
   衍生品    最近一次资金费率、1 天 / 7 天均值、30 天 z-score；溢价指数及其 24h 均值、z-score、变化
3. 标签（主模型）  y_t = log(C_{t+H} / C_t) / (σ_t·√H)，训练时截断到 ±4
   备选：三重障碍（±1·σ·√H，先碰哪边）、meta 标签（7 日动量方向这笔交易是否赚钱）、未来 H 小时已实现波动率
4. 验证  每月 1 日重训一次；训练集 = 2018-01-01 至（本月第一根 K 线 − H − 24 根）；
         其中最后 15% 做早停验证集，与训练集之间再 purge H 根
5. 模型  LightGBM：learning_rate 0.02、num_leaves 31、max_depth 6、min_data_in_leaf 500、
         feature_fraction 0.6、bagging_fraction 0.7、lambda_l2 10；
         早停指标 = 验证集 Pearson IC（patience 200）；按数据量等比放大树数后在 train+valid 上重训；3 个种子平均
6. 信号 → 仓位  s_t = sign(ŷ_t)；pos_t = mean(s_{t−H+1..t})（H 个重叠子组合，各持有 H 小时，天然降换手）
         多空版：永续合约，5 bp/边 + 资金费；只做多版：现货，10 bp/边
7. 评估  月度 Rank IC 及 t 值、方向 AUC、分年夏普、扣费净值、换手、Deflated Sharpe（按全部试过的策略变体校正）
8. 对照  阴性：打乱训练标签；阳性：加入一个与未来标签相关系数 ≈ 0.1 的"泄漏"特征。
         阴性必须没信号、阳性必须被检测到，否则整条流水线不可信
```

`tests/test_leakage.py` 自动检查：截断未来数据后特征不变（point-in-time）、资金费率只在结算后可见、标签不越过 H、各折之间正确 purge、回测时序约定，以及在纯随机游走上必须测不出信号。

<!-- RESULTS-BODY -->

## 6. 复现

```bash
cd research/lightgbm-btc
pip install -r requirements.txt
python -m pytest -q tests/                 # 泄漏 / 对照测试，约 10 秒，无需联网
python run_experiments.py                  # 下载约 11MB 公开数据 + 跑 10 组实验（4 核约 1 小时）
python make_charts.py                      # 生成 results/*.png
python predict_latest.py                   # 用全部历史训练，输出最新一根 K 线的信号
```

输出：`results/signal_summary.csv`（预测质量）、`strategy_summary.csv`（策略绩效）、`yearly.csv`（分年）、
`feature_importance.csv`、`vol_study.json`、`daily_returns.csv`。

## 7. 参考来源

<a id="ref1"></a>[1] Sun, Liu, Sima (2020). A novel cryptocurrency price trend forecasting model based on LightGBM. *Finance Research Letters* 32. https://www.sciencedirect.com/science/article/abs/pii/S1544612318307918 ·
<a id="ref2"></a>[2] Jaquart, Dann, Weinhardt (2021). Short-term bitcoin market prediction via machine learning. *JFDS* 7. https://www.sciencedirect.com/science/article/pii/S2405918821000027 ·
<a id="ref3"></a>[3] Jaquart, Köpke, Weinhardt (2022). Machine learning for cryptocurrency market prediction and trading. *JFDS* 8. https://www.sciencedirect.com/science/article/pii/S2405918822000174 ·
<a id="ref4"></a>[4] Liu, Tsyvinski (2021). Risks and Returns of Cryptocurrency. *RFS* 34(6). https://academic.oup.com/rfs/article-abstract/34/6/2689/5912024 ·
<a id="ref5"></a>[5] Liu, Tsyvinski, Wu (2022). Common Risk Factors in Cryptocurrency. *JF*. https://onlinelibrary.wiley.com/doi/10.1111/jofi.13119 ·
<a id="ref6"></a>[6] Shen, Urquhart, Wang (2022). Bitcoin intraday time series momentum. *Financial Review* 57(2). https://onlinelibrary.wiley.com/doi/10.1111/fire.12290 ·
<a id="ref7"></a>[7] Cryptocurrency price forecasting – A comparative analysis of ensemble learning and deep learning methods (2024). *IRFA*. https://www.sciencedirect.com/science/article/pii/S1057521923005719 ·
<a id="ref8"></a>[8] Forecasting Bitcoin volatility using machine learning techniques (2024) https://www.sciencedirect.com/science/article/pii/S1042443124001306 ；Multivariate forecasting of bitcoin volatility with gradient boosting (2025) https://www.sciencedirect.com/science/article/abs/pii/S0957417425040199 ·
<a id="ref9"></a>[9] López de Prado (2018). The 10 Reasons Most Machine Learning Funds Fail. *JPM* 44(6). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104816 ·
<a id="ref10"></a>[10] Joubert (2022). Meta-Labeling: Theory and Framework. *JFDS*. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4032018 ；代码 https://github.com/hudson-and-thames/meta-labeling ·
<a id="ref11"></a>[11] Bisdoulis (2025). Assets Forecasting with Feature Engineering and Transformation Methods for LightGBM. https://arxiv.org/abs/2501.07580 ·
<a id="ref12"></a>[12] Semi-strong Efficient Market of Bitcoin and Twitter (2024). https://arxiv.org/abs/2409.15988 ·
<a id="ref13"></a>[13] Turn-of-the-candle effect in bitcoin returns (2023). https://pmc.ncbi.nlm.nih.gov/articles/PMC10015199/ ·
<a id="ref14"></a>[14] G-Research: Wrapping up the G-Research Crypto Forecasting Competition. https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/ ；竞赛页 https://www.kaggle.com/competitions/g-research-crypto-forecasting ·
<a id="ref15"></a>[15] G-Research 方案汇总（第 2 / 3 / 7 / 9 名）https://kaggle.curtischong.me/competitions/G-Research-Crypto-Forecasting ；第 2 名 https://www.kaggle.com/competitions/g-research-crypto-forecasting/discussion/323098 ；第 3 名 https://www.kaggle.com/code/sugghi/training-3rd-place-solution ·
<a id="ref16"></a>[16] DRW Crypto Market Prediction. https://www.kaggle.com/competitions/drw-crypto-market-prediction ；第 1 名 https://www.kaggle.com/competitions/drw-crypto-market-prediction/writeups/drw-solution-1st ·
<a id="ref17"></a>[17] Numerai Crypto 文档. https://docs.numer.ai/numerai-crypto/crypto-overview ·
<a id="ref18"></a>[18] FreqAI 文档 https://www.freqtrade.io/en/stable/freqai/ ；示例策略 https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/FreqaiExampleStrategy.py ·
<a id="ref19"></a>[19] Qlib LightGBM Alpha158 基准配置. https://github.com/microsoft/qlib/blob/main/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml ·
<a id="ref20"></a>[20] Hassnat07/crypto-ml-backtest. https://github.com/Hassnat07/crypto-ml-backtest ·
<a id="ref21"></a>[21] intikhab49/edgeproof-crypto-trading-backtest. https://github.com/intikhab49/edgeproof-crypto-trading-backtest ·
<a id="ref22"></a>[22] 例：知乎《时间序列-基于LightGBM预测数字货币收益》https://zhuanlan.zhihu.com/p/666402408 ；腾讯云《Optiver波动率预测大赛系列解读二：LightGBM模型及特征工程》https://cloud.tencent.com/developer/article/1892507 ；VeighNa 社区 LightGBM 期货信号模型 https://www.vnpy.com/forum/topic/34259 ·
<a id="ref23"></a>[23] Flovik (2018). How (not) to use Machine Learning for time series forecasting: Avoiding the pitfalls. https://medium.com/data-science/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424
