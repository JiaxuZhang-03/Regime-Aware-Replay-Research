# Regime-Aware Experience Replay 研究说明

## 1. 这个任务在做什么

本任务服务于一个更大的研究问题：在非平稳金融市场中，强化学习智能体是否应该无差别地复用所有历史经验，还是应该根据当前市场状态选择更相关的历史 transitions 进行 replay。

普通 off-policy 强化学习方法通常依赖 replay buffer。Replay buffer 会存储过去的 `(state, action, reward, next_state)` transitions，并在训练时反复采样这些历史经验。这个机制可以提高样本效率，但在金融市场中会引入一个重要风险：市场不是静态环境。牛市、熊市、高波动震荡、低波动横盘等 regime 之间的切换，会让历史经验的含义发生变化。如果当前市场处于 risk-off 或 high-vol regime，而 replay buffer 仍大量抽到 risk-on 时期的 transitions，智能体可能会被旧 regime 下的行为奖励关系误导。

因此，当前阶段的任务不是直接搭建复杂的 SAC、Transformer 或完整投资组合系统，而是先完成一个更小、更可验证的基础模块：为历史市场数据生成稳定、可解释、可复用的 market-regime labels。后续 replay diagnostics 和 regime-aware sampler 都需要这些标签作为输入。

当前代码仓库已经完成了三类 regime labeling 方法，并将它们统一输出为 replay-buffer 友好的标签格式：

```text
date, method, regime_label, regime_name
```

共享 regime taxonomy 为：

```text
0 risk_on
1 sideways
2 high_vol
3 risk_off
```

这些标签可以通过日期 join 到每条历史 transition 上，使后续实验能够记录“当前 regime”和“被采样 transition 所属 regime”之间是否匹配。

## 2. 研究动机

本项目的核心动机不是简单地说“市场有 regime，所以策略要知道 regime”。已有研究已经讨论了 policy-level regime adaptation，例如 ReCAP 使用 adaptive regime detection、policy vector library 和 regime gate 来组合不同市场状态下的策略知识。另一些工作，例如 DEER，则从一般非平稳强化学习角度研究 replay priority 如何随环境变化调整。

本项目的切入点更窄：金融强化学习中的 replay-buffer contamination。我们关心的是，当 replay buffer 混合了来自不同市场状态的 transitions 时，uniform replay 或 prioritized experience replay 是否会在 regime shift 后继续抽取大量不匹配的旧经验，从而拖慢适应速度、增加风险暴露或放大换手。

因此，项目目前的第一步是构建 regime labels，用来回答以下诊断问题：

1. 当前市场 regime 切换后，标准 replay 是否仍大量采样旧 regime transitions？
2. 被采样 transition 的 regime 分布是否与当前 regime 明显不一致？
3. mismatched-regime samples 是否对应更高 TD-error、更慢 reward recovery、更大 drawdown 或更高 turnover？
4. 一个轻量级 regime-aware replay sampler 是否能减少 mismatch，并改善适应速度或风险稳定性？

## 3. 当前阶段为什么先做标签

在完整 RL 实验之前，最大的研究风险不是模型复杂度，而是 failure mode 是否真实存在。如果 replay contamination 在简单诊断中都不明显，那么直接开发复杂策略网络会消耗大量时间，但不一定能支撑论文贡献。

所以当前阶段的任务被刻意收窄为：

- 将市场历史压缩成按日期排列的 regime timeline。
- 使用多个 labeler 交叉检查 regime 划分是否稳定。
- 输出统一格式，便于 join 到 replay buffer transitions。
- 先验证 replay mismatch 现象，再决定是否扩展到完整 sampler 或策略训练。

换句话说，这一步是整个研究的 measurement layer。它不直接声称提升收益，而是为后续判断 replay 机制是否有问题提供可观测变量。

## 4. 数据与输入

默认输入文件为：

```text
data/market_indices_20080601_20260531/market_regime_features_wide.csv
```

该文件包含 2008-06-02 至 2026-05-29 的市场指数特征，共 4528 个交易日。特征构建模块会将输入数据标准化成每个日期一行的 market-level schema，主要包括：

- `ret_short`: 短期收益特征。
- `ret_long`: 中长期收益特征。
- `vol`: 滚动波动率。
- `trend`: 趋势或价格相对移动均线的位置。
- `vix`: VIX 或类似风险情绪指标。
- `turbulence`: 市场扰动特征，如果输入中不存在则填充为默认值。

代码也支持 long panel 输入，例如 DOW30 的 `DOW30_recap_features.csv`。如果数据中存在多个 ticker，会优先选择主资产 `SPY`；如果主资产不存在，则聚合数值特征形成市场级别序列。

## 5. 三种 regime labeling 方法

### 5.1 Rule-Based Trend/Volatility Labeler

第一种方法是透明的规则基线。它使用滚动收益、趋势、波动率和 VIX 阈值判断市场状态：

- `risk_on`: 趋势和中长期收益为正，且没有处于高波动区间。
- `high_vol`: 波动率或 VIX 高于滚动分位数，但没有明显负趋势。
- `risk_off`: 趋势转负或中长期收益为负。
- `sideways`: 其他无法明确归类的状态。

这个方法的优点是可解释、速度快、适合作为 sanity check。阈值使用滚动窗口并向后 shift 一天，避免未来信息泄露。

### 5.2 Gaussian HMM Labeler

第二种方法是 classical latent-regime baseline。它用一个对角协方差 Gaussian HMM 对市场特征建模，通过 EM 训练和 Viterbi decoding 得到 hard label，同时输出每个状态的概率。

HMM 的目标是捕捉不可直接观测的 latent market regimes。训练后，代码会根据以下风险评分对 hidden states 排序：

```text
return + trend - volatility - VIX
```

这样可以把 HMM 的无名隐状态映射到更容易解释的 `risk_off`、`sideways`、`risk_on` 等 regime 名称，减少每次运行后人工 relabel 的工作。

### 5.3 ReCAP-Inspired CUSUM/ARD Labeler

第三种方法借鉴 ReCAP 中 Adaptive Regime Detection 的思想，但不复现完整 ReCAP。这里仅使用 CUSUM-style change detection 对市场特征做自适应分段。

该方法会：

- 在市场级特征上运行对称 CUSUM 统计量。
- 检测 change points。
- 生成 variable-length `segment_id`。
- 根据每个 segment 的收益、趋势、波动率和 VIX 摘要，将 segment 映射到共享 taxonomy。

这个方法的角色是提供一个 literature-aware 的 adaptive segmentation baseline。它比固定滚动规则更关注 regime switch 的时间结构，也比完整 policy-level ReCAP 更轻量。

## 6. 已生成的产物

当前已生成的标签文件位于：

```text
outputs/regime_labels/
```

主要输出包括：

- `rule_based_labels.csv`
- `hmm_labels.csv`
- `recap_cusum_labels.csv`
- `all_regime_labels.csv`
- `label_summary.csv`
- `label_switches.csv`
- 每个方法对应的 metadata JSON

当前 summary 显示，不同方法给出的 regime 分布并不完全相同，这正是多 labeler 设计的意义：它可以帮助我们区分稳定的 regime 信号和方法敏感性。

截至当前输出：

| 方法 | 主要分布 | label switches |
| --- | --- | ---: |
| rule_based | risk_on 58.46%, risk_off 28.09%, high_vol 13.32%, sideways 0.13% | 366 |
| hmm | sideways 39.77%, risk_on 33.81%, risk_off 26.41% | 97 |
| recap_cusum | risk_on 62.68%, risk_off 27.27%, high_vol 10.05% | 40 |

这个结果符合方法性质：rule-based 对滚动阈值更敏感，因此切换次数更多；HMM 会产生更平滑的隐状态序列；CUSUM/ARD 则显式寻找 change points，因此 segment 数量更少。

## 7. 这些标签如何进入 replay 实验

后续 replay 实验中，每条 transition 都可以通过日期获得 regime label：

```text
transition.date -> regime_label
```

训练时还可以记录当前日期的 regime：

```text
current_date -> current_regime_label
```

有了这两个变量，就可以定义 replay mismatch：

```text
mismatch = sampled_transition_regime != current_regime
```

进一步可以统计：

- sampled regime distribution
- current regime 与 sampled transition regime 的 mismatch rate
- regime switch 前后的 mismatch rate 变化
- TD-error by current/sampled regime pair
- reward recovery time after regime switches
- drawdown after regime switches
- turnover spike after regime switches

如果 uniform replay 或 PER 在 regime switch 后持续抽取大量不匹配 transitions，而 regime-aware replay 能降低 mismatch 并改善至少一个风险或适应指标，那么项目就能形成明确贡献：金融 regime shift 下的 replay-buffer contamination 诊断，以及一个轻量级 regime-aware sampling 修正。

## 8. 最小 regime-aware replay sampler 思路

当前报告对应的后续 sampler 可以先保持非常简单，不必一开始发明复杂模型。一个最小版本可以设置：

```text
50% same-or-similar regime transitions
50% normal replay transitions
```

其中 same-or-similar 可以先用离散标签判断，也可以将 `risk_off` 与 `high_vol` 视为相近风险状态。这样做的好处是实验可解释：如果只改变采样分布，而策略网络、奖励函数、交易成本设置不变，就更容易把性能差异归因于 replay 机制。

对照方法应包括：

- uniform replay
- prioritized experience replay
- sliding-window replay
- regime-aware replay

这四者可以分别回答不同问题：uniform 是普通基线，PER 检查 TD-error priority 是否足够，sliding-window 检查只用近期经验是否有效，regime-aware replay 则检验“历史经验是否应按市场状态选择”。

## 9. 当前阶段的边界

需要明确的是，当前仓库已经实现的是 regime labeling 和基础 diagnostics，并不是完整强化学习交易系统。当前代码还没有完成：

- DQN/SAC 等策略训练主循环。
- replay buffer variants 的统一实验框架。
- transaction cost 下的 portfolio backtest。
- replay mismatch 与 TD-error、drawdown、turnover 的联合分析图。

这些不是缺陷，而是当前研究节奏的刻意安排。先把 regime timeline 和可观测 mismatch 指标做好，可以防止项目过早陷入复杂模型开发，同时也方便判断研究 gap 是否真实存在。

## 10. 与仓库文件的对应关系

当前报告对应的主要代码和文档如下：

- `README.md`: 项目总览、quick start、输出文件说明。
- `docs/regime_labeling_notes.md`: regime labeling 方法笔记和文献定位。
- `src/regime_labeling/features.py`: 将宽表或 long panel 数据统一转换成市场级特征。
- `src/regime_labeling/rule_based.py`: 规则型 trend/volatility 标签器。
- `src/regime_labeling/hmm.py`: Gaussian HMM 标签器。
- `src/regime_labeling/recap_ard.py`: ReCAP-inspired CUSUM/ARD 标签器。
- `src/regime_labeling/diagnostics.py`: 标签分布和切换次数统计。
- `scripts/make_regime_labels.py`: 本地生成所有标签的入口脚本。
- `outputs/regime_labels/`: 已生成标签、summary、switch counts 和 metadata。

这些文件共同完成的不是最终投资策略，而是把金融历史转化为可被 replay buffer 使用的 regime-indexed experience dataset。

## 11. 下一步工作

建议下一阶段按以下顺序推进：

1. 将 `all_regime_labels.csv` join 到 RL 环境生成的 transitions。
2. 在 uniform replay 和 PER 中记录 sampled transition 的 regime。
3. 输出 current/sampled regime pair 的计数矩阵和 mismatch rate。
4. 在 synthetic bull/bear/high-vol 环境中先验证现象是否清晰。
5. 将同样诊断迁移到 ETF 或 DOW30 portfolio 数据。
6. 实现最小 regime-aware sampler，并与 uniform、PER、sliding-window 比较。
7. 如果 mismatch 诊断明确，再扩展到更复杂的策略网络或软 regime 表示。

## 12. 一句话总结

这个任务正在为 “regime-aware experience replay for financial non-stationarity” 建立第一层证据：先把金融历史分成可比较的市场 regime，再用这些标签检查 replay buffer 是否在 regime shift 后复用了不合时宜的旧经验。当前阶段的核心产物不是一个最终交易策略，而是后续 replay-contamination 诊断和 regime-aware sampling 实验所必需的标签与测量基础。
