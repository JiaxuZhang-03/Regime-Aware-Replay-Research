# Synthetic Market 实验阶段报告

## 1. 实验目的

这一阶段的目标不是马上证明交易策略能盈利，而是先构造一个可控的 synthetic market，用已知的真实 regime 来检查 regime-aware replay 是否真的改变了 replay 行为，以及这种机制能否在更干净的环境中帮助 SAC 适应 regime shift。

真实金融数据里同时存在很多问题：

- market signal 很弱；
- reward 噪声很大；
- RL 容易学到 degenerate policy；
- regime label 本身也不一定准确；
- replay 机制的效果容易被 base model 问题盖住。

所以 synthetic market 的作用是把问题拆开：

> 如果我们知道真实 regime，且 market dynamics 是人为控制的，那么不同 replay method 是否会表现出合理的机制差异？

当前 synthetic market 通过脚本生成：

```text
scripts/generate_synthetic_market.py
```

生成结果包括：

```text
market.csv
labels.csv
metadata.json
```

这些实验数据和结果都放在 `outputs/` 下面，不需要上传到 repo。需要上传和维护的主要是 generator 脚本、SAC runner、报告和配置。

---

## 2. Synthetic Market 设计

当前 synthetic market 是 multi-stock portfolio allocation environment。

资产为：

```text
SYN0, SYN1, SYN2, SYN3, SYN4
```

每一天的收益来自 regime-dependent Gaussian distribution：

```text
r_t | z_t = k ~ N(mu_k, Sigma_k)
```

其中：

- `z_t` 是当前 regime；
- `mu_k` 控制每个 regime 下各股票的 expected return；
- `Sigma_k` 控制 volatility 和 correlation；
- SAC agent 不直接看到 true regime；
- replay label 可以使用 oracle true regime，也可以使用 HMM / ReCAP-CUSUM 的 estimated regime。

SAC 的 action 是连续 portfolio logits，环境通过 softmax 转成：

```text
cash + SYN0 + SYN1 + SYN2 + SYN3 + SYN4
```

reward 是 net log return：

```text
gross_return = dot(asset_weights, next_day_asset_returns)
turnover = sum(abs(new_weights - previous_weights))
cost = transaction_cost_bps / 10000 * turnover
reward = log(1 + gross_return) - cost
```

当前默认 transaction cost 是 `10 bps`。

---

## 3. Level 0: Stationary Market

### 设置

Level 0 没有 regime switch：

```text
r_t ~ N(mu, Sigma)
```

目的：

- sanity check；
- 没有 non-stationarity 时，regime-aware replay / DEER 不应该凭空获得优势；
- DEER 应该退化成 PER-like fallback。

### 发现

一开始 Level 0 暴露了一个重要 bug：没有 regime boundary 时，SAC-DEER 仍然会做 DoE/probe refresh，并且 post-boundary 诊断不正确。修复后：

- `current_boundary = 0` 时，DEER 完全退化成 PER fallback；
- 不做 DoE scale refresh；
- 不标记 post-change samples。

修复后 3 seeds、300 steps 的结果符合预期：

```text
PER 和 DEER 结果完全一致
regime replay 没有人工优势
mismatch = 0
```

Level 0 说明当前 DEER fallback 逻辑已经合理。

---

## 4. Level 1: Two-Regime Mean Shift

### 设置

Level 1 有两个 regime：

```text
0 = bull
1 = bear
```

regime 切换由 Markov process 控制：

```text
每天以 switch_prob = 0.035 的概率切换到另一个 regime
```

平均大约每 `1 / 0.035 ≈ 29` 个交易日切换一次。

Level 1 只改变 expected return，不改变 volatility / correlation。

bull mean vector：

```text
SYN0: 0.0005625
SYN1: 0.00065625
SYN2: 0.00075
SYN3: 0.00084375
SYN4: 0.0009375
```

bear mean vector：

```text
SYN0: -0.0009375
SYN1: -0.00080625
SYN2: -0.000675
SYN3: -0.00054375
SYN4: -0.0004125
```

直觉：

- bull 中大多数股票有正收益；
- bear 中大多数股票有负收益；
- 不同股票有不同 regime sensitivity。

### 3-seed smoke 结果

```text
uniform final = 0.8548, MDD = 0.1977, mismatch = 0.4989
PER     final = 0.8551, MDD = 0.1977, mismatch = 0.4908
regime  final = 0.8549, MDD = 0.1976, mismatch = 0.2434
DEER    final = 0.9156, MDD = 0.1718, mismatch = 0.4755
```

### 发现

- regime replay 明显降低 mismatch；
- DEER 在 Level 1 的短跑中 final value 和 MDD 都更好；
- 这说明当 regime 只改变 return direction 时，DoE-style replay 可能确实捕捉到一些 adaptation signal；
- 但这是 300-step smoke，还不能作为正式结论。

---

## 5. Level 2: Mean + Volatility Shift

### 设置

Level 2 同时改变 expected return 和 volatility。

```text
bull:
  positive mean
  daily vol = 0.010

bear:
  negative mean
  daily vol = 0.020
```

bear 的日波动率是 bull 的 2 倍，所以方差大约是 4 倍。

直觉：

- Level 1 只改变收益方向；
- Level 2 同时改变收益方向和风险水平；
- replay 需要处理 stale return signal 和 stale risk signal。

### 3-seed smoke 结果

```text
uniform final = 0.8476, MDD = 0.2181, mismatch = 0.4989
PER     final = 0.8485, MDD = 0.2175, mismatch = 0.4907
regime  final = 0.8481, MDD = 0.2175, mismatch = 0.2426
DEER    final = 0.8812, MDD = 0.2232, mismatch = 0.4756
```

### 发现

- regime replay 继续显著降低 mismatch；
- DEER final value 仍高于 uniform/PER，但 MDD 没有改善；
- 相比 Level 1，Level 2 更难，因为风险结构也改变了；
- 当前 SAC 的 training length 和参数还不足以稳定判断 portfolio-level superiority。

---

## 6. Level 3: Mean + Volatility + Correlation Shift

### 设置

Level 3 同时改变：

```text
expected return
volatility
cross-stock correlation
```

参数：

```text
bull:
  positive mean
  daily vol = 0.010
  corr = 0.15

bear:
  negative mean
  daily vol = 0.020
  corr = 0.45
```

直觉：

- bear 不只是收益更差、波动更高；
- 股票之间也更一起动；
- diversification benefit 在 bear 中下降；
- 这更像 portfolio-specific regime shift。

本次生成的 1000 天数据：

```text
bull = 556 days
bear = 444 days
switches = 37
```

### 3-seed smoke 结果

```text
buy-hold equal weight: final = 1.1737, MDD = 0.1318

uniform final = 0.9694, MDD = 0.1250, mismatch = 0.4989
PER     final = 0.9695, MDD = 0.1244, mismatch = 0.4871
regime  final = 0.9690, MDD = 0.1249, mismatch = 0.2424
DEER    final = 0.9380, MDD = 0.1411, mismatch = 0.4733
```

### 发现

- regime replay 继续降低 mismatch；
- 但 portfolio return 没有提升；
- DEER 在 Level 3 中反而弱于 uniform/PER；
- buy-and-hold equal-weight 明显强于 SAC；
- 这说明 base SAC policy 还没有学好，replay 方法的效果被 base policy 问题盖住。

---

## 7. Level 4: Rare Crisis Regime

### 设置

Level 4 增加第三个 rare crisis regime：

```text
0 = bull
1 = bear
2 = crisis
```

transition matrix：

```text
bull   -> bull 0.960, bear 0.035, crisis 0.005
bear   -> bull 0.045, bear 0.940, crisis 0.015
crisis -> bull 0.080, bear 0.120, crisis 0.800
```

crisis 设置：

```text
crisis mean: roughly -0.0034 to -0.0019
crisis daily vol = 0.035
crisis corr = 0.80
```

直觉：

- crisis 少见；
- 一旦进入 crisis，会持续一段时间；
- crisis 中大部分股票显著负收益；
- volatility 和 correlation 都很高；
- diversification 基本失效。

本次 1000 天数据：

```text
bull = 433 days
bear = 523 days
crisis = 44 days
switches = 58
```

### 3-seed smoke 结果

```text
buy-hold equal weight: final = 1.0867, MDD = 0.1867

uniform final = 0.9265, MDD = 0.1997, mismatch = 0.4919
PER     final = 0.9270, MDD = 0.1996, mismatch = 0.4859
regime  final = 0.9266, MDD = 0.1995, mismatch = 0.2420
DEER    final = 0.8825, MDD = 0.2277, mismatch = 0.4635
```

### 发现

- regime replay 仍然显著降低 mismatch；
- 但收益和 MDD 基本没有比 uniform/PER 更好；
- DEER 在 Level 4 中表现更差；
- buy-and-hold 仍然强于 SAC；
- crisis regime 让环境更接近我们想要的 rare but important regime setting，但也进一步暴露了 SAC base policy 的问题。

---

## 8. Level 5: Hidden / Estimated Regime

### 设置

Level 5 沿用 Level 4 的 hidden true regime market，但 replay label 不一定使用 true regime。

标签来源：

```text
rule_based  = oracle true regime control
hmm         = Gaussian HMM estimated regime
recap_cusum = ReCAP-inspired CUSUM estimated regime
```

也就是说：

- `rule_based` 在 Level 5 中不是普通 rule-based label，而是 oracle true regime；
- `hmm` 和 `recap_cusum` 是从 synthetic market features 估计出来的 noisy regime label。

这可以测试：

> replay method 在 estimated regime 不完美时是否仍然有效？

### 1000-day smoke 发现

HMM 标签：

```text
labels = risk_off / risk_on / sideways
switches = 31
```

ReCAP-CUSUM 标签：

```text
labels = risk_off / high_vol / risk_on
switches = 12
```

它们都不会完美对齐 true regime，这符合 Level 5 的设计目标。

在 300-step smoke 中：

```text
HMM estimated labels 下，DEER final = 1.0258, MDD = 0.1594
ReCAP-CUSUM estimated labels 下，DEER final = 0.9648, MDD = 0.1882
```

短跑里 DEER 在 estimated labels 下看起来更好，但这个结果不稳定，不能直接下结论。

---

## 9. 正常长度 Level 5 实验

### 实验设置

这是目前最正式的一轮 synthetic experiment。

数据：

```text
level = level5_hidden_or_estimated_regime
seed = 42
n_days = 2500
n_assets = 5
```

真实 regime 分布：

```text
bull = 1170 days
bear = 1243 days
crisis = 87 days
switches = 143
```

SAC 设置：

```text
max_steps = 1500
seeds = 0,1,2
warmup_steps = 256
start_steps = 256
batch_size = 128
hidden_dim = 128
label_method = rule_based,hmm,recap_cusum
replays = uniform,per,regime,deer
```

Buy-and-hold benchmark 使用前 1500 个交易日的 equal-weight portfolio：

```text
20% SYN0, 20% SYN1, 20% SYN2, 20% SYN3, 20% SYN4
no rebalancing
turnover = 0
```

### 正常长度实验结果

```text
Buy & Hold Equal Weight:
final value = 0.8944
MDD = 0.3972
turnover = 0
```

Oracle true regime labels：

```text
uniform final = 0.3690, MDD = 0.6477, mismatch = 0.5250
PER     final = 0.3690, MDD = 0.6477, mismatch = 0.5203
regime  final = 0.3673, MDD = 0.6492, mismatch = 0.2598
DEER    final = 0.3758, MDD = 0.6384, mismatch = 0.5145
```

HMM estimated labels：

```text
uniform final = 0.3690, MDD = 0.6477, mismatch = 0.6463
PER     final = 0.3690, MDD = 0.6477, mismatch = 0.6373
regime  final = 0.3680, MDD = 0.6486, mismatch = 0.3099
DEER    final = 0.3751, MDD = 0.6391, mismatch = 0.6381
```

ReCAP-CUSUM estimated labels：

```text
uniform final = 0.3690, MDD = 0.6477, mismatch = 0.4828
PER     final = 0.3690, MDD = 0.6477, mismatch = 0.4809
regime  final = 0.3677, MDD = 0.6488, mismatch = 0.2215
DEER    final = 0.3749, MDD = 0.6392, mismatch = 0.5092
```

### 发现

最重要的发现是：

> SAC 目前明显跑不过 buy-and-hold equal-weight。

这说明当前 synthetic market 已经能跑通，但 SAC base policy 还没有学好。

进一步看：

- regime replay 可以稳定降低 mismatch；
- DEER 的 final value 和 MDD 略好于 uniform/PER/regime；
- 但所有 SAC 方法都远差于 buy-and-hold；
- 因此不能把当前 portfolio result 当成方法有效性的正式证据；
- 现在更适合把这些结果作为 debugging / mechanism diagnostics。

---

## 10. 为什么 Uniform 和 PER 看起来几乎一样

我们检查了 Uniform 和 PER 的训练日志。

结论：

> Uniform 和 PER 不是代码上完全一样，PER 确实在生效；只是它对最终 portfolio 表现影响很小。

证据：

```text
seed 0:
uniform final = 0.3964369
PER final     = 0.3966800
diff          = 0.0002431

seed 1:
uniform final = 0.3523100
PER final     = 0.3519865
diff          = -0.0003235

seed 2:
uniform final = 0.3581472
PER final     = 0.3583056
diff          = 0.0001584
```

它们不是完全同一条轨迹：

```text
max action/weight diff roughly 0.14 to 0.18
portfolio path also differs slightly
```

PER 的 priority 也确实更新：

```text
mean_priority 从 1.0 降到约 0.06-0.08
mean_sample_prob 不是常数
```

为什么最终结果几乎一样：

1. TD-error 后期很快收敛到比较窄的范围，PER sampling 接近 uniform。
2. SAC base policy 还没学好，replay sampling 差异难以转化成 portfolio 差异。
3. importance-sampling weight 会部分校正 PER sampling bias。
4. 高 turnover 和交易成本主导了最终表现。

---

## 11. 当前结论

目前 synthetic market 已经完成了从 Level 0 到 Level 5 的基本实验框架：

```text
Level 0: stationary sanity check
Level 1: mean shift
Level 2: mean + volatility shift
Level 3: mean + volatility + correlation shift
Level 4: rare crisis regime
Level 5: hidden / estimated regime
```

机制上最稳定的发现：

```text
regime-aware replay consistently reduces replay mismatch
```

DEER 的发现：

```text
DEER 在 Level 1/2 smoke 中有一定优势
DEER 在 Level 3/4 smoke 中偏弱
DEER 在 Level 5 normal experiment 中略好于 uniform/PER，但幅度不大
```

最大问题：

```text
SAC base policy 目前显著弱于 buy-and-hold equal-weight
```

因此，当前阶段不能说 replay 方法已经带来有效 portfolio improvement。更准确的说法是：

> synthetic market 已经能作为机制验证平台使用；regime-aware replay 的 mismatch reduction 是稳定存在的；但在正式比较 replay 方法前，需要先让 base SAC policy 至少接近 buy-and-hold。

---

## 12. 下一步建议

### 优先级 1: 修 SAC base policy

当前 SAC 的平均 turnover 大约在：

```text
0.59
```

这意味着每天组合权重变化很大。在 10 bps transaction cost 下，高 turnover 会严重拖累 portfolio value。

建议优先尝试：

```text
1. 降低 action volatility
2. 加 turnover penalty 或提高 transaction cost sensitivity
3. 降低 actor learning rate
4. 延长 training steps
5. 调整 entropy target / alpha
6. 加入 no-trade inertia 或 weight smoothing
```

### 优先级 2: 做 longer synthetic experiment

如果 base policy 有改善，可以跑：

```text
n_days = 5000
max_steps = 4000
seeds = 5
warmup_steps = 512
start_steps = 512
```

### 优先级 3: 增加 post-switch diagnostics

除了 full-period final value，还应该看：

```text
post-switch 20/50/100 day return
post-switch drawdown
time to recover
distance to oracle regime portfolio
stale-regime replay ratio
```

这些指标更贴近 regime-aware replay 的核心贡献。

### 优先级 4: 调 DEER 参数

当前 DEER 的表现不稳定，可能需要调：

```text
deer_s0
deer_half_life
deer_min_post_samples
deer_scale_refresh_freq
deer_lambda
```

尤其在 SAC 里，Q-discrepancy 很可能比 DQN 更噪声，需要更保守的 S schedule。

---

## 13. 文件位置

主要代码：

```text
scripts/generate_synthetic_market.py
src/rl_trading/sac_replay.py
```

主要结果：

```text
outputs/synthetic_market/
outputs/sac_synthetic_level5_normal_1500/
```

正常长度实验聚合表：

```text
outputs/sac_synthetic_level5_normal_1500/analysis/level5_normal_1500_aggregated_with_buy_hold.csv
```

Level 3/4 regime 图：

```text
outputs/synthetic_market/synthetic_level3_level4_syn0_regimes.png
```

Level 1/2 regime 图：

```text
outputs/synthetic_market/synthetic_level1_level2_syn0_regimes.png
```
