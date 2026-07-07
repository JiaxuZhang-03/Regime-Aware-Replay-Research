# ReCAP-Inspired 策略安全层与多模型库更新

## 背景

`mechanism_experiment_plan_0627.html` 已放入 `report/`。该文档指出，当前阶段基础交易模型还可能学到退化策略，因此不应只看最终 portfolio return，而应先验证 regime shift 是否造成可观测的学习不稳定，并分析 replay 机制是否在稳定地选择有用样本。

结合 ReCAP 的思想，本次更新没有只优化单个模型，而是加入一个跨模型可复用的策略安全层和 RL 模型库，用来支持泛化性实验。

## 核心改动

新增 `src/rl_trading/policy_safety.py`，实现一个轻量级 ReCAP-inspired policy guard。它包含一个 regime-conditioned anchor policy library，并在每次模型输出动作后执行：

- 将模型 proposed portfolio 与当前 regime 的 anchor portfolio 混合。
- 设置最低 cash 权重。
- 限制单资产最大权重。
- 限制单步 turnover。

该机制默认对 DQN 和 SAC 都启用，因此不依赖某一个模型结构。它的作用不是替代 RL 策略，而是降低不同 RL 模型学到极端集中、过度换手或在 high-vol/risk-off regime 下过度 risk-on 的坏策略概率。

## 与 ReCAP 的对应关系

ReCAP 使用 adaptive regime detection、policy library 和 regime gate 来应对市场状态变化。本次实现借鉴的是它的机制思想：

- regime labels 作为当前市场状态输入。
- anchor portfolio library 作为简化版 policy library。
- `safety_regime_blend` 作为简化版 regime gate。
- DQN/SAC 输出的动作作为当前模型策略。
- 最终执行动作为模型策略与 regime anchor 的混合，并经过风险约束。

这使得策略提升不只绑定在 SAC 上，而可以用于多个 RL 模型，从而支持“方法具有泛化性”的研究叙事。

## 多模型库

新增 `src/rl_trading/model_registry.py` 和 `scripts/run_rl_library.py`。模型库目前注册了：

- `dqn`: 离散动作 value-based baseline。
- `sac`: 连续动作 actor-critic baseline。

可以用一条命令统一运行：

```bash
python scripts/run_rl_library.py --models dqn,sac --label-method rule_based --replays uniform,regime,deer --seeds 0 --max-steps 120
```

输出合并到：

```text
outputs/rl_library/analysis/rl_library_summary.csv
```

## 研究意义

这一步为后续 performance 提升提供了更稳的基础：

1. 如果 DQN 和 SAC 都能在同一 safety/replay 框架下减少退化策略，说明机制不是某个模型的偶然结果。
2. 如果关闭 safety guard 后模型更容易出现高 drawdown、高 turnover 或极端权重，则可以作为策略安全层有效性的 ablation。
3. 如果 regime-aware/DEER replay 在多个模型上都改善 mismatch、TD-error shock 或 post-boundary recovery，则更能支持 replay 机制的泛化性。

当前阶段建议先运行短窗口和多 seed smoke，再逐步扩大到完整 backtest。
