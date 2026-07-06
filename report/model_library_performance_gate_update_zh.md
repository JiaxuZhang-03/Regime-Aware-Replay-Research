# 模型广场 Performance Gate 更新

## 问题

当前 DQN/SAC 模型广场已经能跑多种 replay 方法，但短窗口和真实市场数据上仍可能出现退化策略：高换手、过度集中、drawdown 较大，或者最终收益低于简单基准。仅继续调参容易把研究带进不稳定的模型细节里。

## 参考思路

本次更新参考了几条论文脉络：

- ReCAP: 使用 regime detection、policy library 和 regime gate 来复用不同市场状态下的策略知识。
- SAC: 通过最大熵 actor-critic 提升探索和 seed 稳定性。
- TD3: 指出 actor-critic 中 value overestimation 会导致坏策略，因此需要更稳健的 critic 和延迟更新思想。
- CQL: 强调离线/历史数据上的 distribution shift 会让 Q 值过乐观，保守评估和策略筛选很重要。

## 本次实现

新增 `src/rl_trading/baseline_policies.py`，加入不需要学习的稳健基准：

- `cash`: 全现金。
- `equal_weight`: 等权持有资产。
- `regime_anchor`: ReCAP-inspired regime anchor policy library。
- `vol_target`: 按历史波动率缩放风险暴露。

新增 `src/rl_trading/performance_gate.py`，在模型库 summary 上自动计算：

- `robust_score`
- `passes_performance_gate`
- `gate_failure_reasons`

默认 gate 条件：

- final portfolio value 不低于 0.90。
- max drawdown 不超过 0.35。
- mean turnover 不超过 1.25。

`scripts/run_rl_library.py` 现在会同时运行 RL 模型和稳健基准，并输出：

```text
outputs/rl_library/analysis/rl_library_summary.csv
outputs/rl_library/analysis/selected_policies.csv
```

## 研究意义

这让模型广场从“只看谁跑出来”变成“先过滤明显坏策略，再比较 replay 机制”。如果 DQN/SAC 的某些组合表现差，基准策略会提供最低参考线；如果 regime-aware/DEER replay 在多个模型和基准之上都能通过 gate，则更能说明研究机制具有泛化性。

下一步可以用 synthetic market level 0-4 跑系统实验，看 performance gate 过滤掉哪些坏策略，以及 `regime_anchor` / `vol_target` 是否在高波动 regime 中提供稳定 fallback。
