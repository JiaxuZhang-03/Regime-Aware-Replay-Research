# 研究进展更新 2026-07-07

## 本次更新概览

自上次 SAC replay 更新报告以来，项目从“单个 RL 模型训练”推进到“机制诊断 + synthetic market + 多模型库筛选”的研究框架。当前重点不再只是让某一次 DQN 或 SAC 回测表现更好，而是先确保 replay 机制、regime shift 影响和策略稳定性能够被系统测量。

## 新增机制实验

远端合并了 mechanism-experiments 工作，DQN runner 增加了围绕 regime boundary 的机制诊断输出。每次 regime 切换后，代码会基于固定 probe states 记录：

- Q-drift
- action flip rate
- Q-margin change
- TD-error shock
- post-boundary recovery

这些指标对应 `mechanism_experiment_plan_0627.md` 中的设计目标：在 full backtest 之前先证明 regime shift 会造成可观测的 value function / policy instability，并观察 DEER-style replay 是否能改变学习过程。

## Synthetic Market 实验基础

项目新增了 synthetic market 生成脚本和文档，用于构造可控的非平稳市场环境。Synthetic levels 覆盖 stationary、mean shift、mean/vol shift、correlation shift 和 crisis regime 等情形。这个模块的作用是把真实市场噪声和机制验证分开，先在已知 regime ground truth 的环境中测试 replay 方法。

## 模型库与 Performance Gate

本地新增了模型库的 baseline 和 performance gate：

- `cash`
- `equal_weight`
- `regime_anchor`
- `vol_target`
- `dqn`
- `sac`

`run_rl_library.py` 现在会同时运行 RL 模型和稳健基准，并输出：

```text
outputs/rl_library/analysis/rl_library_summary.csv
outputs/rl_library/analysis/selected_policies.csv
```

Performance gate 会根据 final portfolio value、max drawdown 和 mean turnover 标记坏策略，避免把明显退化的 RL run 当成有效研究结果。这个设计参考了 ReCAP 的 policy library / regime gate 思路，也借鉴了 SAC、TD3、CQL 中关于稳定性、过估计和保守评估的思想。

## 当前研究意义

当前框架可以支持两个层面的研究叙事：

1. 机制层面：regime shift 是否造成 Q-drift、TD-error shock 和 policy instability。
2. 泛化层面：regime-aware / DEER replay 是否能在 DQN、SAC 和稳健 baseline 对照下保持更稳定的样本选择与策略表现。

这使得项目不再依赖某一个模型的一次收益曲线，而是通过多模型、多环境、多指标来验证 replay 机制。

## 验证状态

本次合并和更新后已通过：

```text
py_compile: OK
unittest: Ran 5 tests, OK
```

当前本地包含 performance gate、本地模型库改动，以及远端 mechanism-experiments 的合并结果。
