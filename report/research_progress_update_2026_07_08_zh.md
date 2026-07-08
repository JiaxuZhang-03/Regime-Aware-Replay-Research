# 研究进展更新 2026-07-08

## 本次更新概览

今天的重点是继续做 ablation test，把三个问题拆开验证：

1. 不加入 DEER 时，模型广场表现是否明显下降。
2. 不加入前置 policy-safety / regime gate 时，SAC 是否还能稳定通过 performance gate。
3. DEER + 朴素 RL Square 与 DEER + gated RL Square 的差别，是否能说明 gate 的方式有效。

为此，代码层面补充了 naive model-square selector；实验层面跑了两组 full-window SAC replay ablation，并整理了 `DEER / no-DEER / safety / no-safety / naive selection / gated selection` 的对照结果。

## 代码更新

`run_rl_library.py` 现在除了原来的 gated selection 外，还会输出朴素选择结果：

```text
outputs/rl_library/analysis/selected_policies_naive.csv
outputs/rl_library/analysis/selection_comparison.csv
```

其中：

- naive selector：每个 `label_method, seed` 组内，只按 `final_portfolio_value` 选择最高的 candidate。
- gated selector：先过滤通过 `passes_performance_gate` 的 candidate，再按 `robust_score` 选择；如果无人通过，则 fallback 到全组 robust score。

新增测试 `tests/test_performance_gate.py` 覆盖了一个关键场景：高收益但高回撤的 candidate 会被 naive 选中，但 gated selector 会选择收益略低、风险合格的 candidate。

## Ablation 设计

本轮两组主实验均使用：

```text
models: sac
label methods: rule_based,hmm,recap_cusum
replays: uniform,per,regime,deer
seeds: 0,1,2
```

每组共有 `3 x 4 x 3 = 36` 个 full-window SAC run。

第一组使用 defensive policy-safety / regime gate：

```text
outputs/rl_library_sac_defensive_replay_ablation_20260708/
```

第二组关闭前置 policy-safety：

```text
outputs/rl_library_sac_no_safety_replay_ablation_20260708/
```

## 主要结果

### 1. 前置 policy-safety 是必要组件

关闭 policy-safety 后，所有 SAC run 都因为 `high_drawdown` 没有通过 performance gate：

| Setting | Pass Rate | Mean Final | Max Drawdown | Mean Turnover |
|---|---:|---:|---:|---:|
| no safety | 0/36 | about 1.03 | 0.4445 | about 0.379 |
| defensive safety | 36/36 | about 2.22 | 0.2309 | about 0.158 |

这说明当前 SAC 的稳定性主要来自前置 action-level policy-safety / regime gate，而不是后置 selection gate。后置 performance gate 可以识别风险，但不能替代策略执行阶段的风险控制。

### 2. DEER 当前没有超过更简单的 replay

在 defensive setting 下，四种 replay 都通过 gate，但 DEER 的平均 final value 略低：

| Replay | Pass Rate | Mean Final | Mean Mismatch |
|---|---:|---:|---:|
| uniform | 9/9 | 2.2244 | 0.5905 |
| per | 9/9 | 2.2225 | 0.5916 |
| regime | 9/9 | 2.2244 | 0.1388 |
| deer | 9/9 | 2.1959 | 0.5647 |

从收益看，DEER 没有超过 uniform / PER / regime replay。从机制诊断看，regime replay 显著降低 mismatch，而 DEER 的 mismatch 仍接近 uniform 和 PER。

这支持一个比较清晰的判断：当前版本的 DEER-style priority 在金融 replay 任务中还没有形成稳定优势；简单 regime-aware replay 更可解释。

### 3. 朴素 RL Square 与 gated RL Square 的差异

在 defensive SAC-only 实验中，所有 candidate 都通过 performance gate，所以 naive selector 和 gated selector 基本选出同样的 run。这说明当 action-level safety 已经把风险压平后，selection gate 的边际作用会变小。

在 no-safety 实验中，所有 candidate 都没通过 gate，因此 gated selector 只能 fallback 到 robust score。这说明 selection gate 能正确发现“没有可靠候选”的情况。

在已有全模型库实验 `outputs/rl_library_large_20260707` 中，naive selector 会因为收益最高而选到高回撤的 `equal_weight`，而 gated selector 会转向通过 gate 的 `regime_anchor` 或 `vol_target`。这说明模型广场层面的 performance gate 是有价值的：它主要防止高收益但高风险的 candidate 被误选。

## 当前研究判断

当前最可靠的组合不是 `SAC + DEER`，而是：

```text
SAC + defensive policy-safety + regime replay
```

理由是：

- defensive policy-safety 明显降低 drawdown 和 turnover；
- regime replay 显著降低 replay mismatch；
- DEER 当前虽然能通过 gate，但收益和 mismatch 都没有超过 regime replay；
- naive/gated comparison 表明 performance gate 适合作为模型广场筛选层，而不是替代前置 safety layer。

## 输出文件

关键实验表格：

```text
outputs/rl_library_sac_defensive_replay_ablation_20260708/analysis/replay_ablation_summary_overall.csv
outputs/rl_library_sac_defensive_replay_ablation_20260708/analysis/safety_replay_ablation_overall.csv
outputs/rl_library_sac_defensive_replay_ablation_20260708/analysis/selection_comparison.csv
outputs/rl_library_sac_no_safety_replay_ablation_20260708/analysis/rl_library_summary.csv
outputs/rl_library_large_20260707/analysis/selection_comparison_ablation.csv
outputs/rl_library_large_20260707/analysis/selection_comparison_rl_ablation.csv
```

更详细的中文 ablation 报告：

```text
report/rl_square_deer_ablation_2026_07_08_zh.md
```

## 验证状态

本次更新后测试通过：

```text
/Users/littleotter/miniconda3/envs/nnenv2/bin/python -m unittest discover -s tests
Ran 6 tests, OK
```

## 下一步建议

1. 把 `SAC + defensive policy-safety + regime replay` 作为当前主线结果，用 rolling-window 或 train/test split 做 out-of-sample 验证。
2. 继续改 DEER：重点不是再调收益，而是让 DEER 的 replay mismatch 和 post-boundary sample behavior 真正区别于 uniform/PER。
3. 在全模型库里保留 gated selection，因为它能避免高收益高回撤基准被误选；但研究叙事中要明确区分 action-level safety 和 model-selection gate。
