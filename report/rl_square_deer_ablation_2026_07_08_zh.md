# RL Square / DEER Ablation 结果更新

日期：2026-07-08

## 实验设置

本轮做了两组 full-window SAC ablation，每组都是：

- models: `sac`
- label methods: `rule_based,hmm,recap_cusum`
- replays: `uniform,per,regime,deer`
- seeds: `0,1,2`
- 共 `3 x 4 x 3 = 36` 个 SAC run

第一组使用 defensive policy-safety / regime gate：

```text
outputs/rl_library_sac_defensive_replay_ablation_20260708/
```

第二组关闭前置 policy-safety：

```text
outputs/rl_library_sac_no_safety_replay_ablation_20260708/
```

同时，`run_rl_library.py` 现在会额外输出朴素模型广场选择：

```text
selected_policies_naive.csv
selection_comparison.csv
```

其中 naive selector 只按 `final_portfolio_value` 选；gated selector 先看 `passes_performance_gate`，再按 `robust_score` 选。

## 总体结果

### 1. 不加前置 policy-safety 时，SAC 全部过不了 gate

| setting | replay | pass rate | mean final | max drawdown | mean turnover | mean mismatch |
|---|---:|---:|---:|---:|---:|---:|
| no safety | uniform | 0/9 | 1.0399 | 0.4445 | 0.3789 | 0.5905 |
| no safety | per | 0/9 | 1.0371 | 0.4445 | 0.3791 | 0.5924 |
| no safety | regime | 0/9 | 1.0397 | 0.4445 | 0.3789 | 0.1392 |
| no safety | deer | 0/9 | 0.9954 | 0.4445 | 0.3794 | 0.5646 |

关闭 safety 后，所有 run 都因为 `high_drawdown` 被 performance gate 拦掉。说明当前 SAC 的主要稳定性来源不是 DEER，而是前置 policy-safety / regime gate。

### 2. 加 defensive policy-safety 后，所有 replay 都能通过 gate

| setting | replay | pass rate | mean final | max drawdown | mean turnover | mean mismatch |
|---|---:|---:|---:|---:|---:|---:|
| defensive | uniform | 9/9 | 2.2244 | 0.2309 | 0.1576 | 0.5905 |
| defensive | per | 9/9 | 2.2225 | 0.2309 | 0.1576 | 0.5916 |
| defensive | regime | 9/9 | 2.2244 | 0.2309 | 0.1576 | 0.1388 |
| defensive | deer | 9/9 | 2.1959 | 0.2309 | 0.1573 | 0.5647 |

defensive safety 把 drawdown 从约 `0.44` 降到约 `0.23`，turnover 从约 `0.38` 降到约 `0.16`，并把所有 run 推到 gate 通过区间。

### 3. DEER 目前没有超过更简单的 replay

在 defensive setting 下，DEER 通过了 gate，但平均 final value 低于 `uniform/per/regime`：

- `deer`: mean final `2.1959`
- `uniform`: mean final `2.2244`
- `per`: mean final `2.2225`
- `regime`: mean final `2.2244`

更重要的是，DEER 的 mismatch rate 仍接近 uniform/PER，而 regime replay 明显降低 mismatch：

- `regime`: mean mismatch `0.1388`
- `deer`: mean mismatch `0.5647`
- `uniform`: mean mismatch `0.5905`
- `per`: mean mismatch `0.5916`

所以当前证据更支持：简单 regime-aware replay 在机制诊断上比当前 DEER 变体更清楚。

### 4. 朴素 RL Square vs gated RL Square

在 defensive SAC-only setting 中，所有 candidate 都通过 gate，因此 naive selector 和 gated selector 基本选出同一批 run。这个结果说明：当前置 safety 已经把风险压平后，selection gate 的边际作用变小。

在无 safety setting 中，所有 candidate 都没通过 gate，因此 gated selector 只能 fallback 到 robust score；这说明 selection gate 能识别风险问题，但不能替代 action-level policy-safety。

在已有全模型库 `outputs/rl_library_large_20260707` 中，naive selector 会因为 final value 最高而选到高回撤 `equal_weight`，但 gated selector 会换成通过 gate 的 `regime_anchor` 或 `vol_target`。这说明模型广场层面的 performance gate 是有用的，但它主要负责过滤 bad candidates，不是提升一个已经稳定 candidate 的收益。

## 当前判断

1. 不加 DEER 并不会损害这组 SAC defensive 结果；`uniform/per/regime` 都略好于 `deer`。
2. 不加前置 policy-safety 时，SAC 全部因为 drawdown 失败，说明前置 safety/regime gate 是必要组件。
3. DEER + 朴素 RL Square 与 DEER + gated RL Square 的差异不明显，因为 defensive setting 下 DEER 本身已经通过 gate，但不是最优 replay。
4. 目前最强、最可解释的组合是 `SAC + defensive policy-safety + regime replay`，而不是 `SAC + DEER`。

## 输出文件

```text
outputs/rl_library_sac_defensive_replay_ablation_20260708/analysis/replay_ablation_summary_overall.csv
outputs/rl_library_sac_defensive_replay_ablation_20260708/analysis/safety_replay_ablation_overall.csv
outputs/rl_library_sac_defensive_replay_ablation_20260708/analysis/selection_comparison.csv
outputs/rl_library_sac_no_safety_replay_ablation_20260708/analysis/rl_library_summary.csv
outputs/rl_library_large_20260707/analysis/selection_comparison_ablation.csv
outputs/rl_library_large_20260707/analysis/selection_comparison_rl_ablation.csv
```
