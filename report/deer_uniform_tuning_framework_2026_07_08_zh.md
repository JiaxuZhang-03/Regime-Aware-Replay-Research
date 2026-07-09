# DEER vs Uniform 固定目标模型调优框架 2026-07-08

## 核心判断

下一步不再做不同 RL agent 之间的模型池比较。若论文主线要突出 DEER 的作用，实验应固定一个 target model，然后只比较 replay mechanism：

```text
Target model: SAC portfolio policy
Comparison:   SAC + uniform replay vs SAC + DEER replay
```

这样可以避免 PPO/SAC/其他 agent 的架构差异掩盖 DEER replay 本身的贡献。

## 严格数据协议

所有调优都必须使用 train/validation/test 三段：

```text
Train:      2008-06-02 到 2020-12-31
Validation: 2021-01-04 到 2021-12-31
Test:       2022-01-03 到 2026-05-28
```

规则：

- 只在 train 上训练 SAC；
- validation 阶段冻结策略，不写 replay，不更新 actor/critic/alpha；
- 只用 validation 选择 DEER replay 参数；
- test 只在 validation 选出的最终 DEER config 上跑一次；
- validation/test feature scaler 都使用 train window 的均值和标准差。

## 固定项

为保证比较只反映 replay 差异，下列部分在 uniform 和 DEER 之间保持完全一致：

- SAC actor/critic 架构；
- seed 列表；
- transaction cost；
- action temperature；
- policy safety 参数；
- label method，当前默认 `recap_cusum`；
- tradable universe，当前默认 `DIA,SPY,QQQ`；
- 训练步数、batch size、learning rate、gamma/tau 等 SAC 超参。

当前 runner 默认使用偏 growth 的 safety profile，因为之前 defensive profile OOS 回撤低但收益空间不足。

## 可调项

只调 DEER replay 参数：

| 参数 | 作用 |
|---|---|
| `deer_s0` | regime boundary 后对新分布样本的初始强调强度 |
| `deer_half_life` | boundary 强调随新样本年龄衰减的速度 |
| `deer_lambda` | DOE/Q-discrepancy 对 priority 的影响强度 |
| `deer_zmax` | 标准化 TD/DOE 的截断上限 |
| `deer_min_post_samples` | 每个 batch 中 boundary 后样本的最低强制采样数 |

默认搜索网格：

```text
deer_s0:               0.8, 1.2
deer_half_life:        3, 8
deer_lambda:           1.0, 2.0
deer_zmax:             5.0
deer_min_post_samples: 4, 8
```

共 16 个 DEER configs，默认 3 个 seeds，因此 tune 阶段是 `3 uniform + 48 DEER` 个 train/val runs。

## 选择指标

每个 DEER config 都和同 seed 的 uniform baseline 做 paired comparison。

脚本支持两种 validation selection objective：

```text
--selection-objective robust        # 默认，按 robust_score delta 排序
--selection-objective final         # 按 final_portfolio_value delta 排序
--selection-objective stable_final  # 先按 final-value win rate 排序，再按 mean final delta 排序
```

Validation robust score:

```text
robust_score = final_portfolio_value - max_drawdown - 0.05 * mean_turnover
```

默认选择 gate：

- `val_win_rate_robust >= 2/3`；
- `mean_val_delta_robust >= 0`；
- `mean_val_delta_final >= 0`；
- `mean_deer_val_dd <= mean_uniform_val_dd + 0.03`。

若没有 config 通过 gate，脚本会 fallback 到 validation mean robust delta 最高的 config，并在结果表中标记 `selection_used_fallback=True`。

## 已实现入口

脚本：

```text
scripts/run_deer_uniform_tuning.py
```

正式运行命令：

```bash
/Users/littleotter/miniconda3/envs/nnenv2/bin/python scripts/run_deer_uniform_tuning.py \
  --label-method recap_cusum \
  --seeds 0,1,2 \
  --output-root outputs/deer_uniform_tuning_recap_20260708
```

若目标是优先最大化 DEER 相对 uniform 的收益差异，可使用：

```bash
/Users/littleotter/miniconda3/envs/nnenv2/bin/python scripts/run_deer_uniform_tuning.py \
  --label-method recap_cusum \
  --seeds 0,1,2 \
  --selection-objective final \
  --output-root outputs/deer_uniform_tuning_recap_final_select_20260708
```

若要优先选择 seed-level 稳定性，而不是只看 mean delta，可使用：

```bash
/Users/littleotter/miniconda3/envs/nnenv2/bin/python scripts/run_deer_uniform_tuning.py \
  --label-method recap_cusum \
  --seeds 0,1,2 \
  --selection-objective stable_final \
  --output-root outputs/deer_uniform_tuning_recap_stable_final_20260708
```

快速 smoke：

```bash
/Users/littleotter/miniconda3/envs/nnenv2/bin/python scripts/run_deer_uniform_tuning.py \
  --seeds 0 \
  --max-configs 1 \
  --train-max-steps 40 \
  --val-max-steps 10 \
  --test-max-steps 10 \
  --output-root outputs/deer_uniform_tuning_smoke_20260708
```

## 输出文件

主要结果在：

```text
outputs/deer_uniform_tuning_*/analysis/
```

关键表：

- `uniform_tune_summary.csv`
- `deer_tune_summary.csv`
- `deer_vs_uniform_val_paired.csv`
- `deer_tune_by_config.csv`
- `selected_deer_config.csv`
- `uniform_final_oos_summary.csv`
- `deer_final_oos_summary.csv`
- `deer_vs_uniform_final_oos_paired.csv`
- `final_oos_comparison_summary.csv`

## 研究目标

这套框架的目标不是证明“某个 RL agent 更好”，而是证明：

```text
在同一个 SAC target model 下，经过 validation-only 调优的 DEER replay
可以在严格 OOS 中超过 uniform replay。
```

如果 test 上 DEER 未超过 uniform，则下一步应继续围绕 DEER replay 参数、boundary definition、DOE scale、以及 post-boundary sampling 强度调优，而不是切换到另一个 RL agent。
