# DEER vs Uniform 固定 SAC 调优结果 2026-07-08

## 实验目的

本轮不再比较不同 RL agent，而是固定 target model：

```text
SAC + uniform replay
SAC + DEER replay
```

目标是检验 DEER replay 是否能在同一个 SAC agent 下 outperform uniform replay。

严格 split：

```text
Train:      2008-06-02 到 2020-12-31
Validation: 2021-01-04 到 2021-12-31
Test:       2022-01-03 到 2026-05-28
```

所有 DEER 参数只通过 validation 选择，test 不参与调参。

## Stage 1: 默认 robust selection

输出目录：

```text
outputs/deer_uniform_tuning_recap_20260708/
```

Validation 选中：

```text
config_id = s00p8_hl8_lam1_z5_post4
deer_s0 = 0.8
deer_half_life = 8
deer_lambda = 1.0
deer_min_post_samples = 4
selection_objective = robust
```

Final OOS paired test:

| Metric | DEER | Uniform | Delta |
|---|---:|---:|---:|
| Mean final value | 1.541507 | 1.542013 | -0.000506 |
| Mean max drawdown | 0.183316 | 0.184003 | -0.000687 |
| Mean turnover | 0.007053 | 0.007343 | -0.000290 |
| Mean robust delta | - | - | +0.000195 |
| Final-value win rate | - | - | 1/3 |
| Robust win rate | - | - | 2/3 |

Seed-level final value delta:

| Seed | DEER | Uniform | Delta |
|---:|---:|---:|---:|
| 0 | 1.535162 | 1.536028 | -0.000867 |
| 1 | 1.546293 | 1.548190 | -0.001897 |
| 2 | 1.543066 | 1.541821 | +0.001245 |

结论：DEER 在 drawdown/turnover 上略好，robust score 略好，但 final value 没有 outperform uniform。

## Stage 2: final-value selection + stronger replay impact

输出目录：

```text
outputs/deer_uniform_tuning_recap_stage2_20260708/
```

主要变化：

- `--selection-objective final`
- `--updates-per-step 2`
- `--per-alpha 0.8`
- `--deer-lambda-grid 2.0`
- `--deer-half-life-grid 8,16`
- `--deer-min-post-samples-grid 4,8`

Validation 阶段明显强于 Stage 1。所有 8 个 DEER configs 都通过 selection gate。

Validation 选中：

```text
config_id = s01_hl16_lam2_z5_post4
deer_s0 = 1.0
deer_half_life = 16
deer_lambda = 2.0
deer_min_post_samples = 4
selection_objective = final
```

Selected config 的 validation paired result：

| Metric | Value |
|---|---:|
| Mean val final delta | +0.000895 |
| Val final win rate | 2/3 |
| Mean val robust delta | +0.000917 |
| Val robust win rate | 2/3 |
| Mean train mismatch | 0.527836 |
| Mean post-boundary sample rate | 0.133580 |

Stage-2 validation top configs：

| Config | Mean Val Final Delta | Val Win Rate |
|---|---:|---:|
| `s01_hl16_lam2_z5_post4` | +0.000895 | 2/3 |
| `s01_hl8_lam2_z5_post8` | +0.000811 | 3/3 |
| `s01p5_hl16_lam2_z5_post4` | +0.000774 | 3/3 |

Final OOS paired test:

| Metric | DEER | Uniform | Delta |
|---|---:|---:|---:|
| Mean final value | 1.532987 | 1.535871 | -0.002884 |
| Mean max drawdown | 0.184754 | 0.185933 | -0.001179 |
| Mean turnover | 0.011370 | 0.010824 | +0.000546 |
| Mean robust delta | - | - | -0.001732 |
| Final-value win rate | - | - | 2/3 |
| Robust win rate | - | - | 2/3 |

Seed-level final value delta:

| Seed | DEER | Uniform | Delta |
|---:|---:|---:|---:|
| 0 | 1.524738 | 1.534172 | -0.009433 |
| 1 | 1.534769 | 1.534152 | +0.000617 |
| 2 | 1.539454 | 1.539291 | +0.000163 |

结论：Stage 2 在 validation 上确实把 DEER 相对 uniform 的收益优势放大了，但 final OOS 仍没有形成 mean final value outperform。虽然 test final win rate 是 2/3，但 seed0 的负 delta 过大，拉低了平均表现。

## 当前判断

不能 claim：

```text
DEER OOS final value outperform uniform
```

可以较谨慎地说：

```text
在固定 SAC 下，增强版 DEER replay 可以在 validation 上稳定产生 paired advantage；
但当前 selection rule 还不能把 validation advantage 稳定迁移到 final OOS mean return。
```

## 下一步建议

优先不要换 agent。下一步仍应固定 SAC target model，并改进 DEER 机制与 selection rule：

1. 将 selection 排序从 `mean_val_delta_final` 改成更稳健的 `val_win_rate_final` 优先，再看 mean delta。Stage 2 中 `s01_hl8_lam2_z5_post8` 和 `s01p5_hl16_lam2_z5_post4` 都是 3/3 validation wins，可能比当前 selected config 更稳。
2. 对 `post8` 周围做 local search，因为 `post8` 在 validation 上更稳定降低 mismatch 且胜率更高。
3. 增加 negative-tail gate，例如要求 `min_val_delta_final >= -0.0005`，避免选到 mean 高但某个 seed 明显脆弱的 config。
4. 若仍无法 OOS mean outperform，应修改 DEER priority 公式，而不是继续扩大参数网格。
