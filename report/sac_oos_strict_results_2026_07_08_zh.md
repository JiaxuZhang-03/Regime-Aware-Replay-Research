# SAC 严格 OOS 重跑结果 2026-07-08

## OOS 协议

本轮修正了之前 full-window online-learning diagnostic 的问题，改成严格的训练 / 测试分离：

```text
Train: 2008-06-02 到 2020-12-31
Test:  2022-01-03 到 2026-05-28
```

测试阶段设置：

- 使用训练阶段学到的 SAC actor；
- `deterministic=True`；
- 不再加入 replay buffer；
- 不再更新 actor / critic / alpha；
- test feature normalization 使用 train window 的均值和标准差，避免 test scaler leakage。

运行命令输出：

```text
outputs/sac_oos_strict_20260708/
outputs/sac_oos_strict_20260708/analysis/sac_oos_summary.csv
outputs/sac_oos_strict_20260708/analysis/sac_oos_by_replay.csv
outputs/sac_oos_strict_20260708/analysis/sac_oos_by_label_replay.csv
```

注意：本轮使用的是前面 defensive policy-safety 配置。也就是说，训练 / 测试执行协议已经 OOS 化，但如果要完全消除配置选择偏差，下一步还应使用 validation window 只在 train/validation 上选择 safety 参数。

## OOS 总体结果

| Replay | Pass Rate | Mean Test Final | Min Test Final | Mean Test DD | Max Test DD | Mean Turnover | Train Mismatch |
|---|---:|---:|---:|---:|---:|---:|---:|
| uniform | 9/9 | 1.4004 | 1.2829 | 0.1082 | 0.1152 | 0.0196 | 0.5942 |
| per | 9/9 | 1.4007 | 1.2838 | 0.1082 | 0.1156 | 0.0196 | 0.5932 |
| regime | 9/9 | 1.4006 | 1.2837 | 0.1083 | 0.1150 | 0.0196 | 0.1387 |
| deer | 9/9 | 1.3996 | 1.2827 | 0.1083 | 0.1147 | 0.0196 | 0.5703 |

结论：

- 四种 replay 在 OOS 上表现非常接近；
- DEER 没有超过 uniform / PER / regime；
- regime replay 仍然明显降低 train replay mismatch；
- 所有 SAC OOS runs 都通过 performance gate；
- OOS 策略的主要特征是低回撤、低换手，而不是高收益。

## 按 Label Method 的 OOS 结果

| Label Method | Best Mean Test Final | Approx Range |
|---|---:|---:|
| rule_based | 1.2849 | about 1.283-1.286 |
| hmm | 1.4118 | about 1.410-1.414 |
| recap_cusum | 1.5061 | about 1.502-1.508 |

`recap_cusum` label 下收益最高，但这并不直接说明 replay method 更强，因为同一 label method 下四种 replay 的差异很小。

## Test Window Buy-and-Hold 对比

测试窗口 `2022-01-03 到 2026-05-28` 的 buy-and-hold：

| Baseline | Final Multiple | Return | Max Drawdown |
|---|---:|---:|---:|
| SPY buy-and-hold | 1.6745 | +67.4% | 0.2450 |
| QQQ buy-and-hold | 1.8804 | +88.0% | 0.3483 |
| DIA buy-and-hold | 1.4966 | +49.7% | 0.2076 |
| DIA/SPY/QQQ equal-weight, no cost | 1.6769 | +67.7% | 0.2603 |

OOS SAC 的 mean final 约 `1.40x`，低于 SPY / QQQ / equal-weight buy-and-hold，也略低于 DIA buy-and-hold 的 `1.4966x`。

但 SAC 的 OOS max drawdown 约 `0.10-0.12`，明显低于 SPY/EW/QQQ，也低于 DIA。当前结果更像是一个 defensive allocation 策略，而不是 alpha 策略。

## 当前研究判断

这次严格 OOS 修正后，结论应该改成：

1. 不能 claim SAC replay variants outperform buy-and-hold。
2. 可以 claim defensive policy-safety 显著控制了 OOS drawdown。
3. 可以 claim regime replay 在 train replay diagnostics 上显著降低 mismatch。
4. DEER 当前没有形成 OOS performance advantage，也没有像 regime replay 那样显著降低 mismatch。
5. 后续如果要写论文结果，应该把 return claim 降低，转向 `risk-controlled regime-aware replay diagnostics` 或继续改进 base policy。

## 下一步

更严格的下一步是：

```text
Train:      2008-06-02 到 2020-12-31
Validation: 2021-01-04 到 2021-12-31
Test:       2022-01-03 到 2026-05-28
```

只用 validation 选择 safety 参数和 replay 配置，然后在 test 上一次性冻结评估。这样才能避免配置选择使用未来信息。
