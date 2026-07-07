# SAC 风险控制与 Regime Replay 调参记录

本轮目标不是继续追求单次收益最大化，而是先让 SAC 在模型广场里稳定通过 performance gate，尤其是把最大回撤控制到 `0.35` 以下，同时保持 final portfolio value 高于 `0.90`。

## 实验设计

本次新增配置文件：

```text
configs/sac_risk_control_grid_20260707.json
```

它包含 6 组 SAC 风控候选，主要调整：

- `safety_min_cash_weight`
- `safety_max_asset_weight`
- `safety_max_turnover`
- `safety_regime_blend`
- `safety_risk_on_cash / sideways_cash / high_vol_cash / risk_off_cash`
- `action_temperature`
- `regime_same_ratio / high_td_ratio / recent_ratio / random_ratio`

正式调参命令使用完整样本窗口、3 个 label 方法和 3 个 seed：

```bash
python scripts/run_sac_replay.py \
  --label-method rule_based,hmm,recap_cusum \
  --replays regime \
  --seeds 0,1,2 \
  --warmup-steps 256 \
  --start-steps 256 \
  --batch-size 64 \
  --hidden-dim 128 \
  --tuning-grid configs/sac_risk_control_grid_20260707.json \
  --output-root outputs/sac_risk_control_tuning_20260707
```

汇总输出：

```text
outputs/sac_risk_control_tuning_20260707/analysis/sac_risk_control_grid_all_runs.csv
outputs/sac_risk_control_tuning_20260707/analysis/sac_risk_control_grid_by_config.csv
outputs/sac_risk_control_tuning_20260707/analysis/sac_risk_control_grid_selected.csv
```

## 主要结果

除最轻量的 `risk_grid_000_balanced` 外，其余 5 组配置全部 `9/9` 通过 performance gate。

最佳配置是：

```text
risk_grid_004_defensive
```

跨 3 个 label 方法和 3 个 seed 的平均结果：

| 配置 | 平均 final value | 最低 final value | 平均 max drawdown | 最差 max drawdown | 平均 turnover | 平均 cash | 通过数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| risk_grid_004_defensive | 2.2244 | 1.6610 | 0.2180 | 0.2309 | 0.1576 | 0.4371 | 9/9 |
| risk_grid_002_anchor | 2.0822 | 1.4914 | 0.2613 | 0.2773 | 0.2181 | 0.3549 | 9/9 |
| risk_grid_005_lowalpha | 1.8645 | 1.4014 | 0.2746 | 0.2892 | 0.2355 | 0.3592 | 9/9 |
| risk_grid_003_cashfloor | 1.6922 | 1.3037 | 0.2941 | 0.3074 | 0.2544 | 0.3562 | 9/9 |
| risk_grid_001_turnover | 1.6492 | 1.3382 | 0.3229 | 0.3388 | 0.2790 | 0.3101 | 9/9 |
| risk_grid_000_balanced | 1.2853 | 1.0329 | 0.3568 | 0.3715 | 0.3405 | 0.2904 | 2/9 |

## 模型广场验证

随后用最佳配置重新跑 `run_rl_library.py`，让模型广场的 gate 直接验证 SAC-regime：

```bash
python scripts/run_rl_library.py \
  --models sac \
  --label-method rule_based,hmm,recap_cusum \
  --replays regime \
  --seeds 0,1,2 \
  --warmup-steps 256 \
  --sac-start-steps 256 \
  --batch-size 64 \
  --hidden-dim 128 \
  --sac-action-temperature 1.5 \
  --safety-min-cash-weight 0.25 \
  --safety-max-asset-weight 0.45 \
  --safety-max-turnover 0.35 \
  --safety-regime-blend 0.60 \
  --safety-risk-on-cash 0.30 \
  --safety-sideways-cash 0.60 \
  --safety-high-vol-cash 0.85 \
  --safety-risk-off-cash 0.95 \
  --regime-same-ratio 0.75 \
  --regime-high-td-ratio 0.10 \
  --regime-recent-ratio 0.10 \
  --regime-random-ratio 0.05 \
  --output-root outputs/rl_library_sac_defensive_20260707
```

模型广场输出：

```text
outputs/rl_library_sac_defensive_20260707/analysis/rl_library_summary.csv
outputs/rl_library_sac_defensive_20260707/analysis/selected_policies.csv
```

结果是 `9/9` 全部通过 gate。最差 seed 的 max drawdown 只有 `0.2309`，明显低于 `0.35`；最低 final value 是 `1.6610`，也明显高于 `0.90`。

## 研究含义

这一步说明当前问题不只是 SAC 学不到策略，而是默认风险层太松，导致 drawdown gate 过不了。加入更强的 ReCAP-style regime anchor、现金比例、单资产上限和 turnover cap 后，SAC-regime replay 可以在三种 label 方法和多 seed 下稳定通过 gate。

这也支持当前研究的泛化方向：不是依赖某一个 RL 模型裸跑出好结果，而是通过模型无关的 safety layer 和 regime-aware replay，降低不同 RL 策略学到坏策略的概率。

下一步建议：

1. 用 `risk_grid_004_defensive` 作为 SAC-regime 的默认稳健配置。
2. 对同一配置补跑 SAC `uniform/per/deer`，确认性能提升来自 regime replay 还是主要来自 safety layer。
3. 做 train/test split 或 rolling-window validation，避免完整窗口内的策略保护效果被误读为真实 out-of-sample alpha。
