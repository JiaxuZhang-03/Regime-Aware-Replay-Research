# Research Progress Update 2026-07-07

## Overview

Since the last SAC replay update, the project has moved from single-model RL training toward a broader research framework built around mechanism diagnostics, synthetic markets, and model-library selection. The goal is no longer only to improve one DQN or SAC backtest, but to measure whether regime shifts affect learning and whether replay mechanisms improve stability across models.

## Mechanism Experiments

The remote mechanism-experiments work has been merged. The DQN runner now records regime-boundary diagnostics using fixed probe states. Around each boundary, the code can measure:

- Q-drift
- action flip rate
- Q-margin change
- TD-error shock
- post-boundary recovery

These metrics implement the plan in `mechanism_experiment_plan_0627.md`: before relying on full portfolio backtests, first show that regime shifts create measurable value-function and policy instability, then test whether DEER-style replay changes the learning process.

## Synthetic Market Foundation

The project now includes synthetic market generation tools and documentation. Synthetic levels cover stationary data, mean shifts, mean/volatility shifts, correlation shifts, and crisis regimes. This makes it possible to validate replay mechanisms under known regime ground truth before moving back to noisier real-market data.

## Model Library and Performance Gate

The local model library now includes both RL models and robust baselines:

- `cash`
- `equal_weight`
- `regime_anchor`
- `vol_target`
- `dqn`
- `sac`

`run_rl_library.py` now writes:

```text
outputs/rl_library/analysis/rl_library_summary.csv
outputs/rl_library/analysis/selected_policies.csv
```

The performance gate marks unstable policies using final portfolio value, max drawdown, and mean turnover. This prevents clearly degenerate RL runs from being treated as valid research evidence. The design is inspired by ReCAP's policy library and regime gate, and by SAC/TD3/CQL concerns around stability, overestimation, and conservative evaluation.

## Research Value

The current framework supports two research claims:

1. Mechanism level: regime shifts create measurable Q-drift, TD-error shock, and policy instability.
2. Generalization level: regime-aware and DEER-style replay can be evaluated across DQN, SAC, and robust baselines instead of relying on one model's return curve.

This makes the project more robust: the evidence can come from multiple models, multiple environments, and multiple diagnostics.

## Validation

After merging and updating, verification passed:

```text
py_compile: OK
unittest: Ran 5 tests, OK
```

The local branch now contains the performance gate/model-library update plus the merged remote mechanism-experiments work.
