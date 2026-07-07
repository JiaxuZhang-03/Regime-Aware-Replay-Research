# SAC Replay Update Brief

## What Was Done

The repository was first synchronized with the latest GitHub updates. The remote branch added a DQN replay experiment framework covering online training, uniform replay, PER, regime-aware replay, and DEER-style replay. On top of that framework, this update adds a SAC version for continuous portfolio-weight training and tuning.

## SAC Changes

The existing DQN runner uses a discrete action set such as cash, single-asset allocation, and equal weight. The new SAC runner uses continuous control instead: the actor outputs continuous logits, and the environment converts them into long-only `cash + tradable assets` portfolio weights through a softmax transform. This allows the model to learn more granular allocations instead of choosing only from a small fixed action list.

New files:

- `src/rl_trading/sac_replay.py`: SAC environment, actor, twin Q critics, replay buffers, and training loop.
- `scripts/run_sac_replay.py`: SAC training entry point.
- `configs/sac_tuning_grid.json`: first-pass tuning template.

## Replay and Diagnostics

The SAC runner keeps the main replay comparisons needed by the project:

- uniform replay
- prioritized experience replay
- regime-aware replay
- DEER-style replay

Outputs still include `trading_log.csv`, `replay_diagnostics.csv`, and `summary.csv`, so experiments can compare portfolio value, drawdown, turnover, TD error, regime mismatch rate, and replay priorities.

## Training and Tuning Preparation

Quick smoke test:

```bash
python scripts/run_sac_replay.py --label-method rule_based --replays uniform --seeds 0 --warmup-steps 32 --start-steps 32 --batch-size 32 --hidden-dim 64 --max-steps 120 --output-root outputs/sac_smoke
```

First tuning pass:

```bash
python scripts/run_sac_replay.py --label-method rule_based --replays uniform,regime --seeds 0,1 --tuning-grid configs/sac_tuning_grid.json
```

Validation completed for this update: Python compile checks, existing labeler unit tests, and a short SAC smoke-training run.
