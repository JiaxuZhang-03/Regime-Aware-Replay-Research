# SAC Replay 更新简报

## 本次完成内容

本次先同步了 GitHub 仓库最新内容。远端新增了 DQN replay 实验框架，包括 online、uniform replay、PER、regime-aware replay 和 DEER-style replay。同步后，本地在该框架基础上新增了 SAC 版本，用于后续连续投资组合权重训练和调优。

## SAC 改造要点

原有 DQN 使用离散动作集合，例如 cash、单资产满仓、equal-weight。新增 SAC 版本改为连续动作控制：actor 输出连续 logits，环境通过 softmax 将其转换为 `cash + tradable assets` 的 long-only portfolio weights。这样模型可以学习更细粒度的资产配置，而不是只能在少数固定组合之间切换。

新增文件包括：

- `src/rl_trading/sac_replay.py`: SAC 环境、actor、双 Q critic、replay buffers 和训练循环。
- `scripts/run_sac_replay.py`: SAC 训练入口。
- `configs/sac_tuning_grid.json`: 第一轮调参模板。

## Replay 与诊断

SAC runner 保留了当前研究的核心 replay 对比：

- uniform replay
- prioritized experience replay
- regime-aware replay
- DEER-style replay

训练输出继续包含 `trading_log.csv`、`replay_diagnostics.csv` 和 `summary.csv`，便于比较 portfolio value、drawdown、turnover、TD-error、regime mismatch rate 和 replay priority 等指标。

## 训练与调优准备

可以用以下命令进行快速 smoke test：

```bash
python scripts/run_sac_replay.py --label-method rule_based --replays uniform --seeds 0 --warmup-steps 32 --start-steps 32 --batch-size 32 --hidden-dim 64 --max-steps 120 --output-root outputs/sac_smoke
```

可以用以下命令启动第一轮调参：

```bash
python scripts/run_sac_replay.py --label-method rule_based --replays uniform,regime --seeds 0,1 --tuning-grid configs/sac_tuning_grid.json
```

本次验证已通过 Python 编译检查、原有 labeler 单元测试，以及 SAC 短训练 smoke run。
