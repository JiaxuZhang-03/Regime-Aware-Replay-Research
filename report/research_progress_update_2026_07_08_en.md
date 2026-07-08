# Research Progress Update 2026-07-08

## Overview

Today's work continued the ablation study by separating three questions:

1. Does the model square degrade when DEER is removed?
2. Can SAC still pass the performance gate without the front-end policy-safety / regime-gate layer?
3. Does DEER + naive RL Square differ from DEER + gated RL Square enough to show that the gating method is useful?

To answer these, the code now supports a naive model-square selector, and two full-window SAC replay ablations were run. The resulting comparison covers `DEER / no-DEER / safety / no-safety / naive selection / gated selection`.

## Code Updates

`run_rl_library.py` now writes the naive selector output in addition to the existing gated selector:

```text
outputs/rl_library/analysis/selected_policies_naive.csv
outputs/rl_library/analysis/selection_comparison.csv
```

The two selectors have different roles:

- Naive selector: within each `label_method, seed` group, select the candidate with the highest `final_portfolio_value`.
- Gated selector: first keep candidates passing `passes_performance_gate`, then choose the highest `robust_score`; if no candidate passes, fall back to robust-score ranking over the full group.

A new test, `tests/test_performance_gate.py`, covers the key case where a high-return but high-drawdown candidate is selected by the naive selector, while the gated selector chooses a slightly lower-return but risk-compliant candidate.

## Ablation Design

Both main experiments used:

```text
models: sac
label methods: rule_based,hmm,recap_cusum
replays: uniform,per,regime,deer
seeds: 0,1,2
```

Each setting contains `3 x 4 x 3 = 36` full-window SAC runs.

The first setting used the defensive policy-safety / regime-gate layer:

```text
outputs/rl_library_sac_defensive_replay_ablation_20260708/
```

The second setting disabled policy safety:

```text
outputs/rl_library_sac_no_safety_replay_ablation_20260708/
```

## Main Results

### 1. Front-end policy safety is necessary

Without policy safety, every SAC run failed the performance gate because of `high_drawdown`:

| Setting | Pass Rate | Mean Final | Max Drawdown | Mean Turnover |
|---|---:|---:|---:|---:|
| no safety | 0/36 | about 1.03 | 0.4445 | about 0.379 |
| defensive safety | 36/36 | about 2.22 | 0.2309 | about 0.158 |

This indicates that the current SAC stability mainly comes from the action-level policy-safety / regime-gate layer, not from the final model-selection gate. The performance gate can detect risk, but it cannot replace risk control during policy execution.

### 2. DEER does not outperform simpler replay variants yet

Under the defensive setting, all replay variants passed the gate, but DEER had a slightly lower average final value:

| Replay | Pass Rate | Mean Final | Mean Mismatch |
|---|---:|---:|---:|
| uniform | 9/9 | 2.2244 | 0.5905 |
| per | 9/9 | 2.2225 | 0.5916 |
| regime | 9/9 | 2.2244 | 0.1388 |
| deer | 9/9 | 2.1959 | 0.5647 |

On portfolio value, DEER does not beat uniform, PER, or regime replay. On mechanism diagnostics, regime replay sharply reduces replay mismatch, while DEER remains close to uniform and PER.

The current evidence therefore favors simple regime-aware replay over the current DEER-style priority variant.

### 3. Naive RL Square vs gated RL Square

In the defensive SAC-only experiment, all candidates passed the performance gate, so the naive and gated selectors chose almost the same runs. This means that once action-level safety already controls risk, the marginal effect of the selection gate becomes small.

In the no-safety experiment, no candidate passed the gate, so the gated selector had to fall back to robust-score ranking. This shows that the selection gate correctly identifies the absence of reliable candidates.

In the existing full model-library run, `outputs/rl_library_large_20260707`, the naive selector can choose high-return but high-drawdown `equal_weight`, while the gated selector moves to passing candidates such as `regime_anchor` or `vol_target`. This confirms that the model-square performance gate is useful as a filter against high-return, high-risk candidates.

## Current Research Interpretation

The strongest current combination is not `SAC + DEER`; it is:

```text
SAC + defensive policy-safety + regime replay
```

The reasons are:

- defensive policy safety substantially reduces drawdown and turnover;
- regime replay clearly reduces replay mismatch;
- DEER passes the gate but does not improve value or mismatch over regime replay;
- the naive/gated comparison shows that the performance gate is valuable as a model-selection layer, but it should not be treated as a substitute for front-end safety.

## Output Files

Key experiment tables:

```text
outputs/rl_library_sac_defensive_replay_ablation_20260708/analysis/replay_ablation_summary_overall.csv
outputs/rl_library_sac_defensive_replay_ablation_20260708/analysis/safety_replay_ablation_overall.csv
outputs/rl_library_sac_defensive_replay_ablation_20260708/analysis/selection_comparison.csv
outputs/rl_library_sac_no_safety_replay_ablation_20260708/analysis/rl_library_summary.csv
outputs/rl_library_large_20260707/analysis/selection_comparison_ablation.csv
outputs/rl_library_large_20260707/analysis/selection_comparison_rl_ablation.csv
```

Detailed Chinese ablation report:

```text
report/rl_square_deer_ablation_2026_07_08_zh.md
```

## Verification

Tests passed after the update:

```text
/Users/littleotter/miniconda3/envs/nnenv2/bin/python -m unittest discover -s tests
Ran 6 tests, OK
```

## Suggested Next Steps

1. Treat `SAC + defensive policy-safety + regime replay` as the current main result and validate it with rolling-window or train/test split evaluation.
2. Continue improving DEER by focusing on replay mismatch and post-boundary sampling behavior, rather than only tuning portfolio return.
3. Keep gated selection in the full model library, because it prevents high-return, high-drawdown candidates from being selected; but keep the research narrative clear about the difference between action-level safety and model-selection gating.
