# Research Note: Regime-Aware Experience Replay

## 1. What This Task Is Doing

This task supports a larger research question: in non-stationary financial markets, should a reinforcement learning agent reuse all historical experience uniformly, or should it replay transitions that are more relevant to the current market regime?

Standard off-policy RL methods rely on replay buffers. A replay buffer stores past `(state, action, reward, next_state)` transitions and repeatedly samples them during training. This improves sample efficiency, but in finance it can create a specific failure mode: markets are not stationary. Bull markets, bear markets, high-volatility stress periods, and sideways regimes may imply different reward dynamics and risk exposures. If the current market is risk-off or high-volatility but the replay buffer keeps sampling many risk-on transitions, the agent may train on stale or mismatched experience.

The current stage of the project therefore does not try to build a large SAC, Transformer, or full portfolio-management system. Instead, it builds a smaller and more testable foundation: stable, interpretable, replay-friendly market-regime labels for historical data. These labels are required before we can measure replay-buffer contamination or implement a regime-aware sampler.

The repository currently implements three regime-labeling methods and exports them using a shared replay-friendly schema:

```text
date, method, regime_label, regime_name
```

The shared regime taxonomy is:

```text
0 risk_on
1 sideways
2 high_vol
3 risk_off
```

These labels can be joined to historical transitions by date. Once each transition has a regime label, later experiments can compare the current regime with the regime of the sampled replay transition.

## 2. Research Motivation

The project is not simply claiming that “markets have regimes, so policies should know the regime.” Prior work already studies policy-level regime adaptation. For example, ReCAP uses adaptive regime detection, a policy-vector library, and a regime gate to combine regime-specific policy knowledge. Other work such as DEER studies replay prioritization in general non-stationary RL environments.

This project focuses on a narrower gap: replay-buffer contamination in financial RL. The key question is whether uniform replay or prioritized experience replay continues to sample mismatched historical transitions after market-regime shifts, and whether this slows adaptation, increases drawdown, or raises turnover.

The current labeling module is designed to support the following diagnostics:

1. After the current market regime changes, does standard replay still sample many transitions from old regimes?
2. Is the sampled-regime distribution inconsistent with the current regime?
3. Are mismatched-regime samples associated with larger TD error, slower reward recovery, larger drawdown, or higher turnover?
4. Can a lightweight regime-aware replay sampler reduce mismatch and improve adaptation speed or risk stability?

## 3. Why This Stage Starts With Labels

Before building a full RL experiment, the main research risk is not model complexity. The main risk is whether the proposed failure mode exists at all. If replay contamination is not visible in simple diagnostics, then adding larger policy networks or more complex environments may not support a clear research contribution.

For this reason, the current stage is intentionally scoped to:

- compress market history into a date-level regime timeline;
- use multiple labelers to cross-check the stability of regime assignments;
- export a consistent format that can be joined to replay-buffer transitions;
- verify replay mismatch before expanding into full sampler or policy-training experiments.

In other words, this stage is the measurement layer of the project. It does not directly claim to improve returns. It creates the observable variables needed to test whether the replay mechanism is problematic under market-regime shifts.

## 4. Data and Inputs

The default input file is:

```text
data/market_indices_20080601_20260531/market_regime_features_wide.csv
```

This file covers 4528 trading days from 2008-06-02 through 2026-05-29. The feature builder normalizes the input into one market-level row per date. The main features are:

- `ret_short`: short-horizon return feature;
- `ret_long`: medium/long-horizon return feature;
- `vol`: rolling volatility;
- `trend`: trend or price-vs-moving-average feature;
- `vix`: VIX or a similar market-risk indicator;
- `turbulence`: market-turbulence feature, defaulting to zero if unavailable.

The code also supports long-panel inputs such as `DOW30_recap_features.csv`. If multiple tickers are present, the feature builder first tries to use the primary symbol `SPY`; otherwise it aggregates numeric features into a market-level time series.

## 5. Three Regime-Labeling Methods

### 5.1 Rule-Based Trend/Volatility Labeler

The first method is a transparent rule-based baseline. It uses rolling returns, trend, volatility, and VIX thresholds:

- `risk_on`: positive trend and positive longer-horizon return, without high volatility;
- `high_vol`: volatility or VIX above a rolling quantile, without a clear negative trend;
- `risk_off`: negative trend or negative longer-horizon return;
- `sideways`: fallback state when no stronger condition applies.

This method is fast, interpretable, and useful as a sanity check. The rolling thresholds are shifted by one day to avoid future leakage when labels are later used in walk-forward diagnostics.

### 5.2 Gaussian HMM Labeler

The second method is a classical latent-regime baseline. It fits a diagonal-covariance Gaussian HMM to market features, trains the model with EM, and decodes hard labels with Viterbi. It also exports posterior probabilities for each ordered label.

The HMM captures latent market regimes that are not directly observed. After training, hidden states are ordered using the following risk score:

```text
return + trend - volatility - VIX
```

This mapping makes the anonymous HMM states interpretable as regimes such as `risk_off`, `sideways`, and `risk_on`, and avoids manual relabeling after each run.

### 5.3 ReCAP-Inspired CUSUM/ARD Labeler

The third method borrows the adaptive-regime-detection idea from ReCAP, but it is not a full reproduction of ReCAP. Here, the method only uses CUSUM-style change detection to segment market-level features.

This method:

- runs symmetric CUSUM statistics over market features;
- detects change points;
- produces variable-length `segment_id` values;
- maps each segment to the shared taxonomy using segment-level return, trend, volatility, and VIX summaries.

Its role is to provide a literature-aware adaptive segmentation baseline. It pays more attention to the timing of regime switches than fixed rolling rules, while remaining much lighter than a full policy-level continual-learning system.

## 6. Current Outputs

The generated labels are stored in:

```text
outputs/regime_labels/
```

The main files are:

- `rule_based_labels.csv`
- `hmm_labels.csv`
- `recap_cusum_labels.csv`
- `all_regime_labels.csv`
- `label_summary.csv`
- `label_switches.csv`
- one metadata JSON file per method

The current summary shows that the labelers do not produce identical regime distributions. This is useful: multiple labelers help distinguish robust regime signals from method-specific sensitivity.

Current output summary:

| Method | Main distribution | Label switches |
| --- | --- | ---: |
| rule_based | risk_on 58.46%, risk_off 28.09%, high_vol 13.32%, sideways 0.13% | 366 |
| hmm | sideways 39.77%, risk_on 33.81%, risk_off 26.41% | 97 |
| recap_cusum | risk_on 62.68%, risk_off 27.27%, high_vol 10.05% | 40 |

This pattern is consistent with the nature of the methods. The rule-based labeler reacts more often to rolling thresholds, so it switches more frequently. The HMM produces a smoother latent-state sequence. The CUSUM/ARD method explicitly detects change points, so it generates fewer segments.

## 7. How These Labels Feed Replay Experiments

In later replay experiments, each transition can obtain a regime label through its date:

```text
transition.date -> regime_label
```

During training, the current date can also be mapped to the current regime:

```text
current_date -> current_regime_label
```

With these two variables, replay mismatch can be defined as:

```text
mismatch = sampled_transition_regime != current_regime
```

This enables the following diagnostics:

- sampled regime distribution;
- mismatch rate between current regime and sampled-transition regime;
- mismatch rate before and after regime switches;
- TD error by current/sampled regime pair;
- reward recovery time after regime switches;
- drawdown after regime switches;
- turnover spikes after regime switches.

If uniform replay or PER keeps sampling many mismatched transitions after regime switches, and a regime-aware replay sampler reduces mismatch while improving at least one adaptation or risk metric, then the project has a clear contribution: diagnosing replay-buffer contamination under financial regime shifts and proposing a lightweight regime-aware sampling correction.

## 8. Minimal Regime-Aware Replay Sampler

The next sampler should initially remain simple. A minimal version could use:

```text
50% same-or-similar regime transitions
50% normal replay transitions
```

Same-or-similar can first be defined using discrete labels. A later version could treat `risk_off` and `high_vol` as related risk states. This design keeps the experiment interpretable: if only the sampling distribution changes while the policy network, reward function, and transaction-cost assumptions remain fixed, performance differences are easier to attribute to the replay mechanism.

The main baselines should be:

- uniform replay;
- prioritized experience replay;
- sliding-window replay;
- regime-aware replay.

These baselines answer different questions. Uniform replay is the standard baseline. PER tests whether TD-error priority is enough. Sliding-window replay tests whether recency alone is enough. Regime-aware replay tests whether historical experience should be selected according to market state.

## 9. Current Scope Boundaries

The current repository implements regime labeling and basic label diagnostics. It is not yet a complete RL trading system. The following components are not yet implemented:

- DQN/SAC policy-training loop;
- unified experiment framework for replay-buffer variants;
- portfolio backtest under transaction costs;
- joint analysis of replay mismatch, TD error, drawdown, and turnover.

This is an intentional research sequence rather than a weakness. Building the regime timeline and mismatch observables first prevents the project from prematurely spending effort on complex models before the replay-contamination hypothesis is measurable.

## 10. Repository Mapping

The main repository files behind this report are:

- `README.md`: project overview, quick start, and output description;
- `docs/regime_labeling_notes.md`: regime-labeling method notes and literature positioning;
- `src/regime_labeling/features.py`: converts wide or long-panel input into market-level features;
- `src/regime_labeling/rule_based.py`: rule-based trend/volatility labeler;
- `src/regime_labeling/hmm.py`: Gaussian HMM labeler;
- `src/regime_labeling/recap_ard.py`: ReCAP-inspired CUSUM/ARD labeler;
- `src/regime_labeling/diagnostics.py`: label distribution and switch-count summaries;
- `scripts/make_regime_labels.py`: local entry point for generating all labels;
- `outputs/regime_labels/`: generated labels, summaries, switch counts, and metadata.

Together, these files do not yet implement the final investment policy. They convert financial history into a regime-indexed experience dataset that a replay buffer can use.

## 11. Next Steps

Recommended next steps:

1. Join `all_regime_labels.csv` to transitions generated by the RL environment.
2. Log sampled-transition regimes for uniform replay and PER.
3. Export count matrices for current/sampled regime pairs and mismatch rates.
4. First validate the phenomenon in a synthetic bull/bear/high-volatility environment.
5. Move the same diagnostics to ETF or DOW30 portfolio data.
6. Implement the minimal regime-aware sampler and compare it with uniform replay, PER, and sliding-window replay.
7. If the mismatch diagnostic is clear, then extend to richer policy networks or soft-regime representations.

## 12. One-Sentence Summary

This task builds the first evidence layer for “regime-aware experience replay for financial non-stationarity”: it converts financial history into comparable market regimes and uses those labels to test whether replay buffers keep reusing stale historical experience after regime shifts. The current output is not a final trading policy; it is the labeling and measurement foundation needed for replay-contamination diagnostics and regime-aware sampling experiments.
