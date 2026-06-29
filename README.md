# Regime-Aware Replay Research

This repo supports the first two-week validation in `ProjectPlan`: before adding
larger RL machinery, test whether standard replay mixes incompatible market
regimes and whether lightweight regime-aware sampling can reduce that mismatch.

The current scope is intentionally small: produce at most three market-regime
labeling methods that can tag replay-buffer transitions.

## Repo Layout

- `ProjectPlan/`: meeting brief and proposal documents.
- `data/`: existing market index and DOW30 feature CSVs.
- `src/regime_labeling/`: reusable Python package for regime labels.
- `scripts/make_regime_labels.py`: local CLI wrapper.
- `docs/regime_labeling_notes.md`: method notes and literature anchors.
- `tests/`: smoke tests for labeler behavior.
- `outputs/`: generated labels and summaries, ignored by git.

## Label Methods

1. `rule_based`: transparent rolling trend/volatility labels. This is the fast
   baseline from the project plan: bull/risk-on, sideways, high-vol, risk-off.
2. `hmm`: diagonal Gaussian HMM over market-level return, trend, volatility, and
   VIX features. This is the classical latent-regime baseline.
3. `recap_cusum`: ReCAP-inspired adaptive regime detection. It uses CUSUM-style
   change detection over market features, produces variable-length `segment_id`,
   then maps each segment into the shared replay-friendly regime taxonomy.

## Quick Start

The local `nnenv2` environment has the needed dependencies:

```bash
/Users/littleotter/miniconda3/envs/nnenv2/bin/python scripts/make_regime_labels.py --method all
```

Expected outputs:

- `outputs/regime_labels/rule_based_labels.csv`
- `outputs/regime_labels/hmm_labels.csv`
- `outputs/regime_labels/recap_cusum_labels.csv`
- `outputs/regime_labels/all_regime_labels.csv`
- `outputs/regime_labels/label_summary.csv`
- `outputs/regime_labels/label_switches.csv`

To run tests:

```bash
/Users/littleotter/miniconda3/envs/nnenv2/bin/python -m unittest discover -s tests
```

For a fresh environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
make-regime-labels --method all
```

## Default Data

The CLI defaults to:

```text
data/market_indices_20080601_20260531/market_regime_features_wide.csv
```

This file already contains daily SPY/DIA/QQQ/index return, volatility, trend,
and VIX features. The feature builder also accepts long panel files such as
`data/dow30_20080601_20260531/DOW30_recap_features.csv` by collapsing them to
one market-level row per date.

## How This Feeds Section 7/8

The labels are meant to be joined to transitions by date:

```text
transition.date -> regime_label
```

Replay diagnostics can then track:

- sampled regime distribution
- current-regime vs sampled-transition mismatch rate
- TD-error by regime
- reward recovery time after regime switches
- drawdown and turnover around switches

## Literature Anchors

- ReCAP: Regime-Adaptive Continual Learning for Portfolio Management,
  Pan et al. (2026), https://arxiv.org/abs/2606.00143.
- Hamilton (1989), Markov-switching models for non-stationary time series,
  https://www.jstor.org/stable/1912559.
- Ang and Timmermann (2012), regime changes in financial markets,
  https://doi.org/10.1146/annurev-financial-102710-144808.
- Page (1954), CUSUM change detection,
  https://en.wikipedia.org/wiki/CUSUM.

## Run DEER Replay

From the repository root, the default command runs DEER only across all three
label methods (`rule_based,hmm,recap_cusum`) and seeds `0,1,2`:

```bash
python scripts/run_dqn_replay.py
```

To explicitly set the post-boundary floor used by DEER sampling:

```bash
python scripts/run_dqn_replay.py --deer-min-post-samples 4
```

This forces each DEER replay batch to include up to 4 current-boundary post
samples when available, then fills the rest of the batch with PER sampling.

## Run SAC Replay

The DQN runner uses a small discrete action set. The SAC runner changes the
model to continuous portfolio control: the actor outputs continuous logits, and
the environment converts them into long-only weights over `cash + tradable
assets` with a softmax transform.

Run SAC across all label methods, replay variants, and seeds:

```bash
python scripts/run_sac_replay.py
```

For a quick smoke run:

```bash
python scripts/run_sac_replay.py --label-method rule_based --replays uniform --seeds 0 --warmup-steps 32 --start-steps 32 --batch-size 32 --hidden-dim 64 --max-steps 120 --output-root outputs/sac_smoke
```

For tuning preparation, use the JSON grid template:

```bash
python scripts/run_sac_replay.py --label-method rule_based --replays uniform,regime --seeds 0,1 --tuning-grid configs/sac_tuning_grid.json
```

SAC outputs follow the same structure as the DQN runner:

- `trading_log.csv`
- `replay_diagnostics.csv`
- `summary.csv`
- analysis plots under `outputs/sac_replay/analysis/`

## Run the RL Model Library

For generalization experiments, use the model-library runner to call multiple
RL models under the same regime labels, replay settings, and safety rules:

```bash
python scripts/run_rl_library.py --models dqn,sac --label-method rule_based --replays uniform,regime,deer --seeds 0 --max-steps 120
```

The shared policy-safety layer is enabled by default for DQN and SAC. It acts as
a lightweight ReCAP-inspired regime gate: each model's proposed portfolio is
blended with a small regime-conditioned anchor policy library, then constrained
by minimum cash, maximum single-asset weight, and turnover caps. This is meant
to reduce degenerate strategies across models before full backtest validation.

Disable the guard for ablations:

```bash
python scripts/run_rl_library.py --models dqn,sac --disable-policy-safety
```

The combined model summary is written to:

```text
outputs/rl_library/analysis/rl_library_summary.csv
```
