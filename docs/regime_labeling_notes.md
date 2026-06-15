# Regime Labeling Notes

## Goal

ProjectPlan section 7 says the immediate risk is not code complexity; it is
whether the replay-contamination failure mode exists. The labeler should
therefore be good enough to tag transitions and measure mismatch, but not so
heavy that it becomes the main research contribution.

## Shared Output Schema

All methods write at least:

```text
date, method, regime_label, regime_name
```

The shared taxonomy is:

```text
0 risk_on
1 sideways
2 high_vol
3 risk_off
```

For replay diagnostics, exact names matter less than consistency: the sampler
needs to compare the current date's label with sampled transition labels.

## Method 1: Rule-Based Trend/Volatility

Reference role: transparent baseline and sanity check.

Inputs:

- short and long rolling returns
- price-vs-moving-average trend
- rolling volatility
- VIX when available

Logic:

- `risk_on`: positive trend and long return, not high volatility.
- `high_vol`: volatility or VIX above rolling quantile, without clear negative trend.
- `risk_off`: negative trend or negative long return.
- `sideways`: fallback.

The thresholds are rolling and shifted by one day to avoid using future
information when labels are later used in walk-forward diagnostics.

## Method 2: Gaussian HMM

Reference role: classical latent-regime baseline. This follows the broad
financial-econometrics idea behind Markov-switching regimes, associated with
Hamilton-style regime-switching models and later financial-market regime
reviews such as Ang and Timmermann.

Implementation choice:

- diagonal Gaussian emissions
- EM training with forward-backward
- Viterbi decoding for the hard label
- state probabilities exported as `prob_label_*`

The states are ordered by a risk score:

```text
return + trend - volatility - VIX
```

This makes HMM labels interpretable enough for replay analysis without requiring
manual state relabeling after every run.

## Method 3: ReCAP-Inspired CUSUM/ARD

Reference role: literature-aware third method. ReCAP's Adaptive Regime Detection
module segments market data into variable-length regimes by applying a
Cumulative Sum style detector to market-level features such as VIX, turbulence,
and aggregate asset statistics. This repo adapts that idea as a labeler only.

Implementation choice:

- run symmetric CUSUM statistics on market-level features
- emit `change_point` and variable-length `segment_id`
- assign each segment to the shared taxonomy using segment-level return, trend,
  volatility, and VIX summaries

This is deliberately not a full ReCAP reproduction. It borrows the regime
segmentation mechanism while keeping our contribution focused on transition-
level replay reuse.

## Suggested Next Diagnostics

1. Join labels to every stored transition by date.
2. Log current label and sampled transition labels for uniform replay, PER,
   sliding-window replay, and regime-aware replay.
3. Compare mismatch rate around label switches.
4. Plot TD-error by current/sampled regime pair.
5. Track drawdown and turnover spikes after detected regime switches.

## References

- Pan, C., Ren, L., Xiong, L., Li, Y., Wei, W., & Yang, X. (2026).
  Regime-Adaptive Continual Learning for Portfolio Management.
  https://arxiv.org/abs/2606.00143
- Hamilton, J. D. (1989). A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle. Econometrica.
  https://www.jstor.org/stable/1912559
- Ang, A., & Timmermann, A. (2012). Regime Changes and Financial Markets.
  https://doi.org/10.1146/annurev-financial-102710-144808
- Page, E. S. (1954). Continuous Inspection Schemes. Biometrika.
  https://en.wikipedia.org/wiki/CUSUM
