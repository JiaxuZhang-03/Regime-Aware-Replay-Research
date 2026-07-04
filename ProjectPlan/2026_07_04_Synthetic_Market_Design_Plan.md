# Synthetic Market Design Plan for Regime-Aware Replay

## Purpose

This note records the current design idea for adding a synthetic market experiment to the regime-aware replay project.

The goal is not to build a fully realistic market simulator. The goal is to create a controlled, interpretable, non-stationary portfolio environment where the ground-truth regime is known by construction. This allows us to test whether regime-aware replay helps an RL agent adapt after market regime changes, especially by reducing the harmful effect of stale experience from previous regimes.

The core research question is:

> When market regimes switch, does regime-aware replay adapt faster and suffer less from outdated replay samples than uniform replay, recency replay, or prioritized replay?

## Motivation from the Literature

The synthetic market design is mainly inspired by three papers.

### Macri, Jaimungal, and Lillo (2025)

*Deep reinforcement learning for optimal trading with partial information* constructs a regime-switching synthetic trading signal and shows that providing posterior regime probabilities improves RL trading performance.

Main takeaway for this project:

- Regime information is useful for trading RL.
- Structured regime information is more interpretable than hidden neural representations.
- Synthetic regime-switching environments are useful for testing whether an algorithm responds correctly to latent market states.

### Brini and Tantari (2023)

*Deep Reinforcement Trading with Predictable Returns* constructs a synthetic market with predictable mean-reverting factors. The point is to separate two issues that are mixed together in real data:

- whether the market contains learnable signal;
- whether the RL algorithm can actually learn the signal.

Main takeaway for this project:

- A controlled synthetic market can provide stronger evidence than only testing on noisy real data.
- The synthetic environment should have a known mechanism, so that failure or success can be interpreted.
- Experiments should include progressively harder market dynamics.

### Garleanu and Pedersen (2013)

*Dynamic Trading with Predictable Returns and Transaction Costs* derives a closed-form dynamic trading policy when returns are predictable and trading is costly.

Main takeaway for this project:

- Transaction costs matter because good policies should not jump aggressively between portfolios every period.
- A useful benchmark should account for dynamic rebalancing and turnover.
- We do not need to derive a full closed-form solution for the regime-switching case, but we can use simple oracle regime benchmarks inspired by the same intuition.

## High-Level Market Design

The proposed environment is a multi-stock portfolio allocation problem. Here, "multi-asset" means multiple stocks, not a mixed universe of cash, bonds, gold, or commodities.

At each day, the market is in a hidden regime:

```text
z_t in {bull, bear, sideways, crisis}
```

The regime follows a Markov transition process:

```text
P(z_{t+1} | z_t) = transition matrix
```

Conditional on the current regime, daily stock returns are generated from a regime-specific distribution:

```text
r_t | z_t = k ~ N(mu_k, Sigma_k)
```

where:

- `r_t` is the vector of daily returns for all stocks;
- `z_t` is the current hidden market regime;
- `mu_k` is the expected return vector under regime `k`;
- `Sigma_k` is the covariance matrix under regime `k`.

This means that different regimes can change not only expected returns, but also volatility and cross-stock correlations.

## Suggested Stock Universe

The first version should stay small for clarity and debugging.

Possible setup:

```text
N = 5 to 10 stocks
frequency = daily
episode length = 500 to 2000 trading days
number of regimes = 2 to 4
```

The stocks can be synthetic only. We do not need to map them to real tickers in the first version. What matters is that they have different sensitivities to different regimes.

For example:

- some stocks perform well in bull regimes;
- some are more defensive during bear regimes;
- some become highly correlated during crisis regimes;
- some have higher idiosyncratic volatility.

## Regime Definitions

An initial four-regime design could be:

### Bull Regime

- Most stocks have positive expected returns.
- Volatility is relatively low.
- Cross-stock correlation is moderate.
- A risk-seeking portfolio should allocate more to high-return stocks.

### Bear Regime

- Most stocks have negative expected returns.
- Volatility is higher than in the bull regime.
- Defensive or low-beta stocks may lose less.
- A good policy should reduce exposure to risky stocks or shift toward relatively safer stocks.

### Sideways Regime

- Expected returns are close to zero.
- Volatility is moderate.
- Cross-stock dispersion can remain meaningful.
- A good policy should avoid unnecessary turnover and may prefer diversified exposure.

### Crisis Regime

- Most stocks have sharply negative expected returns.
- Volatility is very high.
- Cross-stock correlations rise.
- Diversification becomes less effective.
- A good policy should quickly reduce exposure to the most vulnerable stocks.

## Progressive Difficulty Levels

The synthetic market should be built in layers. This makes the experimental story clearer and helps isolate where regime-aware replay starts to matter.

### Level 0: Stationary Market

No regime switching:

```text
r_t ~ N(mu, Sigma)
```

Purpose:

- sanity check;
- regime-aware replay should not show a large artificial advantage when the market is stationary.

### Level 1: Two-Regime Mean Shift

Two regimes, such as bull and bear. Only expected returns change:

```text
r_t | z_t = k ~ N(mu_k, Sigma)
```

Purpose:

- test whether stale experience from the old return direction slows adaptation after regime switches.

### Level 2: Mean and Volatility Shift

Both expected returns and volatilities change:

```text
r_t | z_t = k ~ N(mu_k, Sigma_k)
```

Purpose:

- test whether the replay method helps the agent adjust both direction and risk exposure.

### Level 3: Correlation Shift Across Stocks

Expected returns, volatilities, and cross-stock correlations all change by regime.

Purpose:

- make the environment more portfolio-specific;
- test whether the agent adapts when diversification benefits change across regimes.

### Level 4: Rare Crisis Regime

Add a low-probability crisis regime with severe negative returns, high volatility, and high cross-stock correlation.

Purpose:

- test adaptation under rare but important regimes;
- examine drawdown and post-switch recovery.

### Level 5: Hidden or Noisy Regime

The true regime `z_t` is not directly given to the agent. Instead, the method receives a noisy regime estimate or posterior probability:

```text
p_t(k) = estimated P(z_t = k | recent returns)
```

Purpose:

- move closer to real-world conditions;
- compare oracle regime-aware replay with estimated regime-aware replay.

## Minimal Interface with RL Components

The exact action, reward, and state design will be optimized by other team members. For the synthetic market design, we only need to define a reasonable interface.

Possible high-level assumptions:

- Action: portfolio weights over the synthetic stocks.
- Reward: portfolio return net of transaction costs, possibly with risk adjustment.
- State: recent return history, current portfolio weights, and optional regime-related features.

Important design note:

We should separate the effect of regime-aware replay from the effect of giving regime information directly to the policy. One useful ablation is:

```text
policy does not observe regime, replay uses regime information
policy observes regime, replay is uniform
policy observes regime, replay is regime-aware
```

This helps identify whether performance gains come from better state representation, better replay sampling, or both.

## Replay Baselines

The synthetic market should compare regime-aware replay against several baselines:

- uniform replay;
- recency replay or sliding-window replay;
- prioritized experience replay;
- regime-aware replay using estimated regimes;
- oracle regime-aware replay using true regimes.

One possible sampling design for regime-aware replay:

```text
70% current-regime transitions
20% similar-regime transitions
10% uniform historical transitions
```

The exact ratios can be tuned later. The important principle is to emphasize relevant current-regime experience while keeping some cross-regime memory to reduce catastrophic forgetting.

## Oracle Benchmark

A full closed-form optimal solution for a regime-switching portfolio problem is likely unnecessary for this project. Instead, we can use a simple oracle regime benchmark.

For each regime `k`, define a regime-specific mean-variance portfolio:

```text
w_k^* = projection((1 / gamma) Sigma_k^{-1} mu_k)
```

where the projection step enforces portfolio constraints, such as long-only weights and weights summing to one.

With transaction costs, the oracle can rebalance gradually:

```text
w_t^oracle = (1 - alpha) w_{t-1} + alpha w_{z_t}^*
```

This benchmark knows the true regime and therefore provides a useful upper-reference strategy. The goal is not necessarily to beat the oracle, but to see whether regime-aware replay moves closer to it than standard replay methods.

## Evaluation Metrics

The main metrics should not only measure total return. Since the contribution is about adaptation under non-stationarity, the evaluation should include regime-switch-specific diagnostics.

Recommended metrics:

- cumulative return;
- transaction-cost-adjusted return;
- Sharpe ratio;
- maximum drawdown;
- turnover;
- performance by regime;
- post-switch return over the first 20, 50, or 100 days after a regime change;
- post-switch drawdown;
- time to adapt after a regime switch;
- distance between learned portfolio and oracle regime portfolio;
- fraction of replay samples drawn from stale regimes.

The most important mechanism-level metrics are:

```text
post-switch adaptation speed
stale-regime replay ratio
distance to oracle portfolio after regime switches
```

These directly test whether regime-aware replay solves the intended problem.

## Expected Experimental Story

The ideal pattern of results would be:

### Stationary Market

Regime-aware replay performs similarly to uniform replay.

Interpretation:

- the method does not create artificial gains when there is no non-stationarity.

### Simple Two-Regime Market

Regime-aware replay adapts faster after bull-bear switches.

Interpretation:

- standard replay is harmed by stale transitions from the previous regime.

### Mean-Volatility-Correlation Regime Market

The advantage of regime-aware replay becomes larger.

Interpretation:

- replay relevance matters more when regimes change both return direction and portfolio risk structure.

### Noisy Hidden-Regime Market

Estimated regime-aware replay still helps, but less than oracle regime-aware replay.

Interpretation:

- regime detection quality matters, but the replay idea remains useful under imperfect information.

## Proposed Positioning

The synthetic market can be described as:

> a controlled regime-switching multi-stock portfolio environment designed to isolate the effect of stale experience replay under market non-stationarity.

This positioning keeps the contribution focused. The synthetic market is not meant to prove that the strategy is immediately profitable in real markets. It is meant to show that regime-aware replay addresses a specific learning problem that appears when market dynamics change.

## Immediate Next Steps

1. Implement a small two-regime multi-stock return generator.
2. Add configurable Markov transition matrices.
3. Add regime-specific `mu_k` and `Sigma_k`.
4. Add logging for true regime labels and regime switches.
5. Add oracle regime portfolio benchmark.
6. Integrate with the RL environment interface once action, reward, and state are finalized by the team.
7. Run a first sanity check comparing stationary vs two-regime markets.

