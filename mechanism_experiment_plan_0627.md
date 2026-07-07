# Mechanism Experiment Plan Before Full Backtest Validation

## Context

At the current stage, the base trading models are not stable enough. Online DQN, Uniform Replay, and PER may still learn degenerate policies, so a direct portfolio-return comparison may not be very informative yet.

Instead, we should first run mechanism-level experiments. These experiments do not need to prove that the strategy earns higher return. Their goal is to show whether regime shifts create measurable learning instability, and whether DEER-style replay changes the sample-selection process in a meaningful way.

The main idea is:

> Before proving that the method improves portfolio performance, first prove that regime shifts affect the learned value function/policy and that DEER-style replay responds to those changes in a structured way.

---

## Experiment 1: Regime-Shift Impact on Q Function and Policy Behavior

### Purpose

This experiment checks whether regime changes actually affect the learning state of the agent.

The goal is not to compare financial performance. Instead, we want to answer:

> When a regime boundary occurs, do Online DQN, Uniform Replay, or PER show measurable changes in their value function or policy behavior? Does DEER-style replay reduce instability or recover faster after the shift?

This can be framed as a machine-learning stability/adaptation experiment rather than a financial backtest.

### Core Intuition

If the market regime changes, the meaning of the same state-action pair may also change. For example, a long-risk action that was valuable in a bull regime may become less valuable in a high-volatility selloff.

Therefore, after a regime boundary, we should observe some change in the agent's Q estimates, action choices, or TD errors. DEER-style replay should ideally make this transition less chaotic or help the model adapt faster.

### Candidate Metrics

#### 1. Q-Drift

Measure how much the Q estimate changes for the same state-action pair:

```text
Q-Drift(s,a) = |Q_after(s,a) - Q_before(s,a)|
```

This is close to the DoE idea, but can be reported as a general value-function drift metric.

#### 2. Action Flip Rate

Use a fixed set of probe states. Compare the greedy action before and after a regime boundary:

```text
Action Flip Rate =
number of probe states where argmax_a Q_after(s,a) != argmax_a Q_before(s,a)
/ total probe states
```

This measures how much the policy changes after the regime shift.

#### 3. Q-Margin Change

For each probe state, compute:

```text
Q-Margin = Q_best_action - Q_second_best_action
```

If the margin becomes smaller after a boundary, the policy is less confident. If DEER-style replay works well, the margin may recover faster.

#### 4. TD-Error Shock

Track whether TD-error spikes after a regime boundary:

```text
TD-error shock = average |TD-error| after boundary
```

The important metric is not only the spike size, but also how many updates it takes to return to a stable level.

#### 5. Policy Distribution Shift

On the same probe states, compare the distribution of selected actions before and after the boundary.

Example:

```text
Before boundary: 60% long, 30% neutral, 10% defensive
After boundary: 25% long, 45% neutral, 30% defensive
```

This gives a broader view than the action flip rate.

### Experimental Flow

1. Choose one regime labeling method first, preferably `rule_based`.
2. Identify all regime boundaries from the label sequence.
3. For each replay method, run the same DQN environment with the same seeds:

```text
Online DQN
Uniform Replay
PER
DEER-style Replay
```

4. At each regime boundary, save a frozen Q snapshot:

```text
Q_before = Q network before or at the boundary
```

5. After the boundary, periodically save or evaluate the current Q network:

```text
Q_after_1, Q_after_2, Q_after_3, ...
```

6. Use a fixed probe-state set to evaluate:

```text
Q-Drift
Action Flip Rate
Q-Margin Change
TD-error Shock
Policy Distribution Shift
```

7. Align all boundaries at day/update 0 and plot average curves around regime shifts.

Suggested event window:

```text
[-20 trading days, +60 trading days]
```

or, if measured by gradient updates:

```text
[-K updates, +K updates]
```

### Expected Output

Figures:

- Q-Drift around regime boundaries.
- Action Flip Rate around regime boundaries.
- TD-error shock and recovery curve.
- Q-Margin recovery curve.
- Policy action distribution before and after boundaries.

Tables:

- Average post-boundary Q-Drift.
- Average recovery time of TD-error.
- Average action flip rate.
- Comparison across Online, Uniform, PER, and DEER-style replay.

### What Would Support Our Mechanism?

Useful evidence would include:

- Regime boundaries create clear Q-drift or TD-error shocks.
- Uniform/PER show larger or longer instability after boundaries.
- DEER-style replay reduces recovery time or produces smoother adaptation.
- DEER-style replay does not simply suppress all changes; it helps the model adapt in a more stable way.

### Why This Experiment Matters

Even if portfolio returns are not yet reliable, this experiment can show that regime shifts create a real machine-learning problem for the value function and policy. This supports the need for a regime-aware replay mechanism.

---

## Experiment 2: What Does DoE/Priority Actually Select?

### Purpose

This experiment studies the internal behavior of DEER-style replay.

The goal is to answer:

> What are high-DoE samples? What are high-priority samples? Which transitions are replayed or priority-refreshed more frequently? Does the exploratory initial priority assignment affect the mechanism?

This experiment helps explain whether DEER-style priority is selecting meaningful samples or just amplifying noise.

### Core Intuition

DEER-style replay should not simply replay random high-error samples. Ideally:

- High-DoE samples should correspond to transitions whose value changed after the regime shift.
- High-priority samples should be informative for adaptation, not just noisy.
- More frequently replayed transitions should have interpretable properties, such as being near regime boundaries or having high TD-error/DoE.

### Part A: High DoE vs Low DoE Samples

Split all transitions into groups:

```text
High DoE group: top 20% by DoE
Low DoE group: bottom 20% by DoE
```

Then compare their characteristics.

### Features to Compare

For each group, report:

- Sample age.
- Distance to nearest regime boundary.
- Regime label.
- Boundary ID.
- TD-error.
- Reward magnitude.
- Return sign.
- Volatility level.
- Action type.
- Whether the transition involves a position change.
- Next-day return.
- Realized volatility.

### Expected Pattern

High-DoE samples should ideally be more concentrated around regime changes or market states where the value relationship has clearly changed.

If high-DoE samples look completely random or are dominated by noisy extreme rewards, then the current DoE definition may need adjustment.

---

## Part B: High Priority vs Low Priority Samples

Split transitions into groups:

```text
High priority group: top 20% by priority
Low priority group: bottom 20% by priority
```

Compare:

- TD-error.
- DoE.
- Sample age.
- Distance to boundary.
- Current-boundary vs old-boundary status.
- Regime label.
- Sampling count.
- Priority update count.
- Mean priority over time.
- Maximum priority over time.

### Expected Pattern

High-priority samples should not be high only because of noisy rewards. Ideally, high-priority samples should be associated with:

- Large TD-error when the model has not learned the transition well.
- Large DoE when the transition reveals regime change.
- Post-boundary samples during the early adaptation phase.
- Low-DoE old samples that are still reusable across regimes.

This analysis tells us whether priority is doing something interpretable.

---

## Part C: Sampling Frequency and Priority Refresh Frequency

For each transition, record:

```text
sample_count
priority_update_count
last_sample_step
last_priority_update_step
mean_priority
max_priority
mean_DoE
mean_TD_error
boundary_id
regime_id
```

Then analyze:

- Which transitions are replayed most frequently?
- Are they high-DoE, high-TD-error, or both?
- Are they concentrated near regime boundaries?
- Do they come mostly from the current regime or old regimes?
- Are a small number of transitions dominating replay?

### Useful Plots

- Sampling count vs DoE.
- Sampling count vs TD-error.
- Priority update count vs DoE.
- Priority distribution by regime/boundary.
- Top replayed samples ranked by sample count.
- Replay concentration curve: top 1%, 5%, 10% samples' share of total replay.

### Expected Pattern

DEER-style replay should change replay focus after boundaries, but it should not collapse into replaying only a tiny number of noisy transitions.

---

## Part D: Exploratory Initial Priority Assignment

When a new transition enters the replay buffer, it needs an initial priority before it has been sampled and updated.

This initial assignment may affect whether new-regime samples are learned early enough.

### Candidate Settings

Start with three settings:

```text
Setting A: max priority
Setting B: median priority
Setting C: DoE-based initial priority
```

Optional later settings:

```text
Setting D: TD-error initial priority
Setting E: small constant priority
```

### Metrics to Compare

- Time until a new transition is first sampled.
- Post-boundary sample rate.
- Priority entropy.
- DoE distribution among sampled transitions.
- TD-error recovery after boundary.
- Replay concentration.
- Whether a few samples are over-replayed.

### Expected Pattern

If initial priority is too low, new-regime transitions may enter the buffer but not be sampled early enough.

If initial priority is too high, a few new samples may dominate replay and cause overfitting or instability.

The best setting should allow new-regime samples to be learned early, while still preserving replay diversity.

---

## Recommended Execution Order

### Step 1: Run Experiment 1 First

Start with:

```text
rule_based labels
3 seeds
Online DQN / Uniform Replay / PER / DEER-style Replay
```

Focus on:

- Q-Drift.
- Action Flip Rate.
- TD-error Shock.
- Q-Margin Change.

The key question is:

> Does regime change create measurable instability in Q estimates or policy behavior?

### Step 2: Run Experiment 2A and 2B

Once DoE and priority logs are available, compare:

```text
High DoE vs Low DoE
High Priority vs Low Priority
```

The key question is:

> Are DoE and priority selecting interpretable samples?

### Step 3: Add Sampling Frequency Analysis

Analyze:

```text
sample_count
priority_update_count
priority entropy
replay concentration
```

The key question is:

> Does DEER-style replay change sample selection without collapsing to a few noisy samples?

### Step 4: Add Initial Priority Ablation

Compare:

```text
max priority
median priority
DoE-based initial priority
```

The key question is:

> Does the way new samples enter the buffer affect the effectiveness of the replay mechanism?

---

## Minimal Success Criteria

The mechanism-level experiments are successful if they show at least two of the following:

1. Regime boundaries cause measurable Q-drift, TD-error shock, action flip, or Q-margin change.
2. DEER-style replay reduces instability or shortens recovery time after boundaries.
3. High-DoE samples have interpretable differences from low-DoE samples.
4. High-priority samples are not merely noisy outliers; they correspond to TD-error, DoE, or boundary-related learning needs.
5. Sampling frequency and priority refresh behavior are consistent with the intended DEER-style mechanism.
6. Initial priority assignment affects new-regime learning speed in a measurable way.

---

## How This Supports the Final Paper

These experiments can support the paper even before the full trading backtest is strong.

They can establish the mechanism story:

> Regime shifts cause measurable instability in the learned value function and policy. DEER-style replay changes the replay distribution toward transitions that better explain or stabilize this shift.

After this mechanism story is established, the full portfolio backtest can be used as the higher-level financial validation:

- cumulative return;
- Sharpe / Sortino;
- maximum drawdown;
- turnover;
- transaction-cost robustness;
- post-boundary recovery.

In short, Experiment 1 proves that regime shift matters for learning. Experiment 2 explains what the replay mechanism is actually doing.
