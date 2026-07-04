# 2026-06-18 Meeting Prep

## Tonight's Goal

Do not try to finish the literature review today.

The meeting should reach three decisions:

1. What has the MVP already demonstrated?
2. How is our method different from DEER, and should DEER become a baseline?
3. What is the next experiment needed before improving the RL architecture?

## Current Status

Completed:

- Three regime labelers:
  - rule-based;
  - Gaussian HMM;
  - ReCAP-inspired CUSUM.
- DQN comparison:
  - online / no replay;
  - uniform replay;
  - PER;
  - regime-aware replay.
- Rule-based label experiment with seeds 0, 1, 2.

Current result:

| Method | Final Value | Max Drawdown | Mean Reward | Mismatch Rate |
|---|---:|---:|---:|---:|
| Online | 0.0449 | 0.9609 | -0.000699 | N/A |
| Uniform | **0.0759** | **0.9357** | **-0.000576** | 0.5674 |
| PER | 0.0585 | 0.9499 | -0.000634 | 0.5785 |
| Regime-aware | 0.0603 | 0.9473 | -0.000638 | **0.2772** |

Interpretation:

- Regime-aware replay reduces mismatch by about 51.1% relative to Uniform.
- PER has higher mismatch than Uniform, consistent with the idea that high TD-error does not imply current-regime relevance.
- The mechanism works in the narrow sense: the sampler selects fewer mismatched transitions.
- The mechanism advantage has not translated into better trading performance.
- All policies are currently poor: negative mean reward and roughly 94%-96% drawdown.
- Current results are online training results, not strict out-of-sample evidence.

## What We Can Claim Now

Safe claim:

> In the current rule-based experiment, standard replay samples many transitions from regimes different from the current regime. A simple regime-aware sampler substantially reduces this mismatch.

Cannot claim yet:

- regime-aware replay improves portfolio return;
- regime-aware replay reduces drawdown;
- the method generalizes across labelers;
- the method works out of sample;
- mismatch reduction causes performance improvement.

## DEER Deep-Dive Questions

Read DEER to answer only these questions:

1. What exact failure mode does DEER define?
2. How is Discrepancy of Environment Dynamics computed?
3. How does DEER distinguish policy-induced changes from environment changes?
4. What is the exact replay priority before and after a detected change?
5. What information does DEER require that our financial setting may not provide?
6. Which baselines and environments are used?
7. Is DEER compatible with DQN and our current buffer?
8. What would count as a fair DEER-like baseline in our experiment?

## DEER vs Our MVP

| Dimension | DEER | Our Current MVP |
|---|---|---|
| Setting | General non-stationary control | Financial market regimes |
| Relevance signal | Environment-dynamics discrepancy | Regime-label match plus TD/recent/random mixture |
| Change handling | Detect dynamics change | Use externally inferred regime label |
| Base agent | General off-policy RL setting | DQN |
| Main metric | Sample efficiency / task reward | Mismatch plus financial metrics |
| Current advantage | More general and methodologically stronger | More interpretable and finance-specific |
| Current weakness | No finance evaluation | Weak policy and no strict out-of-sample test |

Possible positioning:

> DEER estimates transition relevance through learned environment-dynamics discrepancy. We study whether explicit, interpretable market-regime information can guide replay in financial environments, and whether replay mismatch is associated with risk and adaptation outcomes.

## RL Policy Diagnosis Before Changing Algorithms

Do these before replacing DQN with SAC/PPO:

1. Verify accounting:
   - initial portfolio value;
   - final-value definition;
   - max-drawdown implementation;
   - transaction-cost deduction;
   - cash handling.
2. Add non-RL sanity baselines:
   - cash;
   - buy-and-hold;
   - equal weight;
   - random policy;
   - simple momentum rule.
3. Inspect policy behavior:
   - action histogram;
   - position duration;
   - turnover by regime;
   - Q-value scale;
   - epsilon schedule.
4. Check learning setup:
   - state normalization;
   - reward scale;
   - warm-up buffer size;
   - target-network update;
   - learning frequency;
   - transaction-cost magnitude.
5. Separate train / validation / test chronologically.
6. Confirm the DQN can learn on a simple synthetic regime-switching environment.

Decision rule:

- If DQN cannot beat trivial baselines in a simplified environment, fix the environment/training pipeline first.
- If DQN works in synthetic data but fails in market data, investigate signal quality and action/reward design.
- Only then consider SAC, TD3, distributional DQN, or another architecture.

## Suggested Next Experiments

Priority 1: validate current result

- Run HMM and ReCAP-CUSUM labels.
- Increase from 3 to at least 5 seeds.
- Report confidence intervals.
- Test `regime_same_ratio` at 0.3, 0.4, 0.5, 0.6.

Priority 2: make the policy credible

- Add trivial financial baselines.
- Diagnose accounting and policy actions.
- Add strict chronological holdout.
- Run a simplified synthetic regime-switching environment.

Priority 3: strengthen method comparison

- Implement a DEER-like baseline or document why exact reproduction is infeasible.
- Add ablation:
  - regime only;
  - TD-error only;
  - recency only;
  - regime + TD;
  - full sampler.
- Test whether mismatch reduction predicts post-switch reward recovery.

## Work Allocation Proposal

| Person | Next 3-5 Day Task | Deliverable |
|---|---|---|
| You | DEER deep dive and literature map | 2-page DEER note, collision table, related-work outline |
| Teammate A | DQN/environment sanity audit | accounting checks, trivial baselines, action/Q diagnostics |
| Teammate B | robustness experiments | HMM/CUSUM runs, ratio sensitivity, 5 seeds, summary plots |

Shared decision after these deliverables:

- continue current DQN MVP;
- revise sampler;
- or change the base RL algorithm.

## 3 PM to 8 PM Personal Schedule

### 15:00-15:30: Understand current evidence

- Read the DQN result report.
- Copy the four-method result table.
- Memorize the one valid conclusion and the main limitation.

Deliverable: current-status section for the meeting.

### 15:30-17:00: Read DEER deeply

- Abstract, introduction, method figure, algorithm, experiments, limitations.
- Answer the eight DEER questions above.
- Do not read every proof on the first pass.

Deliverable: one-page DEER summary.

### 17:00-17:40: Build the collision table

- Compare DEER vs our MVP on problem, signal, sampler, agent, data, metrics, and novelty.
- Decide whether DEER should be a required baseline.

Deliverable: DEER-vs-ours table.

### 17:40-18:20: Scan the other anchor papers

Only inspect abstract, introduction, main figure, result table, and conclusion:

- LILAC / ICML 2021;
- PER;
- ReCAP;
- BADA or TCYB unknown-change-point paper.

Deliverable: one sentence per paper explaining its role.

### 18:20-19:00: Diagnose the bad-policy problem

- Review the result table and current experimental assumptions.
- Prepare the sanity-check list and trivial baselines.
- Do not tune hyperparameters blindly today.

Deliverable: prioritized RL debugging list.

### 19:00-19:35: Prepare meeting notes

Use five sections:

1. What is completed.
2. What the current result actually says.
3. What DEER changes about novelty.
4. Why policy performance is not yet credible.
5. What each person should deliver next.

### 19:35-20:00: Rehearse and reduce

- Keep the update under 10 minutes.
- Highlight decisions, not everything you read.
- Write down three questions that require team agreement.

## Meeting Agenda

1. Status update: 5 minutes.
2. Current DQN result and interpretation: 10 minutes.
3. DEER overlap and positioning: 10 minutes.
4. Bad-policy diagnosis: 10 minutes.
5. Next experiments and role assignment: 15 minutes.
6. Confirm deadlines and Git workflow: 5 minutes.

## Three Decisions to Get from the Team

1. Is the next milestone mechanism validation or profitable policy performance?
2. Must we implement DEER as a baseline now, or after stabilizing DQN?
3. Who owns RL debugging, robustness runs, and literature positioning?

