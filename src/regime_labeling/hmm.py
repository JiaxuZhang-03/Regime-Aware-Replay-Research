"""Small diagonal Gaussian HMM for market-regime labels.

This intentionally avoids hmmlearn so the repo can run with only numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import MarketFeatureConfig, build_market_features, feature_matrix


@dataclass(frozen=True)
class HMMConfig:
    n_states: int = 3
    n_iter: int = 80
    tol: float = 1e-4
    random_seed: int = 7
    covariance_floor: float = 1e-3
    feature_columns: tuple[str, ...] = ("ret_short", "ret_long", "trend", "vol", "vix")


class GaussianHMMLabeler:
    """Diagonal-covariance Gaussian HMM with EM and Viterbi decoding."""

    def __init__(self, config: HMMConfig | None = None):
        self.config = config or HMMConfig()
        if self.config.n_states < 2:
            raise ValueError("n_states must be at least 2")
        self.startprob_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.log_likelihood_: float | None = None

    def fit(self, x: np.ndarray) -> "GaussianHMMLabeler":
        x = _as_2d(x)
        n_samples, n_features = x.shape
        labels = _initial_labels(x, self.config.n_states)
        self.startprob_ = np.full(self.config.n_states, 1.0 / self.config.n_states)
        self.transmat_ = _initial_transition(self.config.n_states)
        self.means_ = np.vstack(
            [
                x[labels == state].mean(axis=0)
                if np.any(labels == state)
                else x.mean(axis=0)
                for state in range(self.config.n_states)
            ]
        )
        self.vars_ = np.vstack(
            [
                x[labels == state].var(axis=0) + self.config.covariance_floor
                if np.any(labels == state)
                else x.var(axis=0) + self.config.covariance_floor
                for state in range(self.config.n_states)
            ]
        )

        previous_log_likelihood = -np.inf
        for _ in range(self.config.n_iter):
            log_b = self._log_emissions(x)
            log_alpha, log_likelihood = self._forward(log_b)
            log_beta = self._backward(log_b)
            gamma = np.exp(log_alpha + log_beta - log_likelihood)
            xi_sum = self._expected_transitions(log_b, log_alpha, log_beta, log_likelihood)

            weights = gamma.sum(axis=0) + 1e-12
            self.startprob_ = gamma[0] + 1e-12
            self.startprob_ /= self.startprob_.sum()
            self.transmat_ = xi_sum + 1e-12
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)
            self.means_ = (gamma.T @ x) / weights[:, None]
            centered = x[:, None, :] - self.means_[None, :, :]
            self.vars_ = (gamma[:, :, None] * centered**2).sum(axis=0) / weights[:, None]
            self.vars_ = np.maximum(self.vars_, self.config.covariance_floor)

            if abs(log_likelihood - previous_log_likelihood) < self.config.tol:
                break
            previous_log_likelihood = log_likelihood

        self.log_likelihood_ = float(previous_log_likelihood)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = _as_2d(x)
        log_b = self._log_emissions(x)
        n_samples = x.shape[0]
        n_states = self.config.n_states
        delta = np.zeros((n_samples, n_states))
        psi = np.zeros((n_samples, n_states), dtype=int)
        delta[0] = np.log(self.startprob_) + log_b[0]
        log_trans = np.log(self.transmat_)

        for t in range(1, n_samples):
            scores = delta[t - 1][:, None] + log_trans
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = np.max(scores, axis=0) + log_b[t]

        states = np.zeros(n_samples, dtype=int)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n_samples - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        x = _as_2d(x)
        log_b = self._log_emissions(x)
        log_alpha, log_likelihood = self._forward(log_b)
        log_beta = self._backward(log_b)
        return np.exp(log_alpha + log_beta - log_likelihood)

    def _log_emissions(self, x: np.ndarray) -> np.ndarray:
        self._check_fitted()
        diff = x[:, None, :] - self.means_[None, :, :]
        return -0.5 * (
            np.sum(np.log(2.0 * np.pi * self.vars_), axis=1)[None, :]
            + np.sum(diff**2 / self.vars_[None, :, :], axis=2)
        )

    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        log_trans = np.log(self.transmat_)
        log_alpha = np.zeros_like(log_b)
        log_alpha[0] = np.log(self.startprob_) + log_b[0]
        for t in range(1, log_b.shape[0]):
            log_alpha[t] = log_b[t] + _logsumexp(log_alpha[t - 1][:, None] + log_trans, axis=0)
        return log_alpha, float(_logsumexp(log_alpha[-1], axis=0))

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        log_trans = np.log(self.transmat_)
        log_beta = np.zeros_like(log_b)
        for t in range(log_b.shape[0] - 2, -1, -1):
            log_beta[t] = _logsumexp(log_trans + log_b[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
        return log_beta

    def _expected_transitions(
        self,
        log_b: np.ndarray,
        log_alpha: np.ndarray,
        log_beta: np.ndarray,
        log_likelihood: float,
    ) -> np.ndarray:
        log_trans = np.log(self.transmat_)
        xi_sum = np.zeros_like(self.transmat_)
        for t in range(log_b.shape[0] - 1):
            log_xi = (
                log_alpha[t][:, None]
                + log_trans
                + log_b[t + 1][None, :]
                + log_beta[t + 1][None, :]
                - log_likelihood
            )
            xi_sum += np.exp(log_xi)
        return xi_sum

    def _check_fitted(self) -> None:
        if self.startprob_ is None or self.transmat_ is None or self.means_ is None or self.vars_ is None:
            raise RuntimeError("model is not fitted")


def label_hmm(
    df: pd.DataFrame,
    feature_config: MarketFeatureConfig | None = None,
    hmm_config: HMMConfig | None = None,
) -> pd.DataFrame:
    hmm_config = hmm_config or HMMConfig()
    features = build_market_features(df, feature_config)
    matrix, selected, _, _ = feature_matrix(features, hmm_config.feature_columns, standardize=True)

    model = GaussianHMMLabeler(hmm_config).fit(matrix)
    raw_states = model.predict(matrix)
    probabilities = model.predict_proba(matrix)
    ordered_labels, ordered_names, state_order = _interpret_states(raw_states, matrix, selected)

    out = features.copy()
    out["method"] = "hmm"
    out["regime_label"] = ordered_labels
    out["regime_name"] = ordered_names
    out["hmm_state"] = raw_states
    for ordered_label, state in enumerate(state_order):
        out[f"prob_label_{ordered_label}"] = probabilities[:, state]
    leading = ["date", "method", "regime_label", "regime_name", "hmm_state"]
    return out[[*leading, *[column for column in out.columns if column not in leading]]]


def _interpret_states(
    raw_states: np.ndarray,
    matrix: np.ndarray,
    selected: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    names_by_count = {
        2: ["risk_off", "risk_on"],
        3: ["risk_off", "sideways", "risk_on"],
        4: ["risk_off", "sideways", "high_vol", "risk_on"],
    }
    fallback_names = ["risk_off", "sideways", "high_vol", "risk_on"]
    n_states = int(raw_states.max()) + 1

    col_index = {name: idx for idx, name in enumerate(selected)}
    ret_idx = col_index.get("ret_long", col_index.get("ret_short", 0))
    trend_idx = col_index.get("trend", ret_idx)
    vol_idx = col_index.get("vol", ret_idx)
    vix_idx = col_index.get("vix", vol_idx)

    scores = []
    for state in range(n_states):
        state_matrix = matrix[raw_states == state]
        if len(state_matrix) == 0:
            scores.append(-np.inf)
            continue
        means = state_matrix.mean(axis=0)
        score = means[ret_idx] + means[trend_idx] - means[vol_idx] - 0.35 * means[vix_idx]
        scores.append(float(score))

    state_order = list(np.argsort(scores))
    name_order = names_by_count.get(n_states, fallback_names[:n_states])
    state_to_label = {state: label for label, state in enumerate(state_order)}
    labels = np.array([state_to_label[state] for state in raw_states], dtype=int)
    names = np.array([name_order[min(label, len(name_order) - 1)] for label in labels], dtype=object)
    return labels, names, state_order


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("expected a 2D array")
    if x.shape[0] < 2:
        raise ValueError("expected at least two observations")
    return x


def _initial_labels(x: np.ndarray, n_states: int) -> np.ndarray:
    score = x[:, 0]
    if x.shape[1] > 2:
        score = score + 0.5 * x[:, 1] - 0.5 * x[:, -2]
    quantiles = np.quantile(score, np.linspace(0.0, 1.0, n_states + 1)[1:-1])
    return np.digitize(score, quantiles)


def _initial_transition(n_states: int) -> np.ndarray:
    if n_states == 1:
        return np.ones((1, 1))
    matrix = np.full((n_states, n_states), 0.1 / (n_states - 1))
    np.fill_diagonal(matrix, 0.9)
    return matrix


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_values)
    summed = np.sum(stable, axis=axis, keepdims=True)
    result = max_values + np.log(summed)
    return np.squeeze(result, axis=axis)
