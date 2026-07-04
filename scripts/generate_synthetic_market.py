from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.regime_labeling.features import MarketFeatureConfig
from src.regime_labeling.hmm import HMMConfig, label_hmm
from src.regime_labeling.recap_ard import RecapCusumConfig, label_recap_cusum


def parse_methods(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def make_equicorrelation_cov(n_assets: int, daily_vol: float, corr: float) -> np.ndarray:
    corr = float(np.clip(corr, -1.0 / max(n_assets - 1, 1) + 1e-6, 0.999))
    base = np.full((n_assets, n_assets), corr, dtype=float)
    np.fill_diagonal(base, 1.0)
    return (daily_vol ** 2) * base


def simulate_two_state_markov(n_days: int, switch_prob: float, rng: np.random.Generator) -> np.ndarray:
    switch_prob = float(np.clip(switch_prob, 0.0, 1.0))
    regimes = np.zeros(n_days, dtype=np.int64)
    for t in range(1, n_days):
        if rng.random() < switch_prob:
            regimes[t] = 1 - regimes[t - 1]
        else:
            regimes[t] = regimes[t - 1]
    return regimes


def simulate_markov_chain(
    n_days: int,
    transition: np.ndarray,
    rng: np.random.Generator,
    start_regime: int = 0,
) -> np.ndarray:
    transition = np.asarray(transition, dtype=float)
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError("transition matrix must be square.")
    row_sums = transition.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("transition matrix rows must have positive sums.")
    transition = transition / row_sums[:, None]

    regimes = np.zeros(n_days, dtype=np.int64)
    regimes[0] = int(start_regime)
    for t in range(1, n_days):
        regimes[t] = rng.choice(transition.shape[0], p=transition[regimes[t - 1]])
    return regimes


def make_regime_mu(n_assets: int, bull_mu: float, bear_mu: float) -> dict[int, np.ndarray]:
    bull = np.full(n_assets, bull_mu, dtype=float)
    bear = np.full(n_assets, bear_mu, dtype=float)

    # Add mild cross-sectional structure so portfolio choice is meaningful.
    tilt = np.linspace(0.75, 1.25, n_assets)
    bull = bull * tilt
    defensive = np.linspace(1.25, 0.55, n_assets)
    bear = bear * defensive
    return {0: bull, 1: bear}


def regime_names_from_labels(regimes: np.ndarray) -> np.ndarray:
    lookup = {0: "bull", 1: "bear", 2: "crisis"}
    return np.array([lookup.get(int(x), f"regime_{int(x)}") for x in regimes], dtype=object)


def market_frame_from_returns(
    *,
    returns: np.ndarray,
    regimes: np.ndarray,
    symbols: list[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    market = pd.DataFrame({"date": dates})

    for i, symbol in enumerate(symbols):
        market[f"ret_{symbol}"] = returns[:, i]
        market[f"close_{symbol}"] = prices[:, i]

    primary_ret = pd.Series(returns[:, 0])
    market["vix"] = (
        primary_ret.rolling(20, min_periods=5).std().fillna(primary_ret.std()) * np.sqrt(252.0) * 100.0
    ).to_numpy()
    demeaned = returns - returns.mean(axis=0, keepdims=True)
    market["turbulence"] = np.sqrt(np.mean(demeaned**2, axis=1))
    market["true_regime_label"] = regimes.astype(int)
    market["true_regime_name"] = regime_names_from_labels(regimes)
    return market


def generate_level0_stationary(
    *,
    n_days: int,
    n_assets: int,
    seed: int,
    start_date: str,
    daily_mu: float,
    daily_vol: float,
    corr: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    symbols = [f"SYN{i}" for i in range(n_assets)]
    dates = pd.bdate_range(start=start_date, periods=n_days)

    mu = np.full(n_assets, daily_mu, dtype=float)
    cov = make_equicorrelation_cov(n_assets, daily_vol, corr)
    returns = rng.multivariate_normal(mean=mu, cov=cov, size=n_days)
    returns[0, :] = 0.0

    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    market = pd.DataFrame({"date": dates})

    for i, symbol in enumerate(symbols):
        market[f"ret_{symbol}"] = returns[:, i]
        market[f"close_{symbol}"] = prices[:, i]

    primary_ret = pd.Series(returns[:, 0])
    market["vix"] = (
        primary_ret.rolling(20, min_periods=5).std().fillna(primary_ret.std()) * np.sqrt(252.0) * 100.0
    ).to_numpy()
    demeaned = returns - returns.mean(axis=0, keepdims=True)
    market["turbulence"] = np.sqrt(np.mean(demeaned**2, axis=1))

    metadata = {
        "level": "level0_stationary",
        "description": "Stationary multivariate Gaussian synthetic market with one constant regime.",
        "seed": seed,
        "n_days": n_days,
        "n_assets": n_assets,
        "symbols": symbols,
        "start_date": start_date,
        "daily_mu": daily_mu,
        "daily_vol": daily_vol,
        "corr": corr,
        "covariance": cov.tolist(),
    }
    return market, metadata


def generate_level1_mean_shift(
    *,
    n_days: int,
    n_assets: int,
    seed: int,
    start_date: str,
    bull_mu: float,
    bear_mu: float,
    daily_vol: float,
    corr: float,
    switch_prob: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    symbols = [f"SYN{i}" for i in range(n_assets)]
    dates = pd.bdate_range(start=start_date, periods=n_days)
    regimes = simulate_two_state_markov(n_days, switch_prob, rng)
    mu_by_regime = make_regime_mu(n_assets, bull_mu, bear_mu)
    cov = make_equicorrelation_cov(n_assets, daily_vol, corr)

    returns = np.zeros((n_days, n_assets), dtype=float)
    for t, regime in enumerate(regimes):
        returns[t, :] = rng.multivariate_normal(mean=mu_by_regime[int(regime)], cov=cov)
    returns[0, :] = 0.0

    market = market_frame_from_returns(returns=returns, regimes=regimes, symbols=symbols, dates=dates)
    metadata = {
        "level": "level1_mean_shift",
        "description": "Two-regime synthetic market where only expected returns change.",
        "seed": seed,
        "n_days": n_days,
        "n_assets": n_assets,
        "symbols": symbols,
        "start_date": start_date,
        "switch_prob": switch_prob,
        "bull_mu_vector": mu_by_regime[0].tolist(),
        "bear_mu_vector": mu_by_regime[1].tolist(),
        "daily_vol": daily_vol,
        "corr": corr,
        "covariance": cov.tolist(),
        "regime_counts": {
            "bull": int(np.sum(regimes == 0)),
            "bear": int(np.sum(regimes == 1)),
        },
        "n_switches": int(np.sum(regimes[1:] != regimes[:-1])),
    }
    return market, metadata


def generate_level2_mean_vol_shift(
    *,
    n_days: int,
    n_assets: int,
    seed: int,
    start_date: str,
    bull_mu: float,
    bear_mu: float,
    bull_vol: float,
    bear_vol: float,
    corr: float,
    switch_prob: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    symbols = [f"SYN{i}" for i in range(n_assets)]
    dates = pd.bdate_range(start=start_date, periods=n_days)
    regimes = simulate_two_state_markov(n_days, switch_prob, rng)
    mu_by_regime = make_regime_mu(n_assets, bull_mu, bear_mu)
    cov_by_regime = {
        0: make_equicorrelation_cov(n_assets, bull_vol, corr),
        1: make_equicorrelation_cov(n_assets, bear_vol, corr),
    }

    returns = np.zeros((n_days, n_assets), dtype=float)
    for t, regime in enumerate(regimes):
        r = int(regime)
        returns[t, :] = rng.multivariate_normal(mean=mu_by_regime[r], cov=cov_by_regime[r])
    returns[0, :] = 0.0

    market = market_frame_from_returns(returns=returns, regimes=regimes, symbols=symbols, dates=dates)
    metadata = {
        "level": "level2_mean_vol_shift",
        "description": "Two-regime synthetic market where expected returns and volatility change.",
        "seed": seed,
        "n_days": n_days,
        "n_assets": n_assets,
        "symbols": symbols,
        "start_date": start_date,
        "switch_prob": switch_prob,
        "bull_mu_vector": mu_by_regime[0].tolist(),
        "bear_mu_vector": mu_by_regime[1].tolist(),
        "bull_vol": bull_vol,
        "bear_vol": bear_vol,
        "corr": corr,
        "bull_covariance": cov_by_regime[0].tolist(),
        "bear_covariance": cov_by_regime[1].tolist(),
        "regime_counts": {
            "bull": int(np.sum(regimes == 0)),
            "bear": int(np.sum(regimes == 1)),
        },
        "n_switches": int(np.sum(regimes[1:] != regimes[:-1])),
    }
    return market, metadata


def generate_level3_mean_vol_corr_shift(
    *,
    n_days: int,
    n_assets: int,
    seed: int,
    start_date: str,
    bull_mu: float,
    bear_mu: float,
    bull_vol: float,
    bear_vol: float,
    bull_corr: float,
    bear_corr: float,
    switch_prob: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    symbols = [f"SYN{i}" for i in range(n_assets)]
    dates = pd.bdate_range(start=start_date, periods=n_days)
    regimes = simulate_two_state_markov(n_days, switch_prob, rng)
    mu_by_regime = make_regime_mu(n_assets, bull_mu, bear_mu)
    cov_by_regime = {
        0: make_equicorrelation_cov(n_assets, bull_vol, bull_corr),
        1: make_equicorrelation_cov(n_assets, bear_vol, bear_corr),
    }

    returns = np.zeros((n_days, n_assets), dtype=float)
    for t, regime in enumerate(regimes):
        r = int(regime)
        returns[t, :] = rng.multivariate_normal(mean=mu_by_regime[r], cov=cov_by_regime[r])
    returns[0, :] = 0.0

    market = market_frame_from_returns(returns=returns, regimes=regimes, symbols=symbols, dates=dates)
    metadata = {
        "level": "level3_mean_vol_corr_shift",
        "description": "Two-regime synthetic market where expected returns, volatility, and correlations change.",
        "seed": seed,
        "n_days": n_days,
        "n_assets": n_assets,
        "symbols": symbols,
        "start_date": start_date,
        "switch_prob": switch_prob,
        "bull_mu_vector": mu_by_regime[0].tolist(),
        "bear_mu_vector": mu_by_regime[1].tolist(),
        "bull_vol": bull_vol,
        "bear_vol": bear_vol,
        "bull_corr": bull_corr,
        "bear_corr": bear_corr,
        "bull_covariance": cov_by_regime[0].tolist(),
        "bear_covariance": cov_by_regime[1].tolist(),
        "regime_counts": {
            "bull": int(np.sum(regimes == 0)),
            "bear": int(np.sum(regimes == 1)),
        },
        "n_switches": int(np.sum(regimes[1:] != regimes[:-1])),
    }
    return market, metadata


def generate_level4_rare_crisis(
    *,
    n_days: int,
    n_assets: int,
    seed: int,
    start_date: str,
    bull_mu: float,
    bear_mu: float,
    crisis_mu: float,
    bull_vol: float,
    bear_vol: float,
    crisis_vol: float,
    bull_corr: float,
    bear_corr: float,
    crisis_corr: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(seed)
    symbols = [f"SYN{i}" for i in range(n_assets)]
    dates = pd.bdate_range(start=start_date, periods=n_days)

    # Rows: bull, bear, crisis. Crisis is rare but persistent once entered.
    transition = np.array(
        [
            [0.960, 0.035, 0.005],
            [0.045, 0.940, 0.015],
            [0.080, 0.120, 0.800],
        ],
        dtype=float,
    )
    regimes = simulate_markov_chain(n_days, transition, rng, start_regime=0)

    mu_by_regime = make_regime_mu(n_assets, bull_mu, bear_mu)
    crisis_tilt = np.linspace(1.35, 0.75, n_assets)
    mu_by_regime[2] = np.full(n_assets, crisis_mu, dtype=float) * crisis_tilt
    cov_by_regime = {
        0: make_equicorrelation_cov(n_assets, bull_vol, bull_corr),
        1: make_equicorrelation_cov(n_assets, bear_vol, bear_corr),
        2: make_equicorrelation_cov(n_assets, crisis_vol, crisis_corr),
    }

    returns = np.zeros((n_days, n_assets), dtype=float)
    for t, regime in enumerate(regimes):
        r = int(regime)
        returns[t, :] = rng.multivariate_normal(mean=mu_by_regime[r], cov=cov_by_regime[r])
    returns[0, :] = 0.0

    market = market_frame_from_returns(returns=returns, regimes=regimes, symbols=symbols, dates=dates)
    metadata = {
        "level": "level4_rare_crisis",
        "description": "Three-regime synthetic market with rare high-volatility crisis periods.",
        "seed": seed,
        "n_days": n_days,
        "n_assets": n_assets,
        "symbols": symbols,
        "start_date": start_date,
        "transition_matrix": transition.tolist(),
        "bull_mu_vector": mu_by_regime[0].tolist(),
        "bear_mu_vector": mu_by_regime[1].tolist(),
        "crisis_mu_vector": mu_by_regime[2].tolist(),
        "bull_vol": bull_vol,
        "bear_vol": bear_vol,
        "crisis_vol": crisis_vol,
        "bull_corr": bull_corr,
        "bear_corr": bear_corr,
        "crisis_corr": crisis_corr,
        "bull_covariance": cov_by_regime[0].tolist(),
        "bear_covariance": cov_by_regime[1].tolist(),
        "crisis_covariance": cov_by_regime[2].tolist(),
        "regime_counts": {
            "bull": int(np.sum(regimes == 0)),
            "bear": int(np.sum(regimes == 1)),
            "crisis": int(np.sum(regimes == 2)),
        },
        "n_switches": int(np.sum(regimes[1:] != regimes[:-1])),
    }
    return market, metadata


def make_labels(
    dates: pd.Series,
    methods: list[str],
    regime_labels: np.ndarray,
    regime_names: np.ndarray | list[str],
) -> pd.DataFrame:
    frames = []
    for method in methods:
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "method": method,
                    "regime_label": regime_labels.astype(int),
                    "regime_name": regime_names,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_level5_hidden_labels(
    market: pd.DataFrame,
    methods: list[str],
    primary_symbol: str,
    seed: int,
) -> pd.DataFrame:
    outputs = []
    feature_config = MarketFeatureConfig(date_col="date", primary_symbol=primary_symbol)

    if "rule_based" in methods:
        oracle = pd.DataFrame(
            {
                "date": market["date"],
                "method": "rule_based",
                "regime_label": market["true_regime_label"].astype(int),
                "regime_name": market["true_regime_name"].astype(str),
                "label_source": "oracle_true_regime",
                "true_regime_label": market["true_regime_label"].astype(int),
                "true_regime_name": market["true_regime_name"].astype(str),
            }
        )
        outputs.append(oracle)

    if "hmm" in methods:
        hmm = label_hmm(
            market,
            feature_config=feature_config,
            hmm_config=HMMConfig(n_states=3, random_seed=seed),
        )
        hmm["label_source"] = "estimated_hmm"
        hmm["true_regime_label"] = market["true_regime_label"].to_numpy(dtype=int)
        hmm["true_regime_name"] = market["true_regime_name"].astype(str).to_numpy()
        outputs.append(hmm)

    if "recap_cusum" in methods:
        recap = label_recap_cusum(
            market,
            feature_config=feature_config,
            cusum_config=RecapCusumConfig(
                reference_window=60,
                min_segment=20,
                drift=0.20,
                threshold=6.0,
            ),
        )
        recap["label_source"] = "estimated_recap_cusum"
        recap["true_regime_label"] = market["true_regime_label"].to_numpy(dtype=int)
        recap["true_regime_name"] = market["true_regime_name"].astype(str).to_numpy()
        outputs.append(recap)

    if not outputs:
        raise ValueError("Level 5 needs at least one label method.")

    return pd.concat(outputs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate controlled synthetic market CSVs for SAC/DQN replay experiments."
    )
    parser.add_argument(
        "--level",
        default="level0",
        choices=["level0", "stationary", "level1", "level2", "level3", "level4", "level5"],
    )
    parser.add_argument("--output-root", default="outputs/synthetic_market")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-days", type=int, default=1500)
    parser.add_argument("--n-assets", type=int, default=5)
    parser.add_argument("--start-date", default="2000-01-03")
    parser.add_argument("--daily-mu", type=float, default=0.00035)
    parser.add_argument("--daily-vol", type=float, default=0.012)
    parser.add_argument("--bull-mu", type=float, default=0.00075)
    parser.add_argument("--bear-mu", type=float, default=-0.00075)
    parser.add_argument("--crisis-mu", type=float, default=-0.0025)
    parser.add_argument("--bull-vol", type=float, default=0.010)
    parser.add_argument("--bear-vol", type=float, default=0.020)
    parser.add_argument("--crisis-vol", type=float, default=0.035)
    parser.add_argument("--corr", type=float, default=0.25)
    parser.add_argument("--bull-corr", type=float, default=0.15)
    parser.add_argument("--bear-corr", type=float, default=0.45)
    parser.add_argument("--crisis-corr", type=float, default=0.80)
    parser.add_argument("--switch-prob", type=float, default=0.035)
    parser.add_argument(
        "--label-methods",
        default="rule_based,hmm,recap_cusum",
        help="Comma-separated method names written into the synthetic label file.",
    )
    args = parser.parse_args()

    if args.n_assets < 2:
        raise ValueError("--n-assets must be at least 2.")
    if args.n_days < 100:
        raise ValueError("--n-days should be at least 100 for rolling features and RL smoke tests.")

    if args.level in {"level0", "stationary"}:
        market, metadata = generate_level0_stationary(
            n_days=args.n_days,
            n_assets=args.n_assets,
            seed=args.seed,
            start_date=args.start_date,
            daily_mu=args.daily_mu,
            daily_vol=args.daily_vol,
            corr=args.corr,
        )
        regime_labels = np.zeros(len(market), dtype=np.int64)
        regime_names = np.array(["stationary"] * len(market), dtype=object)
        default_prefix = f"level0_stationary_seed{args.seed}"
    elif args.level == "level1":
        market, metadata = generate_level1_mean_shift(
            n_days=args.n_days,
            n_assets=args.n_assets,
            seed=args.seed,
            start_date=args.start_date,
            bull_mu=args.bull_mu,
            bear_mu=args.bear_mu,
            daily_vol=args.daily_vol,
            corr=args.corr,
            switch_prob=args.switch_prob,
        )
        regime_labels = market["true_regime_label"].to_numpy(dtype=np.int64)
        regime_names = market["true_regime_name"].to_numpy(dtype=object)
        default_prefix = f"level1_mean_shift_seed{args.seed}"
    elif args.level == "level2":
        market, metadata = generate_level2_mean_vol_shift(
            n_days=args.n_days,
            n_assets=args.n_assets,
            seed=args.seed,
            start_date=args.start_date,
            bull_mu=args.bull_mu,
            bear_mu=args.bear_mu,
            bull_vol=args.bull_vol,
            bear_vol=args.bear_vol,
            corr=args.corr,
            switch_prob=args.switch_prob,
        )
        regime_labels = market["true_regime_label"].to_numpy(dtype=np.int64)
        regime_names = market["true_regime_name"].to_numpy(dtype=object)
        default_prefix = f"level2_mean_vol_shift_seed{args.seed}"
    elif args.level == "level3":
        market, metadata = generate_level3_mean_vol_corr_shift(
            n_days=args.n_days,
            n_assets=args.n_assets,
            seed=args.seed,
            start_date=args.start_date,
            bull_mu=args.bull_mu,
            bear_mu=args.bear_mu,
            bull_vol=args.bull_vol,
            bear_vol=args.bear_vol,
            bull_corr=args.bull_corr,
            bear_corr=args.bear_corr,
            switch_prob=args.switch_prob,
        )
        regime_labels = market["true_regime_label"].to_numpy(dtype=np.int64)
        regime_names = market["true_regime_name"].to_numpy(dtype=object)
        default_prefix = f"level3_mean_vol_corr_shift_seed{args.seed}"
    elif args.level == "level4":
        market, metadata = generate_level4_rare_crisis(
            n_days=args.n_days,
            n_assets=args.n_assets,
            seed=args.seed,
            start_date=args.start_date,
            bull_mu=args.bull_mu,
            bear_mu=args.bear_mu,
            crisis_mu=args.crisis_mu,
            bull_vol=args.bull_vol,
            bear_vol=args.bear_vol,
            crisis_vol=args.crisis_vol,
            bull_corr=args.bull_corr,
            bear_corr=args.bear_corr,
            crisis_corr=args.crisis_corr,
        )
        regime_labels = market["true_regime_label"].to_numpy(dtype=np.int64)
        regime_names = market["true_regime_name"].to_numpy(dtype=object)
        default_prefix = f"level4_rare_crisis_seed{args.seed}"
    else:
        market, metadata = generate_level4_rare_crisis(
            n_days=args.n_days,
            n_assets=args.n_assets,
            seed=args.seed,
            start_date=args.start_date,
            bull_mu=args.bull_mu,
            bear_mu=args.bear_mu,
            crisis_mu=args.crisis_mu,
            bull_vol=args.bull_vol,
            bear_vol=args.bear_vol,
            crisis_vol=args.crisis_vol,
            bull_corr=args.bull_corr,
            bear_corr=args.bear_corr,
            crisis_corr=args.crisis_corr,
        )
        metadata["level"] = "level5_hidden_or_estimated_regime"
        metadata["description"] = (
            "Level 4 rare-crisis market where true regimes are hidden from replay labels; "
            "HMM and ReCAP-CUSUM labels are estimated from market features. "
            "The rule_based label method is kept as oracle true-regime control."
        )
        regime_labels = market["true_regime_label"].to_numpy(dtype=np.int64)
        regime_names = market["true_regime_name"].to_numpy(dtype=object)
        default_prefix = f"level5_hidden_estimated_seed{args.seed}"

    methods = parse_methods(args.label_methods)
    if args.level == "level5":
        labels = make_level5_hidden_labels(market, methods, primary_symbol="SYN0", seed=args.seed)
        metadata["level5_label_sources"] = {
            "rule_based": "oracle true regime control",
            "hmm": "estimated from synthetic market features using Gaussian HMM",
            "recap_cusum": "estimated from synthetic market features using CUSUM-style segmentation",
        }
    else:
        labels = make_labels(market["date"], methods, regime_labels, regime_names)

    prefix = args.prefix or default_prefix
    out_dir = Path(args.output_root) / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    market_path = out_dir / "market.csv"
    labels_path = out_dir / "labels.csv"
    metadata_path = out_dir / "metadata.json"

    market.to_csv(market_path, index=False)
    labels.to_csv(labels_path, index=False)
    metadata["label_methods"] = methods
    metadata["market_csv"] = str(market_path)
    metadata["labels_csv"] = str(labels_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    symbols = ",".join(metadata["symbols"])
    primary = metadata["symbols"][0]
    print(f"[done] wrote synthetic market: {market_path}")
    print(f"[done] wrote synthetic labels: {labels_path}")
    print(f"[done] wrote metadata: {metadata_path}")
    print()
    print("Example SAC smoke command:")
    print(
        "python scripts/run_sac_replay.py "
        f"--market-csv {market_path} "
        f"--labels-csv {labels_path} "
        f"--tradable-symbols {symbols} "
        f"--primary-symbol {primary} "
        "--label-method rule_based "
        "--replays uniform,per,regime,deer "
        "--seeds 0 "
        "--warmup-steps 64 "
        "--start-steps 64 "
        "--batch-size 64 "
        "--hidden-dim 64 "
        "--max-steps 300 "
        f"--output-root outputs/sac_synthetic_{metadata['level']}_smoke"
    )


if __name__ == "__main__":
    main()
