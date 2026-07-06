from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RLModelSpec:
    name: str
    family: str
    action_space: str
    default_replays: tuple[str, ...]
    description: str


MODEL_REGISTRY: dict[str, RLModelSpec] = {
    "dqn": RLModelSpec(
        name="dqn",
        family="value_based",
        action_space="discrete_portfolio_templates",
        default_replays=("uniform", "per", "regime", "deer"),
        description="Discrete-action DQN baseline with replay diagnostics.",
    ),
    "sac": RLModelSpec(
        name="sac",
        family="actor_critic",
        action_space="continuous_softmax_portfolio_weights",
        default_replays=("uniform", "per", "regime", "deer"),
        description="Continuous-action SAC baseline with replay diagnostics.",
    ),
    "cash": RLModelSpec(
        name="cash",
        family="baseline",
        action_space="deterministic_portfolio_weights",
        default_replays=("policy",),
        description="All-cash capital-preservation baseline.",
    ),
    "equal_weight": RLModelSpec(
        name="equal_weight",
        family="baseline",
        action_space="deterministic_portfolio_weights",
        default_replays=("policy",),
        description="Equal-weight long-only asset baseline.",
    ),
    "regime_anchor": RLModelSpec(
        name="regime_anchor",
        family="baseline",
        action_space="regime_conditioned_portfolio_weights",
        default_replays=("policy",),
        description="ReCAP-inspired regime anchor policy library baseline.",
    ),
    "vol_target": RLModelSpec(
        name="vol_target",
        family="baseline",
        action_space="volatility_targeted_portfolio_weights",
        default_replays=("policy",),
        description="Equal-weight asset exposure scaled by realized volatility.",
    ),
}


def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


def validate_models(models: list[str]) -> list[str]:
    invalid = [model for model in models if model not in MODEL_REGISTRY]
    if invalid:
        raise ValueError(f"Invalid models: {invalid}. Valid models: {available_models()}")
    return models


def compatible_replays(model: str, requested: list[str]) -> list[str]:
    spec = MODEL_REGISTRY[model]
    if spec.family == "baseline":
        return ["policy"]
    allowed = set(spec.default_replays)
    if model == "dqn":
        allowed.add("online")
    return [replay for replay in requested if replay in allowed]
