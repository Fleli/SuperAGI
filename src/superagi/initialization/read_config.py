from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from superagi.model.transformer import TransformerConfig


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "specs" / "config.yaml"


@dataclass(frozen=True)
class InitConfig:
    weight_std: float
    scale_residual_projections: bool

    def __post_init__(self) -> None:
        if self.weight_std <= 0:
            raise ValueError("weight_std must be positive")


@dataclass(frozen=True)
class ModelParameters:
    n_layers: int
    dim_embedding: int
    dim_key: int
    ctx_window: int

    def __post_init__(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.dim_embedding <= 0:
            raise ValueError("dim_embedding must be positive")
        if self.dim_key <= 0:
            raise ValueError("dim_key must be positive")
        if self.ctx_window <= 0:
            raise ValueError("ctx_window must be positive")
        if self.dim_embedding % self.dim_key != 0:
            raise ValueError("dim_embedding must be divisible by dim_key")

    @property
    def n_heads(self) -> int:
        return self.dim_embedding // self.dim_key


@dataclass(frozen=True)
class ProjectConfig:
    init: InitConfig
    parameters: ModelParameters

    def to_transformer_config(self, vocab_size: int) -> TransformerConfig:
        return TransformerConfig(
            vocab_size=vocab_size,
            context_length=self.parameters.ctx_window,
            dim_embedding=self.parameters.dim_embedding,
            n_layers=self.parameters.n_layers,
            n_heads=self.parameters.n_heads,
            init_std=self.init.weight_std,
            scale_residual_projections=self.init.scale_residual_projections,
        )


def load_project_config(path: Path | str = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {config_path}")

    init = _required_mapping(payload, "init")
    parameters = _required_mapping(payload, "parameters")
    return ProjectConfig(
        init=InitConfig(
            weight_std=_required_number(init, "weight_std"),
            scale_residual_projections=_required_bool(
                init,
                "scale_residual_projections",
            ),
        ),
        parameters=ModelParameters(
            n_layers=_required_int(parameters, "n_layers"),
            dim_embedding=_required_int(parameters, "dim_embedding"),
            dim_key=_required_int(parameters, "dim_key"),
            ctx_window=_required_int(parameters, "ctx_window"),
        ),
    )


def _required_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"config field {key!r} must be a mapping")
    return value


def _required_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise ValueError(f"config field {key!r} must be an integer")
    return value


def _required_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float):
        raise ValueError(f"config field {key!r} must be numeric")
    return value


def _required_bool(payload: dict[str, Any], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"config field {key!r} must be boolean")
    return value
