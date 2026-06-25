from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_PATH_FIELDS: frozenset[str] = frozenset({
    "dataset_path",
    "train_split_path",
    "test_split_path",
    "dataset_marker_path",
    "raw_path",
    "processed_path",
    "test_data_path",
    "model_weights",
    "scaler",
    "model_info",
    "feature_pipeline",
    "supervised_model",
    "unsupervised_model",
    "directory",
    "simulation_file",
})


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def config_path() -> Path:
    return project_root() / "config" / "settings.yaml"


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class DataConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_path: Path
    train_split_path: Path
    test_split_path: Path
    dataset_marker_path: Path
    raw_path: Path
    processed_path: Path
    columns_to_drop: list[str]
    label_column: str
    test_size: float = Field(gt=0.0, lt=1.0)
    random_state: int
    stratify: bool = True


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_dim: int = Field(ge=1)
    continuous_dim: int = Field(default=40, ge=1)
    hidden_dims: list[int]
    dropouts: list[float]
    num_classes: int = Field(ge=2)
    use_batch_norm: bool = True
    port_vocab_size: int = Field(default=65536, ge=2)
    proto_vocab_size: int = Field(default=256, ge=2)
    src_port_embed_dim: int = Field(default=16, ge=1)
    dst_port_embed_dim: int = Field(default=16, ge=1)
    proto_embed_dim: int = Field(default=8, ge=1)

    @field_validator("hidden_dims")
    @classmethod
    def _positive_hidden_dims(cls, v: list[int]) -> list[int]:
        if not v or any(d < 1 for d in v):
            raise ValueError("hidden_dims must be a non-empty list of positive integers")
        return v

    @field_validator("dropouts")
    @classmethod
    def _valid_dropouts(cls, v: list[float]) -> list[float]:
        if any(d < 0.0 or d >= 1.0 for d in v):
            raise ValueError("dropouts must be in [0.0, 1.0)")
        return v

    @model_validator(mode="after")
    def _dims_match_dropouts(self) -> ModelConfig:
        if len(self.hidden_dims) != len(self.dropouts):
            raise ValueError("hidden_dims and dropouts must have equal length")
        return self


class SchedulerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["reduce_on_plateau"] = "reduce_on_plateau"
    mode: Literal["min", "max"] = "min"
    factor: float = Field(gt=0.0, lt=1.0)
    patience: int = Field(ge=1)


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    directory: Path
    interval_epochs: int = Field(ge=1)
    save_best: bool = True


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    seed: int
    class_weights: list[float]
    optimizer: Literal["adam"] = "adam"
    scheduler: SchedulerConfig
    checkpoint: CheckpointConfig
    device: Literal["auto", "cuda", "cpu"] = "auto"

    @field_validator("class_weights")
    @classmethod
    def _positive_weights(cls, v: list[float]) -> list[float]:
        if not v or any(w <= 0 for w in v):
            raise ValueError("class_weights must be a non-empty list of positive floats")
        return v

    @model_validator(mode="after")
    def _weights_match_classes(self) -> TrainingConfig:
        return self


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_size: int = Field(ge=1)
    default_threshold: float = Field(ge=0.0, le=1.0)
    threshold_sweep: list[float]
    test_data_path: Path

    @field_validator("threshold_sweep")
    @classmethod
    def _valid_thresholds(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("threshold_sweep must be non-empty")
        for t in v:
            if t < 0.0 or t > 1.0:
                raise ValueError("threshold_sweep values must be in [0.0, 1.0]")
        return v


class ArtifactsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_weights: Path
    scaler: Path
    model_info: Path
    feature_pipeline: Path
    supervised_model: Path
    unsupervised_model: Path


class FeaturesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_features: list[str]
    scaler_type: Literal["robust", "standard"] = "robust"

    @field_validator("selected_features")
    @classmethod
    def _non_empty_features(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("selected_features must be non-empty")
        return v


class LightGBMConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_estimators: int = Field(ge=1)
    max_depth: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    num_leaves: int = Field(ge=2)
    subsample: float = Field(gt=0.0, le=1.0)
    colsample_bytree: float = Field(gt=0.0, le=1.0)
    min_child_samples: int = Field(ge=1)
    random_state: int = 42


class IsolationForestConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    n_estimators: int = Field(ge=1)
    contamination: float = Field(gt=0.0, lt=1.0)
    max_samples: str | int | float = "auto"
    random_state: int = 42


class LegacyModelsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    lightgbm: LightGBMConfig
    isolation_forest: IsolationForestConfig


class SnifferConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    interface: str
    bpf_filter: str = "ip"
    queue_maxsize: int = Field(ge=1)
    max_packets: int = Field(ge=0)
    recv_timeout_sec: float = Field(gt=0.0)
    simulation_file: Path | None = None
    backpressure_warn_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    backpressure_critical_ratio: float = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _warn_below_critical(self) -> SnifferConfig:
        if self.backpressure_warn_ratio >= self.backpressure_critical_ratio:
            raise ValueError(
                "backpressure_warn_ratio must be less than backpressure_critical_ratio",
            )
        return self


class AggregatorConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    idle_timeout_sec: float = Field(gt=0.0)
    hard_timeout_sec: float | None = Field(default=None, gt=0.0)
    max_flows: int = Field(default=10_000, ge=1)
    flush_interval_sec: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _default_hard_timeout(self) -> AggregatorConfig:
        if self.hard_timeout_sec is None:
            return self.model_copy(
                update={"hard_timeout_sec": self.idle_timeout_sec * 2.0},
            )
        if self.hard_timeout_sec < self.idle_timeout_sec:
            raise ValueError("hard_timeout_sec must be >= idle_timeout_sec")
        return self


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_mode: bool = True
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    flush_interval_sec: float = Field(gt=0.0)
    detection_threshold: float = Field(ge=0.0, le=1.0)
    detection_event_limit: int = Field(ge=1)
    inference_backend: Literal["neural", "ensemble", "lightgbm"] = "neural"


class MitigationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    table_name: str
    set_name: str
    chain_name: str
    block_timeout_sec: int = Field(ge=0)
    whitelist_ips: list[str]
    whitelist_prefixes: list[str]


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    artifacts: ArtifactsConfig
    features: FeaturesConfig
    models: LegacyModelsConfig
    sniffer: SnifferConfig
    aggregator: AggregatorConfig
    execution: ExecutionConfig
    mitigation: MitigationConfig

    @model_validator(mode="after")
    def _class_weights_match_num_classes(self) -> AppConfig:
        if len(self.training.class_weights) != self.model.num_classes:
            raise ValueError(
                "training.class_weights length must match model.num_classes",
            )
        return self

    @property
    def resolved_device(self) -> str:
        return resolve_device(self.training.device)


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _resolve_paths_in_dict(data: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            resolved[key] = _resolve_paths_in_dict(value, root)
        elif key in _PATH_FIELDS and value is None:
            resolved[key] = None
        elif key in _PATH_FIELDS and isinstance(value, str):
            resolved[key] = _resolve_path(root, value)
        else:
            resolved[key] = value
    return resolved


def _ensure_directories(cfg: AppConfig) -> None:
    dirs = {
        cfg.data.raw_path,
        cfg.data.processed_path,
        cfg.training.checkpoint.directory,
        cfg.artifacts.model_weights.parent,
        cfg.artifacts.scaler.parent,
        cfg.artifacts.feature_pipeline.parent,
    }
    for directory in dirs:
        os.makedirs(directory, mode=0o755, exist_ok=True)


def load_config(path: str | Path | None = None) -> AppConfig:
    root = project_root()
    cfg_path = Path(path) if path else config_path()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with open(cfg_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if raw is None:
        raise ValueError(f"Empty or invalid configuration file: {cfg_path}")

    resolved = _resolve_paths_in_dict(raw, root)
    cfg = AppConfig.model_validate(resolved)
    _ensure_directories(cfg)
    return cfg
