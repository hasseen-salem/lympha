from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.preprocessing import StandardScaler

from src.features.aggregator import FLUSH_COLUMNS, FLOW_FEATURES
from src.models.lympha_net import CATEGORICAL_FEATURE_NAMES, LymphaBatch, LymphaNet
from src.utils.config_loader import AppConfig, load_config, resolve_device
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

F32_MAX = float(np.finfo(np.float32).max)

# Aggregator output layout: 43 flow metrics + src_ip + dst_ip.
REQUIRED_AGGREGATOR_COLUMNS: tuple[str, ...] = tuple(FLUSH_COLUMNS)

# 40 continuous columns fed to StandardScaler (excludes protocol_type, src_port, dst_port).
LIVE_CONTINUOUS_FEATURES: tuple[str, ...] = tuple(
    name for name in FLOW_FEATURES if name not in CATEGORICAL_FEATURE_NAMES
)


class InferenceError(RuntimeError):
    """Base inference pipeline failure."""


class InferenceAlignmentError(InferenceError):
    """Raised when aggregator columns or scaler dimensions do not match expectations."""


class InferenceArtifactError(InferenceError):
    """Raised when model or scaler artifacts cannot be loaded safely."""


@dataclass(frozen=True)
class ThreatProfile:
    src_ip: str
    dst_ip: str
    malicious_prob: float
    should_block: bool


class RealTimeInferenceEngine:
    """Live inference bridge between FlowAggregator output and LymphaNet."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        model: LymphaNet | None = None,
        scaler: StandardScaler | None = None,
        device: str | torch.device | None = None,
        threshold: float | None = None,
        batch_size: int | None = None,
        require_weights: bool = False,
    ) -> None:
        self._config = config or load_config()
        self._threshold = (
            threshold
            if threshold is not None
            else self._config.execution.detection_threshold
        )
        self._batch_size = (
            batch_size
            if batch_size is not None
            else self._config.evaluation.batch_size
        )
        self._device = torch.device(
            device if device is not None else resolve_device(self._config.training.device),
        )

        self._model = (model or LymphaNet.from_config(self._config.model)).to(self._device)
        self._scaler = scaler or self._load_scaler()
        self._validate_scaler_alignment()

        if model is None:
            self._load_model_weights(require_weights=require_weights)

        self._model.eval()

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def continuous_features(self) -> tuple[str, ...]:
        return LIVE_CONTINUOUS_FEATURES

    @property
    def required_columns(self) -> tuple[str, ...]:
        return REQUIRED_AGGREGATOR_COLUMNS

    def _load_scaler(self) -> StandardScaler:
        scaler_path = self._config.artifacts.scaler
        if not scaler_path.exists():
            raise InferenceArtifactError(f"Scaler artifact not found: {scaler_path}")

        scaler = joblib.load(scaler_path)
        if not hasattr(scaler, "transform"):
            raise InferenceArtifactError(
                f"Scaler artifact at {scaler_path} does not expose transform()",
            )
        logger.info("Loaded scaler from %s", scaler_path)
        return scaler

    def _validate_scaler_alignment(self) -> None:
        expected = self._config.model.continuous_dim
        actual = getattr(self._scaler, "n_features_in_", None)
        if actual is None:
            raise InferenceAlignmentError(
                "Scaler is missing n_features_in_; fit the scaler before deployment",
            )
        if actual != expected:
            raise InferenceAlignmentError(
                f"Scaler expects {actual} features but LymphaNet continuous_dim is {expected}. "
                f"Retrain or export a scaler aligned to LIVE_CONTINUOUS_FEATURES ({len(LIVE_CONTINUOUS_FEATURES)} columns).",
            )
        if actual != len(LIVE_CONTINUOUS_FEATURES):
            raise InferenceAlignmentError(
                f"Scaler feature count ({actual}) does not match live continuous layout "
                f"({len(LIVE_CONTINUOUS_FEATURES)})",
            )

    def _load_model_weights(self, *, require_weights: bool) -> None:
        weights_path = self._config.artifacts.model_weights
        if not weights_path.exists():
            message = f"Model weights not found: {weights_path}"
            if require_weights:
                raise InferenceArtifactError(message)
            logger.warning("%s — using initialized LymphaNet parameters", message)
            return

        state_dict = load_file(str(weights_path), device=str(self._device))
        try:
            self._model.load_state_dict(state_dict, strict=True)
            logger.info("Loaded LymphaNet weights from %s", weights_path)
        except RuntimeError as exc:
            message = (
                f"Weights at {weights_path} are incompatible with LymphaNet "
                f"(legacy flat checkpoints require re-export for the embedding architecture)"
            )
            if require_weights:
                raise InferenceArtifactError(message) from exc
            logger.warning("%s: %s", message, exc)

    def validate_dataframe(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        missing = [col for col in REQUIRED_AGGREGATOR_COLUMNS if col not in df.columns]
        if missing:
            raise InferenceAlignmentError(
                f"Aggregator dataframe missing required columns: {missing}",
            )

        extra = [col for col in df.columns if col not in REQUIRED_AGGREGATOR_COLUMNS]
        if extra:
            logger.debug("Ignoring extra dataframe columns not used for inference: %s", extra)

    def split_features(
        self,
        df: pd.DataFrame,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        self.validate_dataframe(df)

        continuous = df.loc[:, LIVE_CONTINUOUS_FEATURES].to_numpy(dtype=np.float32, copy=True)
        continuous = np.clip(continuous, -F32_MAX, F32_MAX).astype(np.float32, copy=False)

        src_port = df["src_port"].to_numpy(dtype=np.int64, copy=False)
        dst_port = df["dst_port"].to_numpy(dtype=np.int64, copy=False)
        protocol = df["protocol_type"].to_numpy(dtype=np.int64, copy=False)

        return continuous, src_port, dst_port, protocol

    def scale_continuous(self, continuous: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        if continuous.size == 0:
            return continuous
        scaled = self._scaler.transform(continuous)
        return np.asarray(scaled, dtype=np.float32)

    def _predict_batch(self, batch: LymphaBatch) -> torch.Tensor:
        with torch.no_grad():
            logits = self._model.forward_batch(batch)
            return torch.softmax(logits, dim=-1)[:, 1]

    def predict_flows(self, df: pd.DataFrame) -> list[ThreatProfile]:
        if df.empty:
            return []

        self._model.eval()
        self.validate_dataframe(df)

        continuous, src_port, dst_port, protocol = self.split_features(df)
        scaled = self.scale_continuous(continuous)

        profiles: list[ThreatProfile] = []
        src_ips = df["src_ip"].astype(str).tolist()
        dst_ips = df["dst_ip"].astype(str).tolist()
        total = len(df)

        for start in range(0, total, self._batch_size):
            end = min(start + self._batch_size, total)
            batch = LymphaBatch(
                continuous=torch.tensor(scaled[start:end], dtype=torch.float32, device=self._device),
                src_port=torch.tensor(src_port[start:end], dtype=torch.long, device=self._device),
                dst_port=torch.tensor(dst_port[start:end], dtype=torch.long, device=self._device),
                protocol=torch.tensor(protocol[start:end], dtype=torch.long, device=self._device),
            )
            malicious_probs = self._predict_batch(batch).detach().cpu().numpy()

            for index, prob in enumerate(malicious_probs):
                row = start + index
                malicious_prob = float(prob)
                profiles.append(
                    ThreatProfile(
                        src_ip=src_ips[row],
                        dst_ip=dst_ips[row],
                        malicious_prob=malicious_prob,
                        should_block=malicious_prob >= self._threshold,
                    ),
                )

        return profiles

    @classmethod
    def from_artifacts(
        cls,
        config: AppConfig | None = None,
        *,
        require_weights: bool = False,
    ) -> RealTimeInferenceEngine:
        return cls(config=config, require_weights=require_weights)

    @staticmethod
    def build_mock_scaler(
        feature_count: int | None = None,
        seed: int = 42,
    ) -> StandardScaler:
        """Utility for tests and dry-runs when a live 40-dim scaler is not yet exported."""
        count = feature_count or len(LIVE_CONTINUOUS_FEATURES)
        rng = np.random.default_rng(seed)
        samples = rng.normal(size=(64, count)).astype(np.float32)
        scaler = StandardScaler()
        scaler.fit(samples)
        return scaler
