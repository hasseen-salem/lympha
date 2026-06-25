from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from src.utils.config_loader import ModelConfig

# Live aggregator categorical columns (excluded from the continuous tensor).
CATEGORICAL_FEATURE_NAMES: tuple[str, ...] = ("protocol_type", "src_port", "dst_port")

# Offline parquet categorical columns (NF-UNSW-NB15-v2 layout).
PARQUET_CATEGORICAL_COLUMNS: tuple[str, ...] = ("L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL")


class LymphaBatch(NamedTuple):
    continuous: torch.Tensor
    src_port: torch.Tensor
    dst_port: torch.Tensor
    protocol: torch.Tensor


def clamp_indices(indices: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map out-of-bounds categorical tokens to the last valid bucket."""
    max_index = vocab_size - 1
    return torch.clamp(indices.to(dtype=torch.long), min=0, max=max_index)


def fused_input_dim(config: ModelConfig) -> int:
    return (
        config.continuous_dim
        + config.src_port_embed_dim
        + config.dst_port_embed_dim
        + config.proto_embed_dim
    )


class LymphaNet(nn.Module):
    """Embedding-augmented MLP for live flow classification."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.src_port_embedding = nn.Embedding(config.port_vocab_size, config.src_port_embed_dim)
        self.dst_port_embedding = nn.Embedding(config.port_vocab_size, config.dst_port_embed_dim)
        self.proto_embedding = nn.Embedding(config.proto_vocab_size, config.proto_embed_dim)

        layers: list[nn.Module] = []
        in_features = fused_input_dim(config)

        for hidden_dim, dropout in zip(config.hidden_dims, config.dropouts, strict=True):
            layers.append(nn.Linear(in_features, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, config.num_classes))
        self.head = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_config(cls, config: ModelConfig) -> LymphaNet:
        return cls(config)

    def embed_categorical(
        self,
        src_port: torch.Tensor,
        dst_port: torch.Tensor,
        protocol: torch.Tensor,
    ) -> torch.Tensor:
        src_idx = clamp_indices(src_port, self.config.port_vocab_size)
        dst_idx = clamp_indices(dst_port, self.config.port_vocab_size)
        proto_idx = clamp_indices(protocol, self.config.proto_vocab_size)

        src_emb = self.src_port_embedding(src_idx)
        dst_emb = self.dst_port_embedding(dst_idx)
        proto_emb = self.proto_embedding(proto_idx)
        return torch.cat([src_emb, dst_emb, proto_emb], dim=-1)

    def forward(
        self,
        continuous: torch.Tensor,
        src_port: torch.Tensor,
        dst_port: torch.Tensor,
        protocol: torch.Tensor,
    ) -> torch.Tensor:
        if continuous.dim() != 2:
            raise ValueError(f"continuous must be 2-D (batch, features); got {continuous.dim()}-D")
        if continuous.size(-1) != self.config.continuous_dim:
            raise ValueError(
                f"continuous feature dim mismatch: expected {self.config.continuous_dim}, "
                f"got {continuous.size(-1)}",
            )

        batch_size = continuous.size(0)
        for name, tensor in (
            ("src_port", src_port),
            ("dst_port", dst_port),
            ("protocol", protocol),
        ):
            if tensor.dim() != 1 or tensor.size(0) != batch_size:
                raise ValueError(
                    f"{name} must be 1-D with batch size {batch_size}; got shape {tuple(tensor.shape)}",
                )

        categorical = self.embed_categorical(src_port, dst_port, protocol)
        fused = torch.cat([continuous, categorical], dim=-1)
        return self.head(fused)

    def forward_batch(self, batch: LymphaBatch) -> torch.Tensor:
        return self.forward(batch.continuous, batch.src_port, batch.dst_port, batch.protocol)


class TrafficClassifier(nn.Module):
    """Legacy flat MLP for pre-trained safetensors checkpoints (parquet offline path)."""

    def __init__(self, input_dim: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig(
                input_dim=input_dim,
                continuous_dim=max(input_dim - 3, 1),
                hidden_dims=[512, 256, 128, 64],
                dropouts=[0.3, 0.3, 0.2, 0.1],
                num_classes=2,
            )

        layers: list[nn.Module] = []
        in_features = input_dim
        for hidden_dim, dropout in zip(config.hidden_dims, config.dropouts, strict=True):
            layers.append(nn.Linear(in_features, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, config.num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
