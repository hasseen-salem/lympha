from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.aggregator import FLOW_FEATURES
from src.models.lympha_net import (
    LymphaBatch,
    LymphaNet,
    TrafficClassifier,
    clamp_indices,
    fused_input_dim,
)
from src.utils.config_loader import ModelConfig, load_config


def _model_config(**overrides) -> ModelConfig:
    defaults = {
        "input_dim": 41,
        "continuous_dim": 40,
        "hidden_dims": [512, 256, 128, 64],
        "dropouts": [0.3, 0.3, 0.2, 0.1],
        "num_classes": 2,
        "use_batch_norm": True,
        "port_vocab_size": 65536,
        "proto_vocab_size": 256,
        "src_port_embed_dim": 16,
        "dst_port_embed_dim": 16,
        "proto_embed_dim": 8,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _dummy_batch(batch_size: int = 8, config: ModelConfig | None = None) -> LymphaBatch:
    config = config or _model_config()
    return LymphaBatch(
        continuous=torch.randn(batch_size, config.continuous_dim),
        src_port=torch.randint(0, 1024, (batch_size,)),
        dst_port=torch.randint(0, 1024, (batch_size,)),
        protocol=torch.randint(0, 255, (batch_size,)),
    )


def test_fused_input_dim_matches_config():
    config = _model_config()
    model = LymphaNet.from_config(config)
    assert fused_input_dim(config) == 40 + 16 + 16 + 8
    assert model.head[0].in_features == fused_input_dim(config)


def test_forward_output_shape():
    config = _model_config()
    model = LymphaNet.from_config(config)
    batch = _dummy_batch(batch_size=16, config=config)

    logits = model.forward_batch(batch)
    assert logits.shape == (16, config.num_classes)


def test_embedding_concatenation_dimension():
    config = _model_config()
    model = LymphaNet.from_config(config)
    batch = _dummy_batch(batch_size=4, config=config)

    categorical = model.embed_categorical(batch.src_port, batch.dst_port, batch.protocol)
    assert categorical.shape == (4, config.src_port_embed_dim + config.dst_port_embed_dim + config.proto_embed_dim)

    fused = torch.cat([batch.continuous, categorical], dim=-1)
    assert fused.shape == (4, fused_input_dim(config))


def test_continuous_dim_mismatch_raises():
    config = _model_config(continuous_dim=40)
    model = LymphaNet.from_config(config)
    batch = _dummy_batch(batch_size=2, config=config)
    bad_continuous = torch.randn(2, 32)

    with pytest.raises(ValueError, match="continuous feature dim mismatch"):
        model.forward(bad_continuous, batch.src_port, batch.dst_port, batch.protocol)


def test_out_of_bounds_categorical_indices_are_clamped():
    config = _model_config(port_vocab_size=1024, proto_vocab_size=64)
    model = LymphaNet.from_config(config)
    batch = _dummy_batch(batch_size=3, config=config)

    oob_src = torch.tensor([0, 1023, 999_999], dtype=torch.long)
    oob_dst = torch.tensor([0, 500, 80_000], dtype=torch.long)
    oob_proto = torch.tensor([0, 63, 9_999], dtype=torch.long)

    logits = model.forward(batch.continuous, oob_src, oob_dst, oob_proto)
    assert logits.shape == (3, config.num_classes)
    assert torch.equal(clamp_indices(oob_src, 1024), torch.tensor([0, 1023, 1023]))
    assert torch.equal(clamp_indices(oob_proto, 64), torch.tensor([0, 63, 63]))


def test_gradient_propagation():
    config = _model_config()
    model = LymphaNet.from_config(config)
    batch = _dummy_batch(batch_size=4, config=config)
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    model.train()
    logits = model.forward_batch(batch)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert model.src_port_embedding.weight.grad is not None
    assert model.dst_port_embedding.weight.grad is not None
    assert model.proto_embedding.weight.grad is not None
    assert model.head[0].weight.grad is not None
    assert not torch.isnan(model.src_port_embedding.weight.grad).any()


def test_device_movement_cpu():
    config = _model_config()
    model = LymphaNet.from_config(config).cpu()
    batch = _dummy_batch(batch_size=2, config=config)
    batch = LymphaBatch(
        continuous=batch.continuous.cpu(),
        src_port=batch.src_port.cpu(),
        dst_port=batch.dst_port.cpu(),
        protocol=batch.protocol.cpu(),
    )
    logits = model.forward_batch(batch)
    assert logits.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_movement_cuda():
    config = _model_config()
    model = LymphaNet.from_config(config).cuda()
    batch = _dummy_batch(batch_size=2, config=config)
    batch = LymphaBatch(
        continuous=batch.continuous.cuda(),
        src_port=batch.src_port.cuda(),
        dst_port=batch.dst_port.cuda(),
        protocol=batch.protocol.cuda(),
    )
    logits = model.forward_batch(batch)
    assert logits.device.type == "cuda"


def test_forward_batch_matches_explicit_forward():
    config = _model_config()
    model = LymphaNet.from_config(config)
    model.eval()
    batch = _dummy_batch(batch_size=5, config=config)

    with torch.no_grad():
        explicit = model.forward(batch.continuous, batch.src_port, batch.dst_port, batch.protocol)
        wrapped = model.forward_batch(batch)
    assert torch.allclose(explicit, wrapped)


def test_model_from_loaded_config():
    cfg = load_config()
    model = LymphaNet.from_config(cfg.model)
    batch = _dummy_batch(batch_size=3, config=cfg.model)
    logits = model.forward_batch(batch)
    assert logits.shape == (3, cfg.model.num_classes)


def test_live_aggregator_continuous_feature_count():
    categorical = set(("protocol_type", "src_port", "dst_port"))
    continuous_count = len([name for name in FLOW_FEATURES if name not in categorical])
    assert continuous_count == 40

    config = _model_config(continuous_dim=continuous_count)
    model = LymphaNet.from_config(config)
    model.eval()
    batch = _dummy_batch(batch_size=2, config=config)
    with torch.no_grad():
        assert model.forward_batch(batch).shape == (2, 2)


def test_legacy_traffic_classifier_forward():
    model = TrafficClassifier(input_dim=41)
    x = torch.randn(8, 41)
    logits = model(x)
    assert logits.shape == (8, 2)


def test_eval_mode_disables_dropout_behavior():
    config = _model_config()
    model = LymphaNet.from_config(config)
    batch = _dummy_batch(batch_size=4, config=config)

    model.train()
    train_out = model.forward_batch(batch)

    model.eval()
    with torch.no_grad():
        eval_out = model.forward_batch(batch)

    assert train_out.shape == eval_out.shape
