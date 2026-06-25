from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.aggregator import FLUSH_COLUMNS, FLOW_FEATURES
from src.models.inference import (
    InferenceAlignmentError,
    InferenceArtifactError,
    LIVE_CONTINUOUS_FEATURES,
    REQUIRED_AGGREGATOR_COLUMNS,
    RealTimeInferenceEngine,
)
from src.models.lympha_net import CATEGORICAL_FEATURE_NAMES, LymphaNet
from src.utils.config_loader import ModelConfig, load_config


def _model_config(**overrides) -> ModelConfig:
    defaults = {
        "input_dim": 41,
        "continuous_dim": 40,
        "hidden_dims": [64, 32],
        "dropouts": [0.1, 0.1],
        "num_classes": 2,
        "use_batch_norm": False,
        "port_vocab_size": 1024,
        "proto_vocab_size": 64,
        "src_port_embed_dim": 8,
        "dst_port_embed_dim": 8,
        "proto_embed_dim": 4,
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _mock_flow_row(
    src_ip: str = "10.0.0.1",
    dst_ip: str = "10.0.0.2",
    src_port: float = 12345.0,
    dst_port: float = 80.0,
    protocol_type: float = 6.0,
    fill: float = 1.0,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {name: fill for name in FLOW_FEATURES}
    row["protocol_type"] = protocol_type
    row["src_port"] = src_port
    row["dst_port"] = dst_port
    row["src_ip"] = src_ip
    row["dst_ip"] = dst_ip
    return row


def _mock_flow_df(rows: int = 4) -> pd.DataFrame:
    data = [
        _mock_flow_row(
            src_ip=f"10.0.0.{index + 1}",
            dst_ip="10.0.0.99",
            src_port=10_000 + index,
            dst_port=80 + index,
            protocol_type=6.0,
            fill=1.0 + index,
        )
        for index in range(rows)
    ]
    return pd.DataFrame(data, columns=FLUSH_COLUMNS)


def _engine(
    threshold: float = 0.5,
    batch_size: int = 2,
    scaler: StandardScaler | None = None,
) -> RealTimeInferenceEngine:
    config = load_config()
    model = LymphaNet.from_config(config.model)
    fitted_scaler = scaler or RealTimeInferenceEngine.build_mock_scaler()
    return RealTimeInferenceEngine(
        config=config,
        model=model,
        scaler=fitted_scaler,
        device="cpu",
        threshold=threshold,
        batch_size=batch_size,
    )


def test_live_continuous_feature_layout():
    assert len(LIVE_CONTINUOUS_FEATURES) == 40
    assert set(LIVE_CONTINUOUS_FEATURES) == set(FLOW_FEATURES) - set(CATEGORICAL_FEATURE_NAMES)
    assert REQUIRED_AGGREGATOR_COLUMNS == tuple(FLUSH_COLUMNS)


def test_empty_dataframe_returns_no_profiles():
    engine = _engine()
    assert engine.predict_flows(pd.DataFrame(columns=FLUSH_COLUMNS)) == []


def test_missing_columns_raise_alignment_error():
    engine = _engine()
    df = _mock_flow_df(1).drop(columns=["src_ip"])
    with pytest.raises(InferenceAlignmentError, match="missing required columns"):
        engine.predict_flows(df)


def test_split_features_shapes_and_types():
    engine = _engine()
    df = _mock_flow_df(3)
    continuous, src_port, dst_port, protocol = engine.split_features(df)

    assert continuous.shape == (3, 40)
    assert continuous.dtype == np.float32
    assert src_port.shape == (3,)
    assert dst_port.shape == (3,)
    assert protocol.shape == (3,)
    assert np.array_equal(src_port, df["src_port"].to_numpy(dtype=np.int64))


def test_scale_continuous_uses_fitted_scaler():
    engine = _engine()
    df = _mock_flow_df(2)
    continuous, _, _, _ = engine.split_features(df)
    scaled = engine.scale_continuous(continuous)

    assert scaled.shape == (2, 40)
    assert scaled.dtype == np.float32
    assert np.isfinite(scaled).all()


def test_scaler_dimension_mismatch_raises():
    config = load_config()
    bad_scaler = StandardScaler()
    bad_scaler.fit(np.random.randn(10, 41))

    with pytest.raises(InferenceAlignmentError, match="Scaler expects"):
        RealTimeInferenceEngine(
            config=config,
            model=LymphaNet.from_config(config.model),
            scaler=bad_scaler,
            device="cpu",
        )


def test_out_of_bounds_ports_do_not_crash_inference():
    engine = _engine(batch_size=4)
    df = _mock_flow_df(2)
    df.loc[0, "src_port"] = 999_999
    df.loc[0, "dst_port"] = 888_888
    df.loc[0, "protocol_type"] = 9_999

    profiles = engine.predict_flows(df)
    assert len(profiles) == 2
    for profile in profiles:
        assert 0.0 <= profile.malicious_prob <= 1.0


def test_predict_flows_end_to_end_multi_batch():
    engine = _engine(threshold=0.5, batch_size=2)
    df = _mock_flow_df(5)
    profiles = engine.predict_flows(df)

    assert len(profiles) == 5
    for profile in profiles:
        assert isinstance(profile.src_ip, str)
        assert isinstance(profile.dst_ip, str)
        assert 0.0 <= profile.malicious_prob <= 1.0
        assert profile.should_block == (profile.malicious_prob >= 0.5)


def test_threshold_controls_should_block():
    engine = _engine(threshold=0.0, batch_size=8)
    df = _mock_flow_df(3)
    profiles = engine.predict_flows(df)
    assert all(profile.should_block for profile in profiles)


def test_legacy_scaler_artifact_raises_clear_error():
    config = load_config()
    scaler_path = config.artifacts.scaler
    if not scaler_path.exists():
        pytest.skip("Legacy scaler artifact not present")

    with pytest.raises(InferenceAlignmentError):
        RealTimeInferenceEngine(config=config, device="cpu")


def test_missing_scaler_raises_artifact_error(tmp_path: Path):
    config = load_config()
    missing = tmp_path / "missing_scaler.pkl"
    config = config.model_copy(
        update={
            "artifacts": config.artifacts.model_copy(update={"scaler": missing}),
        },
    )
    with pytest.raises(InferenceArtifactError, match="Scaler artifact not found"):
        RealTimeInferenceEngine(config=config, device="cpu")


def test_model_eval_and_no_grad_during_inference():
    engine = _engine(batch_size=4)
    engine._model.train()
    df = _mock_flow_df(2)

    for param in engine._model.parameters():
        param.grad = None

    profiles = engine.predict_flows(df)
    assert len(profiles) == 2
    assert not engine._model.training


def test_device_cpu_compliance():
    engine = _engine(batch_size=2)
    assert engine.device.type == "cpu"
    profiles = engine.predict_flows(_mock_flow_df(2))
    assert len(profiles) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_cuda_compliance():
    config = load_config()
    model = LymphaNet.from_config(config.model)
    scaler = RealTimeInferenceEngine.build_mock_scaler()
    engine = RealTimeInferenceEngine(
        config=config,
        model=model,
        scaler=scaler,
        device="cuda",
        batch_size=2,
    )
    profiles = engine.predict_flows(_mock_flow_df(2))
    assert len(profiles) == 2
    assert next(engine._model.parameters()).device.type == "cuda"
