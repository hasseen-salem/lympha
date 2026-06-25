from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config_loader import (
    AppConfig,
    DataConfig,
    ExecutionConfig,
    IsolationForestConfig,
    LegacyModelsConfig,
    LightGBMConfig,
    MitigationConfig,
    ModelConfig,
    TrainingConfig,
    load_config,
    project_root,
    resolve_device,
)


def test_config_loader_loads_default():
    cfg = load_config()
    assert isinstance(cfg, AppConfig)
    assert isinstance(cfg.data, DataConfig)
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.models, LegacyModelsConfig)
    assert isinstance(cfg.models.lightgbm, LightGBMConfig)
    assert isinstance(cfg.models.isolation_forest, IsolationForestConfig)
    assert isinstance(cfg.execution, ExecutionConfig)
    assert isinstance(cfg.mitigation, MitigationConfig)

    assert cfg.data.test_size == 0.2
    assert cfg.data.random_state == 42
    assert cfg.training.batch_size == 128
    assert cfg.training.epochs == 30
    assert cfg.training.learning_rate == 0.001
    assert cfg.training.class_weights == [1.0, 25.0]
    assert cfg.model.hidden_dims == [512, 256, 128, 64]
    assert cfg.model.dropouts == [0.3, 0.3, 0.2, 0.1]
    assert cfg.execution.simulation_mode is True
    assert cfg.execution.log_level == "INFO"
    assert cfg.mitigation.table_name == "lympha"

    assert cfg.data.raw_path.exists()
    assert cfg.data.processed_path.exists()
    assert cfg.training.checkpoint.directory.exists()


def test_config_loader_resolves_artifact_paths():
    cfg = load_config()
    weights = cfg.artifacts.model_weights
    assert weights.name == "model.safetensors"
    assert weights.parent.name == "output"
    assert str(weights).endswith("output/model.safetensors")
    assert weights.exists()


def test_config_loader_creates_directories():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        yaml_content = {
            "data": {
                "dataset_path": "Training_data/dataset.parquet",
                "train_split_path": "custom_processed/train.parquet",
                "test_split_path": "custom_processed/test.parquet",
                "dataset_marker_path": "custom_processed/.marker",
                "raw_path": "custom_raw",
                "processed_path": "custom_processed",
                "columns_to_drop": ["Attack"],
                "label_column": "Label",
                "test_size": 0.3,
                "random_state": 99,
                "stratify": True,
            },
            "model": {
                "input_dim": 41,
                "hidden_dims": [256, 128],
                "dropouts": [0.3, 0.2],
                "num_classes": 2,
                "use_batch_norm": True,
            },
            "training": {
                "batch_size": 64,
                "epochs": 10,
                "learning_rate": 0.01,
                "seed": 1,
                "class_weights": [1.0, 10.0],
                "optimizer": "adam",
                "scheduler": {
                    "name": "reduce_on_plateau",
                    "mode": "min",
                    "factor": 0.5,
                    "patience": 2,
                },
                "checkpoint": {
                    "enabled": True,
                    "directory": "custom_checkpoints",
                    "interval_epochs": 2,
                    "save_best": True,
                },
                "device": "cpu",
            },
            "evaluation": {
                "batch_size": 128,
                "default_threshold": 0.5,
                "threshold_sweep": [0.5, 0.9],
                "test_data_path": "custom_processed/test.parquet",
            },
            "artifacts": {
                "model_weights": "custom_output/model.safetensors",
                "scaler": "custom_output/scaler.pkl",
                "model_info": "custom_output/model_info.txt",
                "feature_pipeline": "custom_models/feature_pipeline.joblib",
                "supervised_model": "custom_models/supervised.joblib",
                "unsupervised_model": "custom_models/unsupervised.joblib",
            },
            "features": {
                "selected_features": ["packet_length", "ttl"],
                "scaler_type": "standard",
            },
            "models": {
                "lightgbm": {
                    "n_estimators": 50,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "num_leaves": 15,
                    "subsample": 0.7,
                    "colsample_bytree": 0.7,
                    "min_child_samples": 10,
                },
                "isolation_forest": {
                    "n_estimators": 50,
                    "contamination": 0.05,
                    "max_samples": "auto",
                },
            },
            "sniffer": {
                "interface": "lo",
                "bpf_filter": "ip",
                "queue_maxsize": 1000,
                "max_packets": 0,
                "recv_timeout_sec": 0.1,
            },
            "aggregator": {
                "idle_timeout_sec": 3.0,
            },
            "execution": {
                "simulation_mode": False,
                "log_level": "DEBUG",
                "flush_interval_sec": 1,
                "detection_threshold": 0.7,
                "detection_event_limit": 25,
                "inference_backend": "ensemble",
            },
            "mitigation": {
                "enabled": False,
                "table_name": "test_table",
                "set_name": "test_set",
                "chain_name": "input",
                "block_timeout_sec": 60,
                "whitelist_ips": ["127.0.0.1"],
                "whitelist_prefixes": ["224."],
            },
        }
        cfg_path = tmp_path / "custom_settings.yaml"
        with open(cfg_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(yaml_content, handle)

        cfg = load_config(cfg_path)
        assert cfg.training.batch_size == 64
        assert cfg.aggregator.hard_timeout_sec == 6.0
        assert Path(cfg.data.raw_path).exists()
        assert Path(cfg.data.processed_path).exists()
        assert str(cfg.data.raw_path).endswith("custom_raw")


def test_config_loader_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path.yaml")


def test_config_loader_raises_on_empty_file():
    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", encoding="utf-8") as handle:
        handle.write("")
        handle.flush()
        with pytest.raises(ValueError, match="Empty or invalid"):
            load_config(handle.name)


def test_config_is_frozen():
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.training.batch_size = 64


def test_class_weights_must_match_num_classes():
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "bad.yaml"
        base = yaml.safe_load((project_root() / "config" / "settings.yaml").read_text())
        base["training"]["class_weights"] = [1.0, 2.0, 3.0]
        cfg_path.write_text(yaml.safe_dump(base), encoding="utf-8")
        with pytest.raises(ValidationError, match="class_weights"):
            load_config(cfg_path)


def test_resolve_device_auto():
    device = resolve_device("auto")
    assert device in ("cuda", "cpu")


def test_resolve_device_explicit():
    assert resolve_device("cpu") == "cpu"
