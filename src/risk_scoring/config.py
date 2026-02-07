from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "csv"
    csv_path: str = "data/transactions.csv"
    target_col: str = "is_fraud"
    id_col: str = "transaction_id"
    snowflake_query: str = ""


@dataclass
class ModelConfig:
    xgboost_params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "reg_lambda": 2.0,
            "random_state": 42,
            "n_jobs": -1,
        }
    )


@dataclass
class TrainingConfig:
    test_size: float = 0.20
    validation_size: float = 0.25
    random_state: int = 42


@dataclass
class ThresholdConfig:
    max_false_positive_rate: float = 0.02
    candidate_steps: int = 500


@dataclass
class CostConfig:
    chargeback_cost: float = 150.0
    false_positive_cost: float = 25.0


@dataclass
class ArtifactConfig:
    output_dir: str = "artifacts"
    model_file: str = "risk_model.joblib"
    metrics_file: str = "metrics.json"
    threshold_file: str = "threshold.json"
    threshold_curve_file: str = "threshold_curve.csv"


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)


def _ensure_mapping(section: Any, name: str) -> dict[str, Any]:
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return section


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("Root config must be a mapping.")

    data_section = _ensure_mapping(raw.get("data"), "data")
    model_section = _ensure_mapping(raw.get("model"), "model")
    training_section = _ensure_mapping(raw.get("training"), "training")
    threshold_section = _ensure_mapping(raw.get("threshold"), "threshold")
    cost_section = _ensure_mapping(raw.get("costs"), "costs")
    artifact_section = _ensure_mapping(raw.get("artifacts"), "artifacts")

    data_cfg = DataConfig(**data_section)

    default_model_cfg = ModelConfig()
    xgboost_section = _ensure_mapping(model_section.get("xgboost"), "model.xgboost")
    merged_xgb = {**default_model_cfg.xgboost_params, **xgboost_section}
    model_cfg = ModelConfig(xgboost_params=merged_xgb)

    training_cfg = TrainingConfig(**training_section)
    threshold_cfg = ThresholdConfig(**threshold_section)
    costs_cfg = CostConfig(**cost_section)
    artifacts_cfg = ArtifactConfig(**artifact_section)

    return ProjectConfig(
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        threshold=threshold_cfg,
        costs=costs_cfg,
        artifacts=artifacts_cfg,
    )
