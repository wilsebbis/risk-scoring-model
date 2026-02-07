from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from .config import ProjectConfig, load_config
from .data import load_raw_dataset, prepare_features
from .evaluation import ranking_metrics, select_threshold, sweep_thresholds, threshold_metrics
from .modeling import build_xgboost_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud/risk scoring model")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_training(cfg: ProjectConfig) -> dict:
    raw_df = load_raw_dataset(cfg.data)
    prepared = prepare_features(raw_df, cfg.data.target_col, cfg.data.id_col)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        prepared.X,
        prepared.y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state,
        stratify=prepared.y,
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval,
        y_trainval,
        test_size=cfg.training.validation_size,
        random_state=cfg.training.random_state,
        stratify=y_trainval,
    )

    model = build_xgboost_model(cfg.model.xgboost_params, y_train.to_numpy())
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    y_score_test = model.predict_proba(X_test)[:, 1]
    threshold_curve = sweep_thresholds(
        y_true=y_test.to_numpy(),
        y_score=y_score_test,
        costs=cfg.costs,
        candidate_steps=cfg.threshold.candidate_steps,
    )

    selected = select_threshold(
        threshold_curve=threshold_curve,
        max_false_positive_rate=cfg.threshold.max_false_positive_rate,
    )

    baseline_50 = threshold_metrics(
        y_true=y_test.to_numpy(),
        y_score=y_score_test,
        threshold=0.5,
        costs=cfg.costs,
    )

    ranking = ranking_metrics(y_test.to_numpy(), y_score_test)

    output_dir = Path(cfg.artifacts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / cfg.artifacts.model_file
    metrics_path = output_dir / cfg.artifacts.metrics_file
    threshold_path = output_dir / cfg.artifacts.threshold_file
    threshold_curve_path = output_dir / cfg.artifacts.threshold_curve_file

    model_bundle = {
        "model": model,
        "feature_columns": prepared.feature_columns,
        "target_col": cfg.data.target_col,
        "id_col": cfg.data.id_col,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_threshold": selected.threshold,
    }
    joblib.dump(model_bundle, model_path)

    threshold_curve.to_csv(threshold_curve_path, index=False)

    train_fraud_rate = float(np.mean(y_trainval))
    test_fraud_rate = float(np.mean(y_test))

    threshold_payload = {
        "threshold": selected.threshold,
        "constraint_met": selected.constraint_met,
        "max_false_positive_rate": cfg.threshold.max_false_positive_rate,
        "selected_metrics": selected.metrics,
    }
    _save_json(threshold_path, threshold_payload)

    metrics_payload = {
        "dataset": {
            "rows": int(len(prepared.y)),
            "feature_count": int(len(prepared.feature_columns)),
            "train_rows": int(len(y_trainval)),
            "test_rows": int(len(y_test)),
            "train_fraud_rate": train_fraud_rate,
            "test_fraud_rate": test_fraud_rate,
        },
        "ranking_metrics": ranking,
        "selected_threshold": threshold_payload,
        "metrics_at_0_5": baseline_50,
        "artifacts": {
            "model_path": str(model_path),
            "threshold_curve_path": str(threshold_curve_path),
            "threshold_path": str(threshold_path),
        },
    }
    _save_json(metrics_path, metrics_payload)

    return metrics_payload


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    metrics = run_training(cfg)

    selected = metrics["selected_threshold"]
    m = selected["selected_metrics"]
    print("Training complete")
    print(f"AUPRC: {metrics['ranking_metrics']['auprc']:.4f}")
    print(
        "Selected threshold "
        f"{selected['threshold']:.4f} | FPR {m['false_positive_rate']:.4f} "
        f"| Recall {m['recall']:.4f} | Precision {m['precision']:.4f}"
    )
    if not selected["constraint_met"]:
        print("Warning: No threshold satisfied configured max_false_positive_rate")


if __name__ == "__main__":
    main()
