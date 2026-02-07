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
from .evaluation import (
    bootstrap_ci,
    calibration_report,
    ranking_metrics,
    select_threshold,
    sweep_thresholds,
    threshold_metrics,
    top_k_capture,
)
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


def _temporal_split(X, y, test_size: float):
    """Time-ordered split: last test_size fraction used as test set."""
    n = len(y)
    split_idx = int(n * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def run_training(cfg: ProjectConfig) -> dict:
    raw_df = load_raw_dataset(cfg.data)
    prepared = prepare_features(raw_df, cfg.data.target_col, cfg.data.id_col)

    split_mode = cfg.training.split_mode.lower().strip()

    if split_mode == "time":
        # Temporal split: assume data is already sorted by time (row order).
        X_trainval, X_test, y_trainval, y_test = _temporal_split(
            prepared.X, prepared.y, cfg.training.test_size,
        )
    else:
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
    top_k = top_k_capture(y_test.to_numpy(), y_score_test, k_pct=1.0)

    ci = bootstrap_ci(
        y_true=y_test.to_numpy(),
        y_score=y_score_test,
        costs=cfg.costs,
        threshold=selected.threshold,
        n_bootstrap=1_000,
    )

    cal = calibration_report(y_test.to_numpy(), y_score_test)

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
        "review_threshold": cfg.threshold.review_threshold,
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

    # Build 3-way decision band summary if review threshold is set
    decision_bands = None
    if cfg.threshold.review_threshold is not None:
        review_t = cfg.threshold.review_threshold
        decline_t = selected.threshold
        y_test_np = y_test.to_numpy()
        n_total = len(y_test_np)
        n_decline = int((y_score_test >= decline_t).sum())
        n_review = int(((y_score_test >= review_t) & (y_score_test < decline_t)).sum())
        n_approve = int((y_score_test < review_t).sum())
        decision_bands = {
            "decline_threshold": decline_t,
            "review_threshold": review_t,
            "decline_count": n_decline,
            "review_count": n_review,
            "approve_count": n_approve,
            "decline_rate": n_decline / n_total if n_total else 0.0,
            "review_rate": n_review / n_total if n_total else 0.0,
            "approve_rate": n_approve / n_total if n_total else 0.0,
        }

    metrics_payload = {
        "dataset": {
            "rows": int(len(prepared.y)),
            "feature_count": int(len(prepared.feature_columns)),
            "train_rows": int(len(y_trainval)),
            "test_rows": int(len(y_test)),
            "train_fraud_rate": train_fraud_rate,
            "test_fraud_rate": test_fraud_rate,
            "split_mode": split_mode,
        },
        "ranking_metrics": ranking,
        "top_k_capture_1pct": top_k,
        "bootstrap_ci_95": ci,
        "calibration": cal,
        "selected_threshold": threshold_payload,
        "decision_bands": decision_bands,
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
    ci = metrics["bootstrap_ci_95"]
    print("Training complete")
    print(f"  Split mode: {metrics['dataset']['split_mode']}")
    print(f"  AUPRC: {metrics['ranking_metrics']['auprc']:.4f}")
    print(f"  Top-1% capture: {metrics['top_k_capture_1pct']:.1%}")
    print(
        f"  Selected threshold {selected['threshold']:.4f} | "
        f"FPR {m['false_positive_rate']:.4f} | "
        f"Recall {m['recall']:.4f} | "
        f"Precision {m['precision']:.4f}"
    )
    print(
        f"  Net savings: {m['net_savings_pct_of_baseline']:.1%} of baseline "
        f"({ci['lower']:.1%} – {ci['upper']:.1%}, 95% CI)"
    )
    print(f"  Decline rate: {m['decline_rate']:.2%}")
    print(f"  Calibration Brier: {metrics['calibration']['brier_score']:.4f}")

    if metrics.get("decision_bands"):
        db = metrics["decision_bands"]
        print(
            f"  3-way bands: DECLINE {db['decline_rate']:.2%} | "
            f"REVIEW {db['review_rate']:.2%} | "
            f"APPROVE {db['approve_rate']:.2%}"
        )

    if not selected["constraint_met"]:
        print("  Warning: No threshold satisfied configured max_false_positive_rate")


if __name__ == "__main__":
    main()
