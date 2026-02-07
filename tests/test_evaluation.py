from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_scoring.config import CostConfig
from risk_scoring.evaluation import (
    assign_decision,
    bootstrap_ci,
    calibration_report,
    select_threshold,
    threshold_metrics,
    top_k_capture,
)


# ---------------------------------------------------------------------------
# threshold_metrics — original tests + new fields
# ---------------------------------------------------------------------------


def test_threshold_metrics_computes_confusion_and_costs() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1])
    costs = CostConfig(chargeback_cost=150.0, false_positive_cost=25.0)

    metrics = threshold_metrics(y_true=y_true, y_score=y_score, threshold=0.5, costs=costs)

    assert metrics["tp"] == 1.0
    assert metrics["fp"] == 1.0
    assert metrics["tn"] == 1.0
    assert metrics["fn"] == 1.0
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["baseline_cost"] == pytest.approx(300.0)
    assert metrics["model_cost"] == pytest.approx(175.0)
    assert metrics["net_savings"] == pytest.approx(125.0)
    assert metrics["net_savings_per_txn"] == pytest.approx(31.25)

    # New fields
    assert metrics["net_savings_pct_of_baseline"] == pytest.approx(125.0 / 300.0)
    assert metrics["decline_rate"] == pytest.approx(0.5)  # (1 TP + 1 FP) / 4


def test_threshold_metrics_net_savings_pct_zero_baseline() -> None:
    """When there's no fraud, baseline cost is 0 — pct should be 0.0."""
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.9, 0.2, 0.8, 0.1])
    costs = CostConfig(chargeback_cost=150.0, false_positive_cost=25.0)

    metrics = threshold_metrics(y_true=y_true, y_score=y_score, threshold=0.5, costs=costs)
    assert metrics["net_savings_pct_of_baseline"] == pytest.approx(0.0)


def test_select_threshold_chooses_best_feasible_by_net_savings() -> None:
    curve = pd.DataFrame(
        [
            {
                "threshold": 0.90,
                "false_positive_rate": 0.010,
                "net_savings": 100.0,
                "recall": 0.20,
                "precision": 0.95,
            },
            {
                "threshold": 0.80,
                "false_positive_rate": 0.019,
                "net_savings": 150.0,
                "recall": 0.25,
                "precision": 0.90,
            },
            {
                "threshold": 0.70,
                "false_positive_rate": 0.030,
                "net_savings": 250.0,
                "recall": 0.40,
                "precision": 0.85,
            },
        ]
    )

    selected = select_threshold(curve, max_false_positive_rate=0.02)

    assert selected.constraint_met is True
    assert selected.threshold == pytest.approx(0.80)


def test_select_threshold_uses_recall_and_precision_tiebreakers() -> None:
    curve = pd.DataFrame(
        [
            {
                "threshold": 0.85,
                "false_positive_rate": 0.010,
                "net_savings": 120.0,
                "recall": 0.30,
                "precision": 0.98,
            },
            {
                "threshold": 0.80,
                "false_positive_rate": 0.012,
                "net_savings": 120.0,
                "recall": 0.32,
                "precision": 0.90,
            },
            {
                "threshold": 0.75,
                "false_positive_rate": 0.013,
                "net_savings": 120.0,
                "recall": 0.32,
                "precision": 0.92,
            },
        ]
    )

    selected = select_threshold(curve, max_false_positive_rate=0.02)

    assert selected.constraint_met is True
    assert selected.threshold == pytest.approx(0.75)


def test_select_threshold_fallback_when_no_threshold_meets_fpr_constraint() -> None:
    curve = pd.DataFrame(
        [
            {
                "threshold": 0.90,
                "false_positive_rate": 0.010,
                "net_savings": 80.0,
                "recall": 0.20,
                "precision": 0.95,
            },
            {
                "threshold": 0.85,
                "false_positive_rate": 0.010,
                "net_savings": 120.0,
                "recall": 0.18,
                "precision": 0.97,
            },
            {
                "threshold": 0.80,
                "false_positive_rate": 0.015,
                "net_savings": 200.0,
                "recall": 0.40,
                "precision": 0.90,
            },
        ]
    )

    selected = select_threshold(curve, max_false_positive_rate=0.005)

    assert selected.constraint_met is False
    assert selected.threshold == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# top_k_capture
# ---------------------------------------------------------------------------


def test_top_k_capture_known_answer() -> None:
    # 100 samples, 10 fraud. Top 1% = 1 sample.
    y_true = np.zeros(100, dtype=int)
    y_score = np.linspace(0, 1, 100)

    # Place all fraud in the highest-scoring positions
    y_true[-10:] = 1

    # Top 1% = 1 sample (the highest score), which is fraud
    capture = top_k_capture(y_true, y_score, k_pct=1.0)
    assert capture == pytest.approx(1 / 10)  # 1 of 10 fraud caught


def test_top_k_capture_no_fraud() -> None:
    y_true = np.zeros(50, dtype=int)
    y_score = np.random.default_rng(0).random(50)
    assert top_k_capture(y_true, y_score) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_bounds() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.05, size=1000)
    y_score = rng.random(1000)
    costs = CostConfig(chargeback_cost=150.0, false_positive_cost=25.0)

    ci = bootstrap_ci(y_true, y_score, costs, threshold=0.5, n_bootstrap=200, seed=42)
    assert "lower" in ci
    assert "upper" in ci
    assert "point" in ci
    assert ci["lower"] <= ci["point"] <= ci["upper"]


# ---------------------------------------------------------------------------
# calibration_report
# ---------------------------------------------------------------------------


def test_calibration_report_structure() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.binomial(1, 0.1, size=500)
    y_score = np.clip(y_true * 0.8 + rng.random(500) * 0.3, 0, 1)

    report = calibration_report(y_true, y_score)
    assert "brier_score" in report
    assert "bins" in report
    assert isinstance(report["bins"], list)
    assert 0.0 <= report["brier_score"] <= 1.0

    for b in report["bins"]:
        assert "mean_predicted" in b
        assert "fraction_positive" in b
        assert "count" in b
        assert b["count"] > 0


# ---------------------------------------------------------------------------
# assign_decision (3-way)
# ---------------------------------------------------------------------------


def test_assign_decision_three_bands() -> None:
    scores = np.array([0.95, 0.70, 0.30, 0.10])
    decisions = assign_decision(scores, decline_threshold=0.8, review_threshold=0.5)

    assert list(decisions) == ["DECLINE", "REVIEW", "APPROVE", "APPROVE"]


def test_assign_decision_boundary_values() -> None:
    scores = np.array([0.8, 0.5, 0.49999])
    decisions = assign_decision(scores, decline_threshold=0.8, review_threshold=0.5)

    assert list(decisions) == ["DECLINE", "REVIEW", "APPROVE"]
