from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risk_scoring.config import CostConfig
from risk_scoring.evaluation import select_threshold, threshold_metrics


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
