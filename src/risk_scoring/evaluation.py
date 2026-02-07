from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .config import CostConfig


@dataclass
class ThresholdSelection:
    threshold: float
    constraint_met: bool
    metrics: dict[str, float]


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0


def threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    costs: CostConfig,
) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tp, fp, tn, fn = _confusion_counts(y_true, y_pred)
    n = len(y_true)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    false_positive_rate = _safe_div(fp, fp + tn)

    baseline_cost = float((y_true == 1).sum() * costs.chargeback_cost)
    model_cost = (fn * costs.chargeback_cost) + (fp * costs.false_positive_cost)
    net_savings = baseline_cost - model_cost

    return {
        "threshold": float(threshold),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "baseline_cost": baseline_cost,
        "model_cost": float(model_cost),
        "net_savings": float(net_savings),
        "net_savings_per_txn": float(net_savings / n) if n else 0.0,
        "net_savings_pct_of_baseline": _safe_div(net_savings, baseline_cost),
        "decline_rate": _safe_div(tp + fp, n),
    }


def sweep_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    costs: CostConfig,
    candidate_steps: int,
) -> pd.DataFrame:
    if candidate_steps < 10:
        raise ValueError("candidate_steps must be at least 10")

    thresholds = np.linspace(0.0, 1.0, candidate_steps + 1)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        rows.append(threshold_metrics(y_true, y_score, float(threshold), costs))

    df = pd.DataFrame(rows)
    return df.sort_values("threshold").reset_index(drop=True)


def select_threshold(
    threshold_curve: pd.DataFrame,
    max_false_positive_rate: float,
) -> ThresholdSelection:
    feasible = threshold_curve[
        threshold_curve["false_positive_rate"] <= max_false_positive_rate
    ].copy()

    if feasible.empty:
        fallback = (
            threshold_curve.sort_values(
                by=["false_positive_rate", "net_savings", "recall"],
                ascending=[True, False, False],
            )
            .head(1)
            .iloc[0]
        )
        return ThresholdSelection(
            threshold=float(fallback["threshold"]),
            constraint_met=False,
            metrics={k: float(v) for k, v in fallback.items()},
        )

    best = (
        feasible.sort_values(
            by=["net_savings", "recall", "precision"],
            ascending=[False, False, False],
        )
        .head(1)
        .iloc[0]
    )
    return ThresholdSelection(
        threshold=float(best["threshold"]),
        constraint_met=True,
        metrics={k: float(v) for k, v in best.items()},
    )


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"auprc": 0.0, "roc_auc": 0.0}

    return {
        "auprc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


# ---------------------------------------------------------------------------
# New proof-grade metrics
# ---------------------------------------------------------------------------


def top_k_capture(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_pct: float = 1.0,
) -> float:
    """Fraction of total fraud captured in the top-k% highest-risk scores."""
    n = len(y_true)
    if n == 0:
        return 0.0

    k_count = max(int(np.ceil(n * k_pct / 100.0)), 1)
    top_indices = np.argsort(y_score)[::-1][:k_count]
    fraud_in_top_k = int(y_true[top_indices].sum())
    total_fraud = int(y_true.sum())
    return _safe_div(fraud_in_top_k, total_fraud)


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    costs: CostConfig,
    threshold: float,
    n_bootstrap: int = 1_000,
    ci: float = 95.0,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap 95% confidence interval for net_savings_pct_of_baseline."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    estimates: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = threshold_metrics(y_true[idx], y_score[idx], threshold, costs)
        estimates.append(m["net_savings_pct_of_baseline"])

    alpha = (100.0 - ci) / 2.0
    lower = float(np.percentile(estimates, alpha))
    upper = float(np.percentile(estimates, 100.0 - alpha))
    point = float(np.mean(estimates))

    return {"lower": lower, "upper": upper, "point": point}


def calibration_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Reliability-diagram data + Brier score for score calibration assessment."""
    brier = float(brier_score_loss(y_true, y_score))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_score >= lo) & (y_score < hi) if i < n_bins - 1 else (y_score >= lo) & (y_score <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_predicted = float(y_score[mask].mean())
        fraction_positive = float(y_true[mask].mean())
        bins.append({
            "bin_lower": float(lo),
            "bin_upper": float(hi),
            "count": count,
            "mean_predicted": mean_predicted,
            "fraction_positive": fraction_positive,
        })

    return {"brier_score": brier, "bins": bins}


def assign_decision(
    y_score: np.ndarray,
    decline_threshold: float,
    review_threshold: float,
) -> np.ndarray:
    """3-way decision: DECLINE / REVIEW / APPROVE based on two thresholds."""
    decisions = np.full(len(y_score), "APPROVE", dtype=object)
    decisions[y_score >= review_threshold] = "REVIEW"
    decisions[y_score >= decline_threshold] = "DECLINE"
    return decisions
