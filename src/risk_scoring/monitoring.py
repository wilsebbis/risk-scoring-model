from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CostConfig
from .evaluation import threshold_metrics


PSI_TRIAGE_POLICY = {
    "ok": "< 0.10",
    "investigate": "0.10 – 0.25",
    "retrain": "> 0.25",
}

LABEL_DELAY_NOTE = (
    "Fraud labels typically arrive 30–90 days post-transaction. "
    "Recall metrics in this window are provisional."
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor score drift and live performance")
    parser.add_argument("--baseline", required=True, help="CSV with historical scores")
    parser.add_argument("--current", required=True, help="CSV with current scores")
    parser.add_argument(
        "--score-col", default="score", help="Column name containing model score"
    )
    parser.add_argument(
        "--label-col",
        default="is_fraud",
        help="Optional label column; if missing, only drift metrics are computed",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for live metric calculation",
    )
    parser.add_argument(
        "--output",
        default="artifacts/monitoring_report.json",
        help="Path to JSON report",
    )
    return parser.parse_args()


def _population_stability_index(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    bins: int = 10,
) -> float:
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(baseline_scores, quantiles)
    edges = np.unique(edges)

    if len(edges) < 2:
        return 0.0

    expected_hist, _ = np.histogram(baseline_scores, bins=edges)
    actual_hist, _ = np.histogram(current_scores, bins=edges)

    expected_pct = np.clip(expected_hist / max(expected_hist.sum(), 1), 1e-8, None)
    actual_pct = np.clip(actual_hist / max(actual_hist.sum(), 1), 1e-8, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def _psi_triage_action(psi: float) -> str:
    """Determine triage action based on PSI value."""
    if psi < 0.10:
        return "OK"
    elif psi <= 0.25:
        return "investigate"
    else:
        return "retrain"


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def main() -> None:
    args = _parse_args()
    baseline_df = pd.read_csv(args.baseline)
    current_df = pd.read_csv(args.current)

    if args.score_col not in baseline_df.columns:
        raise KeyError(f"Missing score column in baseline: {args.score_col}")
    if args.score_col not in current_df.columns:
        raise KeyError(f"Missing score column in current: {args.score_col}")

    baseline_scores = baseline_df[args.score_col].astype(float).to_numpy()
    current_scores = current_df[args.score_col].astype(float).to_numpy()

    psi = _population_stability_index(baseline_scores, current_scores)
    action = _psi_triage_action(psi)

    report: dict[str, object] = {
        "score_summary": {
            "baseline": _summary(baseline_scores),
            "current": _summary(current_scores),
        },
        "drift": {
            "psi": psi,
        },
        "triage": {
            "psi": psi,
            "action": action,
            "policy": PSI_TRIAGE_POLICY,
            "label_delay_note": LABEL_DELAY_NOTE,
        },
        "threshold": float(args.threshold),
    }

    if args.label_col in current_df.columns:
        y_true = current_df[args.label_col].astype(int).to_numpy()
        perf = threshold_metrics(
            y_true=y_true,
            y_score=current_scores,
            threshold=float(args.threshold),
            costs=CostConfig(),
        )
        report["performance"] = perf

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(f"Monitoring report written to {output_path}")
    print(f"PSI: {psi:.4f}  →  Action: {action}")
    if action != "OK":
        print(f"  ⚠ {PSI_TRIAGE_POLICY[action]}")
    print(f"  Note: {LABEL_DELAY_NOTE}")


if __name__ == "__main__":
    main()
