from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import prepare_inference_features
from .evaluation import assign_decision


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch score transactions")
    parser.add_argument("--model", required=True, help="Path to risk_model.joblib")
    parser.add_argument("--input", required=True, help="Input CSV to score")
    parser.add_argument("--output", required=True, help="Output CSV with score column")
    parser.add_argument("--score-col", default="score", help="Score column name")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decline threshold (scores >= threshold are flagged/declined)",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        help="Review threshold for 3-way decisions (APPROVE / REVIEW / DECLINE)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_bundle = joblib.load(args.model)

    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    target_col = model_bundle.get("target_col")
    id_col = model_bundle.get("id_col")

    # Fall back to bundle thresholds if not specified on CLI
    decline_threshold = args.threshold or model_bundle.get("selected_threshold")
    review_threshold = args.review_threshold or model_bundle.get("review_threshold")

    df = pd.read_csv(args.input)
    X = prepare_inference_features(
        df=df,
        feature_columns=feature_columns,
        target_col=target_col,
        id_col=id_col,
    )

    scores = model.predict_proba(X)[:, 1]
    scored = df.copy()
    scored[args.score_col] = scores

    if decline_threshold is not None and review_threshold is not None:
        # 3-way decision mode
        scored["decision"] = assign_decision(
            y_score=scores,
            decline_threshold=decline_threshold,
            review_threshold=review_threshold,
        )
        scored["prediction"] = (scores >= decline_threshold).astype(int)
    elif decline_threshold is not None:
        scored["prediction"] = (scores >= decline_threshold).astype(int)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    print(f"Wrote {len(scored):,} scored rows to {output_path}")
    print(f"Score mean: {scores.mean():.4f}")

    if "decision" in scored.columns:
        counts = scored["decision"].value_counts()
        for action in ["DECLINE", "REVIEW", "APPROVE"]:
            pct = counts.get(action, 0) / len(scored) * 100
            print(f"  {action}: {counts.get(action, 0):,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
