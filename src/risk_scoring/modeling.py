from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier


def build_xgboost_model(params: dict[str, Any], y_train: np.ndarray) -> XGBClassifier:
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    imbalance_weight = (negative / positive) if positive else 1.0

    merged_params = dict(params)
    merged_params.setdefault("scale_pos_weight", float(max(imbalance_weight, 1.0)))

    return XGBClassifier(**merged_params)
