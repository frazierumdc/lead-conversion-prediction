"""Metrics for the converted class, rounded the way the results table reports them."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import export_text


@dataclass(frozen=True)
class Scores:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def rounded(self, digits: int = 2) -> dict[str, float]:
        return {k: round(v, digits) for k, v in asdict(self).items()}


def score(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Scores:
    """Precision, recall and F1 are for class 1 (converted)."""
    return Scores(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )


def confusion(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = ["Not converted", "Converted"]
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred),
        index=[f"Actual: {x}" for x in labels],
        columns=[f"Predicted: {x}" for x in labels],
    )


def feature_importance(model, columns: list[str]) -> pd.Series:
    return pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)


def tree_rules(model, columns: list[str], max_depth: int = 3) -> str:
    return export_text(model, feature_names=columns, max_depth=max_depth)
