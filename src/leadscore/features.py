"""Encoding and the train/test split."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from leadscore import config


def encode(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop the ID, one-hot encode categoricals (drop first level), split off the target."""
    data = df.drop(columns=[config.ID_COLUMN], errors="ignore")
    data = pd.get_dummies(data, columns=config.CATEGORICAL_COLUMNS, dtype=float, drop_first=True)
    y = data.pop(config.TARGET)
    return data, y


def split(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """70/30 split, stratified so both sides keep the ~30% conversion rate."""
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )
    return x_train, x_test, y_train, y_test
