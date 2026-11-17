"""The four models compared: default and tuned decision tree, default and tuned random forest."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from leadscore import config


def default_tree(x: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(class_weight="balanced", random_state=config.SEED).fit(x, y)


def tuned_tree(
    x: pd.DataFrame, y: pd.Series, grid: dict[str, list[Any]] | None = None
) -> tuple[DecisionTreeClassifier, dict[str, Any]]:
    search = GridSearchCV(
        DecisionTreeClassifier(random_state=config.SEED),
        param_grid=grid or config.TREE_GRID,
        cv=config.TREE_CV_FOLDS,
        scoring=config.SCORING,
    ).fit(x, y)
    return search.best_estimator_, search.best_params_


def default_forest(x: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    return RandomForestClassifier(class_weight="balanced", random_state=config.SEED).fit(x, y)


def tuned_forest(
    x: pd.DataFrame, y: pd.Series, grid: dict[str, list[Any]] | None = None, n_jobs: int = -1
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    search = GridSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=config.SEED),
        param_grid=grid or config.FOREST_GRID,
        cv=config.FOREST_CV_FOLDS,
        scoring=config.SCORING,
        n_jobs=n_jobs,
    ).fit(x, y)
    return search.best_estimator_, search.best_params_
