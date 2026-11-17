"""Constants that define the analysis. Changing any of these changes the results."""

from __future__ import annotations

from typing import Any

SEED: int = 1
TEST_SIZE: float = 0.3
TARGET: str = "status"
ID_COLUMN: str = "ID"

NUMERIC_COLUMNS: list[str] = [
    "age",
    "website_visits",
    "time_spent_on_website",
    "page_views_per_visit",
]

# Marketing channels: "Yes"/"No" flags for whether the lead saw or heard about the company there.
CHANNEL_COLUMNS: list[str] = [
    "digital_media",
    "educational_channels",
    "print_media_type1",
    "print_media_type2",
    "referral",
]
CHANNEL_LABELS: dict[str, str] = {
    "print_media_type1": "Newspaper",
    "print_media_type2": "Magazine",
    "digital_media": "Digital ads",
    "educational_channels": "Education channels",
    "referral": "Referral",
}

# Order matters: it fixes the one-hot column order, which the random forest's feature
# sampling depends on. This is the order of the original run.
CATEGORICAL_COLUMNS: list[str] = [
    *CHANNEL_COLUMNS,
    "current_occupation",
    "first_interaction",
    "profile_completed",
    "last_activity",
]

TREE_GRID: dict[str, list[Any]] = {
    "max_depth": [None, 3, 5, 7, 10],
    "min_samples_split": [2, 10, 25, 50],
    "min_samples_leaf": [1, 5, 10, 20],
    "class_weight": [None, "balanced"],
}
TREE_CV_FOLDS: int = 5

# Kept small on purpose to bound runtime (48 candidates x 3 folds).
FOREST_GRID: dict[str, list[Any]] = {
    "n_estimators": [100, 200],
    "max_depth": [5, 7, 10],
    "min_samples_split": [10, 20],
    "min_samples_leaf": [5, 10],
    "class_weight": [None, "balanced"],
}
FOREST_CV_FOLDS: int = 3

# Converters are the minority class (~30%), and missing one costs more than an extra call.
SCORING: str = "recall"
