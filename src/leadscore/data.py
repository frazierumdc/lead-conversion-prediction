"""Loading and validating the leads file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from leadscore import config

REQUIRED_COLUMNS: list[str] = [
    config.ID_COLUMN,
    "age",
    "current_occupation",
    "first_interaction",
    "profile_completed",
    "website_visits",
    "time_spent_on_website",
    "page_views_per_visit",
    "last_activity",
    *config.CHANNEL_COLUMNS,
    config.TARGET,
]

ALLOWED_VALUES: dict[str, set[str]] = {
    "current_occupation": {"Professional", "Unemployed", "Student"},
    "first_interaction": {"Website", "Mobile App"},
    "profile_completed": {"Low", "Medium", "High"},
    "last_activity": {"Email Activity", "Phone Activity", "Website Activity"},
    **{c: {"Yes", "No"} for c in config.CHANNEL_COLUMNS},
}


class SchemaError(ValueError):
    """Raised when the leads file does not match the expected schema."""


def validate(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing columns: {missing}")
    for col, allowed in ALLOWED_VALUES.items():
        unexpected = set(df[col].unique()) - allowed
        if unexpected:
            raise SchemaError(f"Unexpected values in {col}: {sorted(unexpected)}")
    if not set(df[config.TARGET].unique()) <= {0, 1}:
        raise SchemaError("Target must be 0/1")


def load_leads(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return df
