"""Conversion rates by segment and by marketing channel."""

from __future__ import annotations

import pandas as pd

from leadscore import config


def conversion_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Leads, conversions and conversion rate for each value of a column."""
    grouped = df.groupby(column)[config.TARGET].agg(leads="count", converted="sum")
    grouped["rate"] = grouped["converted"] / grouped["leads"]
    return grouped.sort_values("rate", ascending=False)


def conversion_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Conversion rate among leads who saw each channel (a lead can see several)."""
    rows = []
    for col in config.CHANNEL_COLUMNS:
        seen = df[df[col] == "Yes"]
        rows.append(
            {
                "channel": config.CHANNEL_LABELS[col],
                "leads": len(seen),
                "converted": int(seen[config.TARGET].sum()),
                "rate": float(seen[config.TARGET].mean()) if len(seen) else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("channel").sort_values("rate", ascending=False)
