"""Generate fake leads with the real schema, for tests and `make demo`.

The relationships are planted loosely (website first touch, time on site and referrals
raise conversion) so the models have something to learn. The numbers mean nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadscore import config
from leadscore.data import REQUIRED_COLUMNS


def make_leads(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    website_first = rng.random(n) < 0.55
    time_on_site = rng.gamma(1.5, 480, n).clip(0, 2500).round()
    profile = rng.choice(["High", "Medium", "Low"], n, p=[0.49, 0.49, 0.02])
    referral = rng.random(n) < 0.03

    logit = (
        -2.2
        + 1.4 * website_first
        + 0.0018 * time_on_site
        + 0.9 * (profile == "High")
        + 1.5 * referral
    )
    status = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    yes_no = lambda p: np.where(rng.random(n) < p, "Yes", "No")  # noqa: E731
    df = pd.DataFrame(
        {
            config.ID_COLUMN: [f"LEAD{i:05d}" for i in range(n)],
            "age": rng.integers(18, 64, n),
            "current_occupation": rng.choice(
                ["Professional", "Unemployed", "Student"], n, p=[0.57, 0.31, 0.12]
            ),
            "first_interaction": np.where(website_first, "Website", "Mobile App"),
            "profile_completed": profile,
            "website_visits": rng.poisson(3.5, n),
            "time_spent_on_website": time_on_site.astype(int),
            "page_views_per_visit": rng.gamma(2.3, 1.3, n).round(3),
            "last_activity": rng.choice(
                ["Email Activity", "Phone Activity", "Website Activity"], n, p=[0.5, 0.27, 0.23]
            ),
            "print_media_type1": yes_no(0.11),
            "print_media_type2": yes_no(0.05),
            "digital_media": yes_no(0.11),
            "educational_channels": yes_no(0.15),
            "referral": np.where(referral, "Yes", "No"),
            config.TARGET: status,
        }
    )
    return df[REQUIRED_COLUMNS]
