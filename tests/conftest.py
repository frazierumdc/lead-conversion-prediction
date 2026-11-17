import pandas as pd
import pytest

from leadscore import config
from leadscore.synthetic import make_leads


@pytest.fixture
def leads() -> pd.DataFrame:
    return make_leads(n=500, seed=3)


@pytest.fixture
def small_grids(monkeypatch):
    """Shrink the search grids so tests stay fast."""
    monkeypatch.setattr(config, "TREE_GRID", {"max_depth": [3, 5], "class_weight": ["balanced"]})
    monkeypatch.setattr(
        config,
        "FOREST_GRID",
        {"n_estimators": [20], "max_depth": [3, 5], "class_weight": ["balanced"]},
    )
