import pandas as pd
import pytest

from leadscore import config
from leadscore.analysis import conversion_by, conversion_by_channel
from leadscore.data import SchemaError, validate
from leadscore.features import encode, split


def test_schema_accepts_synthetic(leads):
    validate(leads)


def test_schema_rejects_unknown_category(leads):
    bad = leads.copy()
    bad.loc[0, "first_interaction"] = "Carrier pigeon"
    with pytest.raises(SchemaError):
        validate(bad)


def test_encode_drops_id_and_target_and_gives_16_features(leads):
    x, y = encode(leads)
    assert config.ID_COLUMN not in x.columns
    assert config.TARGET not in x.columns
    assert x.shape[1] == 16
    assert list(x.columns[:4]) == config.NUMERIC_COLUMNS
    assert set(y.unique()) <= {0, 1}


def test_split_is_stratified_and_reproducible(leads):
    x, y = encode(leads)
    a = split(x, y)
    b = split(x, y)
    assert len(a[0]) == 350 and len(a[1]) == 150
    assert abs(a[2].mean() - a[3].mean()) < 0.02
    assert a[1].index.equals(b[1].index)


def test_channel_conversion_counts():
    df = pd.DataFrame(
        {
            "referral": ["Yes", "Yes", "No", "No"],
            "digital_media": ["No", "Yes", "Yes", "No"],
            "educational_channels": ["No"] * 4,
            "print_media_type1": ["No"] * 4,
            "print_media_type2": ["No"] * 4,
            "status": [1, 1, 0, 0],
        }
    )
    table = conversion_by_channel(df)
    assert table.loc["Referral", "leads"] == 2
    assert table.loc["Referral", "rate"] == 1.0
    assert table.loc["Digital ads", "rate"] == 0.5
    assert table.index[0] == "Referral"


def test_conversion_by_column():
    df = pd.DataFrame(
        {"first_interaction": ["Website", "Website", "Mobile App"], "status": [1, 0, 0]}
    )
    table = conversion_by(df, "first_interaction")
    assert table.loc["Website", "rate"] == 0.5
