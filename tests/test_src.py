"""Light tests for src/ on small synthetic data (no raw data needed).

Run with:  python -m tests.test_src   (from the project root)
or with pytest if it is installed.
"""
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_utils import find_tf_columns, load_city_listings, missing_report, parse_amenities, to_bool
from src.quality import DataQualityChecker, DataQualityError, RangeRule
from src.split_utils import attach_split, train_test
from src.preprocess import (AmenityIndicators, QuantileClipper, build_features, feature_spec,
                            make_preprocessor, spec_columns)
import numpy as np


def _raises(exc_type, func, *args, **kwargs) -> bool:
    try:
        func(*args, **kwargs)
    except exc_type:
        return True
    return False


def _toy() -> pd.DataFrame:
    return pd.DataFrame({
        "listing_id": [1, 2, 3, 3],
        "price": [100, 0, 250, None],
        "instant_bookable": ["t", "f", None, "t"],
        "room_type": ["Entire place", "Private room", "Private room", "Shared room"],
    })


def test_find_tf_columns():
    assert find_tf_columns(_toy()) == ["instant_bookable"]


def test_to_bool_keeps_missing():
    out = to_bool(_toy())
    assert str(out["instant_bookable"].dtype) == "boolean"
    assert out["instant_bookable"].tolist()[:2] == [True, False]
    assert out["instant_bookable"].isna().sum() == 1


def test_parse_amenities_valid():
    assert parse_amenities('["Wifi", "Heating", " "]') == ["Wifi", "Heating"]


def test_parse_amenities_bad_input():
    assert parse_amenities("not a list") == []
    assert parse_amenities(None) == []
    assert parse_amenities(float("nan")) == []
    assert parse_amenities('{"a": 1}') == []


def test_missing_report():
    rep = missing_report(_toy())
    assert set(rep.index) == {"price", "instant_bookable"}   # room_type and listing_id have no gaps
    assert rep.loc["price", "n_missing"] == 1


def test_checker_finds_duplicate_ids():
    assert DataQualityChecker(_toy()).check_unique_id()["n_violations"] == 1


def test_checker_range_ignores_missing():
    res = DataQualityChecker(_toy()).check_range(RangeRule("price", 1, 200))
    assert res["n_violations"] == 2          # 0 is too low, 250 too high, NaN ignored


def test_checker_unknown_column_raises():
    checker = DataQualityChecker(_toy())
    assert _raises(DataQualityError, checker.check_range, RangeRule("no_such_col", 0, 1))


def test_checker_rejects_non_dataframe():
    assert _raises(TypeError, DataQualityChecker, [1, 2, 3])


def test_assert_clean_strict_mode():
    checker = DataQualityChecker(_toy())
    assert _raises(DataQualityError, checker.assert_clean, [RangeRule("price", 1, None)])


def test_load_missing_file_raises():
    with tempfile.TemporaryDirectory() as tmp:
        assert _raises(FileNotFoundError, load_city_listings, "New York", Path(tmp))


def _write_split(tmp, rows):
    path = Path(tmp) / "split.csv"
    pd.DataFrame(rows, columns=["listing_id", "split"]).to_csv(path, index=False)
    return path


def test_train_test_uses_saved_assignment():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_split(tmp, [(1, "train"), (2, "test"), (3, "train")])
        df = pd.DataFrame({"listing_id": [3, 1, 2], "price": [30, 10, 20]})
        train, test = train_test(df, path)
        assert sorted(train["listing_id"]) == [1, 3] and list(test["listing_id"]) == [2]
        assert "split" not in train.columns


def test_attach_split_rejects_unassigned_listing():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write_split(tmp, [(1, "train")])
        df = pd.DataFrame({"listing_id": [1, 2]})
        assert _raises(ValueError, attach_split, df, path)


def _raw_listings() -> pd.DataFrame:
    """Six fake listings with the raw columns that build_features expects."""
    return pd.DataFrame({
        "listing_id": [1, 2, 3, 4, 5, 6], "host_id": [10, 10, 20, 30, 30, 30],
        "accommodates": [2, 0, 4, 1, 3, 2], "bedrooms": [1.0, None, 2.0, 1.0, None, 1.0],
        "minimum_nights": [1, 30, 400, 2, 3, 30], "maximum_nights": [1125, 30, 2147483647, 60, 1125, 365],
        "host_since": pd.to_datetime(["2015-01-01", "2015-01-01", "2020-01-01", None, "2018-06-01", "2018-06-01"]),
        "host_location": ["New York, New York, United States", None, "Paris, France", "New York", "Brooklyn, New York", "US"],
        "amenities": ['["Wifi", "Kitchen"]', '["Wifi"]', "bad", '["Wifi", "Washer"]', '["Kitchen"]', '["Wifi", "Kitchen", "Gym"]'],
        "host_response_time": ["within an hour", None, "within a day", "a few days or more", None, "within an hour"],
        "host_response_rate": [1.0, None, 0.8, 0.5, None, 1.0], "host_acceptance_rate": [0.9, None, None, 0.4, 0.7, 1.0],
        "n_reviews": [5, 0, None, 2, 0, 1],
        "host_is_superhost": pd.array([True, True, False, None, False, False], dtype="boolean"),
        "host_has_profile_pic": pd.array([True] * 6, dtype="boolean"),
        "host_identity_verified": pd.array([True, True, False, True, False, True], dtype="boolean"),
        "instant_bookable": pd.array([False, True, False, True, False, True], dtype="boolean"),
        "room_type": ["Entire place", "Private room", "Entire place", "Shared room", "Private room", "Entire place"],
        "district": ["Manhattan", "Manhattan", "Brooklyn", "Queens", "Brooklyn", "Manhattan"],
        "neighbourhood": ["Harlem", "Harlem", "Bushwick", "Astoria", "Bushwick", "Chelsea"],
        "property_type": ["Entire apartment", "Private room in apartment", "Entire apartment", "Shared room in apartment",
                          "Private room in apartment", "Entire loft"],
        "latitude": [40.80, 40.81, 40.70, 40.76, 40.69, 40.74], "longitude": [-73.95, -73.94, -73.92, -73.92, -73.91, -74.00],
    })


def test_build_features_domain_rules():
    f = build_features(_raw_listings())
    assert np.isnan(f.loc[1, "accommodates"])                       # 0 guests -> missing
    assert f.loc[2, "minimum_nights_capped"] == 365 and f.loc[2, "maximum_nights_capped"] == 1125
    assert f["max_nights_over_1125"].tolist() == [0, 0, 1, 0, 0, 0]
    assert f["max_nights_is_default"].tolist() == [1, 0, 0, 0, 1, 0]
    assert f["host_ny_listings"].tolist() == [2, 2, 1, 3, 3, 3]
    assert f["host_in_ny"].tolist() == [1, 0, 0, 1, 1, 0]
    assert f.loc[2, "n_amenities"] == 0 and f["bedrooms_missing"].sum() == 2


def test_superhost_primary_spec_excludes_target_and_activity():
    cols = spec_columns(feature_spec("superhost", "primary"))
    assert "host_is_superhost" not in cols and "host_response_rate" not in cols and "host_activity_missing" not in cols
    assert "host_response_rate" in spec_columns(feature_spec("superhost", "ablation"))
    assert "host_is_superhost" in spec_columns(feature_spec("price"))


def test_quantile_clipper_uses_fit_data_only():
    clip = QuantileClipper(0.0, 0.5).fit(np.array([[1.0], [2.0], [3.0]]))
    assert clip.transform(np.array([[10.0], [0.0]])).ravel().tolist() == [2.0, 1.0]


def test_amenity_indicators_vocabulary_from_fit():
    enc = AmenityIndicators(min_share=0.5, max_share=1.0).fit(pd.Series([["Wifi", "Gym"], ["Wifi"], ["Pool"]]))
    assert enc.vocabulary_ == ["Wifi"]
    assert enc.transform(pd.Series([["Sauna"], ["Wifi"]])).tolist() == [[0.0], [1.0]]


def test_make_preprocessor_outputs_complete_matrix():
    f = build_features(_raw_listings())
    for family in ["ols", "scaled", "tree"]:
        pre = make_preprocessor(family, feature_spec("price"), min_frequency=1)
        X = pre.fit_transform(f)
        assert X.shape[0] == 6 and not np.isnan(X).any(), family
        assert len(pre.get_feature_names_out()) == X.shape[1]


def run_all() -> pd.DataFrame:
    """Run every test_* function in this module and return a PASS/FAIL table."""
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    rows = []
    for name, fn in tests:
        try:
            fn()
            rows.append({"test": name, "result": "PASS", "error": ""})
        except Exception as exc:                 # report every failure instead of stopping
            rows.append({"test": name, "result": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = run_all()
    print(table.to_string(index=False))
    sys.exit(0 if (table["result"] == "PASS").all() else 1)
