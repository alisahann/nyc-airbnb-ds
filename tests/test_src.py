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
