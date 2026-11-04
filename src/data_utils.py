"""Reusable data-loading and parsing helpers for the NYC Airbnb project."""
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
BOOL_MAP = {"t": True, "f": False}


def load_city_listings(city: str = "New York", raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read Listings.csv (latin-1 encoded) and return the rows of one city.

    Raises FileNotFoundError if the raw file has not been downloaded yet,
    and ValueError if the city does not exist in the file.
    """
    path = Path(raw_dir) / "Listings.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle dataset and extract it into data/raw/ (see README)."
        )
    listings = pd.read_csv(path, encoding="latin-1", low_memory=False)
    city_df = listings.loc[listings["city"] == city].reset_index(drop=True)
    if city_df.empty:
        known = ", ".join(sorted(listings["city"].dropna().unique()))
        raise ValueError(f"No listings found for city={city!r}. Known cities: {known}")
    return city_df


def find_tf_columns(df: pd.DataFrame) -> list:
    """Return the text columns whose non-missing values are only 't' and 'f'."""
    text_cols = df.select_dtypes(exclude="number").columns
    return [c for c in text_cols
            if df[c].notna().any() and set(df[c].dropna().unique()) <= set(BOOL_MAP)]


def to_bool(df: pd.DataFrame, cols: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Return a copy where 't'/'f' columns become pandas 'boolean' (missing stays <NA>)."""
    out = df.copy()
    cols = list(cols) if cols is not None else find_tf_columns(out)
    for col in cols:
        out[col] = out[col].map(BOOL_MAP).astype("boolean")
    return out


def parse_amenities(value) -> list:
    """Turn the amenities string '["Wifi", "Heating"]' into a Python list.

    Missing or malformed values return an empty list instead of crashing.
    """
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values for the columns that have any, largest first."""
    report = pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
    })
    return report.loc[report["n_missing"] > 0].sort_values("n_missing", ascending=False)
