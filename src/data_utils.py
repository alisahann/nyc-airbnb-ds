"""Reusable data-loading helpers for the NYC Airbnb project."""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_city_listings(city: str = "New York", raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read Listings.csv (latin-1 encoded) and return the rows of one city.

    Raises a clear error if the raw file has not been downloaded yet.
    """
    path = Path(raw_dir) / "Listings.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the Kaggle dataset and extract it into data/raw/ (see README)."
        )
    listings = pd.read_csv(path, encoding="latin-1", low_memory=False)
    city_df = listings.loc[listings["city"] == city].reset_index(drop=True)
    if city_df.empty:
        raise ValueError(f"No listings found for city={city!r}.")
    return city_df
