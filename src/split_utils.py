"""Helpers to reuse the approved train/test split (decision D13) in every notebook."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "train_test_split.csv"


def attach_split(df: pd.DataFrame, path: Path = SPLIT_PATH, id_col: str = "listing_id") -> pd.DataFrame:
    """Return a copy of df with a 'split' column ('train'/'test') read from the saved split file.

    Raises ValueError if the file has duplicate ids, unknown labels, or if any listing in df is unassigned.
    """
    ids = pd.read_csv(path)
    if not ids[id_col].is_unique:
        raise ValueError("split file contains duplicate listing ids")
    if not set(ids["split"]).issubset({"train", "test"}):
        raise ValueError("split file contains labels other than train/test")
    out = df.drop(columns="split", errors="ignore").merge(ids[[id_col, "split"]], on=id_col,
                                                           how="left", validate="one_to_one")
    n_missing = int(out["split"].isna().sum())
    if n_missing:
        raise ValueError(f"{n_missing} listings have no split assignment")
    return out


def train_test(df: pd.DataFrame, path: Path = SPLIT_PATH) -> tuple:
    """Split df into (train, test) DataFrames using the saved assignment; the 'split' column is dropped."""
    d = attach_split(df, path)
    train = d[d["split"] == "train"].drop(columns="split").reset_index(drop=True)
    test = d[d["split"] == "test"].drop(columns="split").reset_index(drop=True)
    return train, test
