"""A small, reusable data-quality checker (reports problems, never changes the data)."""
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


class DataQualityError(Exception):
    """Raised when a data-quality rule cannot be evaluated or is violated in strict mode."""


@dataclass(frozen=True)
class RangeRule:
    """Plausible range for one numeric column; None means 'no limit on this side'."""
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    description: str = ""


class DataQualityChecker:
    """Runs simple checks on a DataFrame and collects the results in a table."""

    def __init__(self, df: pd.DataFrame, id_col: str = "listing_id"):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
        if id_col not in df.columns:
            raise DataQualityError(f"Identifier column {id_col!r} not found")
        self.df = df
        self.id_col = id_col
        self.results = []

    def __repr__(self) -> str:
        return f"DataQualityChecker(rows={len(self.df)}, checks_run={len(self.results)})"

    def check_unique_id(self) -> dict:
        n_dup = int(self.df[self.id_col].duplicated().sum())
        result = {"check": f"unique {self.id_col}", "column": self.id_col,
                  "n_violations": n_dup, "pct_violations": round(100 * n_dup / len(self.df), 3)}
        self.results.append(result)
        return result

    def check_range(self, rule: RangeRule) -> dict:
        if rule.column not in self.df.columns:
            raise DataQualityError(f"Column {rule.column!r} not found for rule {rule}")
        values = pd.to_numeric(self.df[rule.column], errors="coerce")
        too_low = values < rule.min_value if rule.min_value is not None else pd.Series(False, index=values.index)
        too_high = values > rule.max_value if rule.max_value is not None else pd.Series(False, index=values.index)
        n_bad = int((too_low | too_high).sum())          # NaN never counts as a violation
        result = {"check": rule.description or f"{rule.column} in [{rule.min_value}, {rule.max_value}]",
                  "column": rule.column, "n_violations": n_bad,
                  "pct_violations": round(100 * n_bad / len(self.df), 3)}
        self.results.append(result)
        return result

    def run(self, rules: Iterable[RangeRule]) -> pd.DataFrame:
        """Run the id check and every range rule; return one row per check."""
        self.results = []
        self.check_unique_id()
        for rule in rules:
            self.check_range(rule)
        return pd.DataFrame(self.results)

    def assert_clean(self, rules: Iterable[RangeRule]) -> None:
        """Strict mode: raise DataQualityError if any check has violations."""
        report = self.run(rules)
        failed = report.loc[report["n_violations"] > 0, "check"].tolist()
        if failed:
            raise DataQualityError(f"{len(failed)} check(s) failed: {failed}")


# Plausibility rules for the New York listings (used for reporting only)
NYC_RULES = (
    RangeRule("price", 1, None, "price >= 1 USD"),
    RangeRule("accommodates", 1, None, "accommodates >= 1 guest"),
    RangeRule("bedrooms", 0, 20, "bedrooms between 0 and 20"),
    RangeRule("minimum_nights", 1, 365, "minimum_nights between 1 and 365"),
    RangeRule("maximum_nights", 1, 1125, "maximum_nights between 1 and 1125 (Airbnb default max)"),
    RangeRule("latitude", 40.49, 40.92, "latitude inside the NYC bounding box"),
    RangeRule("longitude", -74.27, -73.68, "longitude inside the NYC bounding box"),
    RangeRule("host_response_rate", 0, 1, "host_response_rate between 0 and 1"),
    RangeRule("review_scores_rating", 20, 100, "review_scores_rating between 20 and 100"),
)
