"""Preprocessing for the price and Superhost models (decisions D15, D16).

Two layers:
1. build_features(df): stateless, row-wise rules (no parameters learned from data), so it can be
   applied before cross-validation without leakage.
2. make_preprocessor(family, spec): a ColumnTransformer whose learned steps (imputation values,
   clipping quantiles, scaling, one-hot vocabularies, amenity vocabulary) are fitted on the
   training fold only when it is used inside a Pipeline.
"""
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (activates IterativeImputer)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.data_utils import parse_amenities

SNAPSHOT = pd.Timestamp("2021-02-05")           # last NYC review date (S4)
MIN_NIGHTS_CAP = 365                            # longer minimum stays are treated as 365
MAX_NIGHTS_CAP = 1125                           # Airbnb default; larger values are sentinels
RESPONSE_TIME_ORDER = {"within an hour": 0, "within a few hours": 1, "within a day": 2, "a few days or more": 3}
OLS_REFERENCE = {"room_type": "Entire place", "district": "Manhattan"}


# ---------------------------------------------------------------- 1. stateless feature building
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-ready columns using fixed domain rules only (no statistics of the data).

    `host_ny_listings` counts the host's listings in `df`, including the current one. Because the
    split is grouped by host, all listings of a host are always on the same side, so the count is
    the same whether df is the full data, the training part or the test part.
    """
    out = df.copy()
    out["accommodates"] = out["accommodates"].astype(float).mask(out["accommodates"] == 0)  # 0 guests is invalid
    out["minimum_nights_capped"] = out["minimum_nights"].clip(upper=MIN_NIGHTS_CAP).astype(float)
    out["min_nights_ge_30"] = (out["minimum_nights"] >= 30).astype(float)
    out["maximum_nights_capped"] = out["maximum_nights"].clip(upper=MAX_NIGHTS_CAP).astype(float)
    out["max_nights_is_default"] = (out["maximum_nights"] == MAX_NIGHTS_CAP).astype(float)
    out["max_nights_over_1125"] = (out["maximum_nights"] > MAX_NIGHTS_CAP).astype(float)
    out["host_tenure_days"] = (SNAPSHOT - pd.to_datetime(out["host_since"])).dt.days.astype(float)
    out["host_in_ny"] = out["host_location"].astype("string").str.contains("New York", na=False).astype(float)
    out["amenity_list"] = out["amenities"].apply(parse_amenities)
    out["n_amenities"] = out["amenity_list"].apply(len).astype(float)
    out["host_ny_listings"] = out.groupby("host_id")["listing_id"].transform("size").astype(float)
    out["host_response_time_ord"] = out["host_response_time"].map(RESPONSE_TIME_ORDER).astype(float)
    out["host_activity_missing"] = out["host_response_rate"].isna().astype(float)
    out["acceptance_missing"] = out["host_acceptance_rate"].isna().astype(float)
    out["bedrooms_missing"] = out["bedrooms"].isna().astype(float)
    out["has_reviews"] = (out["n_reviews"].fillna(0) > 0).astype(float)        # ablation only (D7)
    for c in ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]:
        out[c] = out[c].astype("Float64").astype(float)                        # booleans -> 0/1, NaN kept
    for c in ["room_type", "district", "neighbourhood", "property_type"]:
        out[c] = out[c].astype(str)
    return out


HOST_ACTIVITY = ["host_response_rate", "host_acceptance_rate", "host_response_time_ord",
                 "host_activity_missing", "acceptance_missing"]


def feature_spec(task: str = "price", variant: str = "primary") -> dict:
    """Column groups for a task. task: 'price' or 'superhost'; variant: 'primary' or 'ablation'."""
    spec = {
        "skewed": ["accommodates", "minimum_nights_capped", "maximum_nights_capped", "host_ny_listings"],
        "bedrooms": ["bedrooms"],
        "numeric": ["n_amenities", "host_tenure_days", "host_response_rate", "host_acceptance_rate",
                    "host_response_time_ord", "latitude", "longitude"],
        "binary": ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable",
                   "host_in_ny", "min_nights_ge_30", "max_nights_is_default", "max_nights_over_1125",
                   "host_activity_missing", "acceptance_missing", "bedrooms_missing"],
        "cat_low": ["room_type", "district"],
        "cat_high": ["neighbourhood", "property_type"],
        "amenities": ["amenity_list"],
    }
    if task == "superhost":
        spec["binary"].remove("host_is_superhost")                                # the target itself
        if variant == "primary":                                                  # REVIEW #10: activity block is close to the Superhost criteria
            spec["numeric"] = [c for c in spec["numeric"] if c not in HOST_ACTIVITY]
            spec["binary"] = [c for c in spec["binary"] if c not in HOST_ACTIVITY]
    elif task != "price":
        raise ValueError("task must be 'price' or 'superhost'")
    return spec


def spec_columns(spec: dict) -> list:
    """All input columns used by a spec, in order."""
    return [c for group in spec.values() for c in group]


# ---------------------------------------------------------------- 2. fold-fitted transformers
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to quantiles learned in fit (winsorizing); NaN values are left as NaN."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if not 0 <= self.lower < self.upper <= 1:
            raise ValueError("need 0 <= lower < upper <= 1")
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


class AmenityIndicators(BaseEstimator, TransformerMixin):
    """0/1 columns for amenities whose share in the *fit* data lies in [min_share, max_share]."""

    def __init__(self, min_share: float = 0.05, max_share: float = 0.95):
        self.min_share = min_share
        self.max_share = max_share

    @staticmethod
    def _lists(X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            X = X[:, 0]
        return [v if isinstance(v, (list, tuple)) else [] for v in X]

    def fit(self, X, y=None):
        lists = self._lists(X)
        counts = Counter(a for items in lists for a in set(items))
        n = max(len(lists), 1)
        self.vocabulary_ = sorted(a for a, c in counts.items() if self.min_share <= c / n <= self.max_share)
        return self

    def transform(self, X):
        check_is_fitted(self, "vocabulary_")
        index = {a: j for j, a in enumerate(self.vocabulary_)}
        lists = self._lists(X)
        out = np.zeros((len(lists), len(index)))
        for i, items in enumerate(lists):
            for a in items:
                j = index.get(a)
                if j is not None:
                    out[i, j] = 1.0                                               # unseen amenities are ignored
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"amen_{a}" for a in self.vocabulary_], dtype=object)


def make_preprocessor(family: str, spec: dict, bedrooms_fill: str = "median",
                      clip: tuple = (0.01, 0.99), min_frequency: int = 30, imputer: str = "median",
                      random_state: int = 42) -> ColumnTransformer:
    """ColumnTransformer for a model family.

    family: 'ols' (plain linear regression: reference levels dropped; neighbourhood and property_type
            left out because they are nested in district / room_type and would make the design
            matrix rank-deficient), 'scaled' (regularized linear, KNN, SVM, ANN, PCA, clustering) or
            'tree' (no clipping, no log, no scaling).
    bedrooms_fill: 'median' or 'zero' (studio hypothesis; compared in CV).
    imputer: 'median' (default, column-wise), 'knn' (KNNImputer, k=5) or 'iterative' (IterativeImputer
            with ExtraTrees). The multivariate options (ols/scaled only) treat all numeric columns as one
            block: clip -> log1p (skewed counts, bedrooms) -> standardize -> impute -> standardize again,
            so KNN distances are computed on comparable scales (REVIEW #11). Missing flags come from
            build_features in every variant. Compared by CV in notebook 08 Part C.
    """
    if family not in ("ols", "scaled", "tree"):
        raise ValueError("family must be 'ols', 'scaled' or 'tree'")
    if bedrooms_fill not in ("median", "zero"):
        raise ValueError("bedrooms_fill must be 'median' or 'zero'")
    if imputer not in ("median", "knn", "iterative"):
        raise ValueError("imputer must be 'median', 'knn' or 'iterative'")
    if imputer != "median" and family == "tree":
        raise ValueError("the tree family uses median imputation (or native NaN handling in S10)")
    bed_imp = SimpleImputer(strategy="median") if bedrooms_fill == "median" else SimpleImputer(strategy="constant", fill_value=0)
    log1p = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
    cat_high = list(spec["cat_high"])

    if family == "tree":
        skewed = Pipeline([("impute", SimpleImputer(strategy="median"))])
        bedrooms = Pipeline([("impute", bed_imp)])
        numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    else:
        skewed = Pipeline([("impute", SimpleImputer(strategy="median")), ("clip", QuantileClipper(*clip)),
                           ("log1p", log1p), ("scale", StandardScaler())])
        bedrooms = Pipeline([("impute", bed_imp), ("clip", QuantileClipper(*clip)),
                             ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                             ("scale", StandardScaler())])
        numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("clip", QuantileClipper(*clip)),
                            ("scale", StandardScaler())])
    binary = SimpleImputer(strategy="most_frequent")

    if family == "ols":
        cat_high = []                                                            # nested categoricals -> exact collinearity
        low = OneHotEncoder(drop=[OLS_REFERENCE[c] for c in spec["cat_low"]], handle_unknown="ignore", sparse_output=False)
        high = None
    else:
        low = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        high = OneHotEncoder(min_frequency=min_frequency, handle_unknown="infrequent_if_exist", sparse_output=False)

    if imputer == "median":
        parts = [("skewed", skewed, spec["skewed"]), ("bedrooms", bedrooms, spec["bedrooms"]),
                 ("numeric", numeric, spec["numeric"])]
    else:
        bed_steps = [("fill0", SimpleImputer(strategy="constant", fill_value=0))] if bedrooms_fill == "zero" else []
        prep = ColumnTransformer([
            ("skewed", Pipeline([("clip", QuantileClipper(*clip)), ("log1p", log1p)]), spec["skewed"]),
            ("bedrooms", Pipeline(bed_steps + [("clip", QuantileClipper(*clip)),
                                               ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one"))]),
             spec["bedrooms"]),
            ("numeric", QuantileClipper(*clip), spec["numeric"])], remainder="drop", sparse_threshold=0.0)
        if imputer == "knn":
            imp = KNNImputer(n_neighbors=5)
        else:
            imp = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=30, min_samples_leaf=5, n_jobs=-1,
                                                                 random_state=random_state),
                                   max_iter=5, random_state=random_state)
        block = Pipeline([("prep", prep), ("scale", StandardScaler()), ("impute", imp), ("rescale", StandardScaler())])
        parts = [("num_block", block, list(spec["skewed"]) + list(spec["bedrooms"]) + list(spec["numeric"]))]
    parts += [("binary", binary, spec["binary"]), ("cat_low", low, spec["cat_low"])]
    if cat_high:
        parts.append(("cat_high", high, cat_high))
    if spec.get("amenities"):
        parts.append(("amenities", AmenityIndicators(), spec["amenities"][0]))
    return ColumnTransformer(parts, remainder="drop", sparse_threshold=0.0)
