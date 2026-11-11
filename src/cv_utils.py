"""Cross-validation helpers for S8-S11 (decision D17).

- make_cv_folds / save_folds / load_folds: one persisted, host-grouped, stratified 5-fold assignment
  per task on the TRAIN part only; every model of a task reuses exactly these folds.
- regression_metrics / classification_metrics: the approved metrics (D5, D6).
- evaluate_cv: fits a full Pipeline per fold and returns per-fold metrics + out-of-fold predictions.
- paired_comparison: mean and standard error of fold-wise paired differences (used for the 1-SE rule).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, f1_score, mean_absolute_error, median_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
STRATIFY = {"price": "room_type", "superhost": "host_is_superhost"}      # D17


def task_frame(train: pd.DataFrame, task: str) -> pd.DataFrame:
    """Rows of the training part that belong to a task, in a fixed order (sorted by listing_id)."""
    if task == "price":
        d = train[train["price"] > 0]
    elif task == "superhost":
        d = train[train["host_is_superhost"].notna()]
    else:
        raise ValueError("task must be 'price' or 'superhost'")
    return d.sort_values("listing_id").reset_index(drop=True)


def make_cv_folds(d: pd.DataFrame, task: str, n_splits: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Host-grouped, stratified fold labels for the rows of d (output: listing_id, host_id, fold)."""
    y = d[STRATIFY[task]].astype(str)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold = np.full(len(d), -1)
    for k, (_, va) in enumerate(sgkf.split(d, y=y, groups=d["host_id"])):
        fold[va] = k
    out = pd.DataFrame({"listing_id": d["listing_id"].values, "host_id": d["host_id"].values, "fold": fold})
    check_folds(out)
    return out


def check_folds(assign: pd.DataFrame) -> None:
    """Raise if a row is unassigned, an id repeats, or a host appears in more than one fold."""
    if (assign["fold"] < 0).any():
        raise ValueError("some rows have no fold")
    if not assign["listing_id"].is_unique:
        raise ValueError("duplicate listing ids in fold assignment")
    hosts_per_fold = assign.groupby("host_id")["fold"].nunique()
    if (hosts_per_fold > 1).any():
        raise ValueError(f"{int((hosts_per_fold > 1).sum())} hosts appear in more than one fold")


def save_folds(assign: pd.DataFrame, task: str, meta: dict, split_dir: Path = SPLIT_DIR) -> Path:
    """Write data/splits/cv_folds_<task>.csv and add the task to cv_manifest.json."""
    path = split_dir / f"cv_folds_{task}.csv"
    assign[["listing_id", "fold"]].to_csv(path, index=False)
    man_path = split_dir / "cv_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    manifest[task] = meta
    man_path.write_text(json.dumps(manifest, indent=2))
    return path


def load_folds(d: pd.DataFrame, task: str, split_dir: Path = SPLIT_DIR) -> list:
    """Positional (train_idx, val_idx) pairs for the rows of d, read from the saved assignment."""
    saved = pd.read_csv(split_dir / f"cv_folds_{task}.csv")
    fold = d[["listing_id"]].merge(saved, on="listing_id", how="left", validate="one_to_one")["fold"]
    if fold.isna().any():
        raise ValueError(f"{int(fold.isna().sum())} rows of d have no saved fold")
    fold = fold.to_numpy(dtype=int)
    return [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in sorted(np.unique(fold))]


# ---------------------------------------------------------------- metrics
def regression_metrics(y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    """D5: log-RMSE (primary), MAE/MedAE/RMSE in USD after expm1, R2 on the log scale."""
    y_log, pred_log = np.asarray(y_log, float), np.asarray(pred_log, float)
    y_usd, p_usd = np.expm1(y_log), np.expm1(pred_log)
    return {"log_rmse": float(np.sqrt(mean_squared_error(y_log, pred_log))),
            "mae_usd": float(mean_absolute_error(y_usd, p_usd)),
            "medae_usd": float(median_absolute_error(y_usd, p_usd)),
            "r2_log": float(r2_score(y_log, pred_log)),
            "rmse_usd": float(np.sqrt(mean_squared_error(y_usd, p_usd)))}


def classification_metrics(y: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """D6: ROC-AUC (primary), PR-AUC, and recall/precision/F1 at a fixed threshold."""
    y, proba = np.asarray(y, int), np.asarray(proba, float)
    pred = (proba >= threshold).astype(int)
    return {"roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "prevalence": float(y.mean())}


def evaluate_cv(estimator, X: pd.DataFrame, y: pd.Series, folds: list, task: str, threshold: float = 0.5):
    """Fit a clone of estimator on each training fold; score on the validation fold.

    Returns (per_fold DataFrame, out-of-fold predictions). For 'price' y is log1p(price) and the
    predictions are on the log scale; for 'superhost' they are positive-class probabilities.
    """
    oof = np.full(len(y), np.nan)
    rows = []
    for k, (tr, va) in enumerate(folds):
        t0 = time.perf_counter()
        est = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        if task == "price":
            p = est.predict(X.iloc[va])
            m = regression_metrics(y.iloc[va], p)
        else:
            p = est.predict_proba(X.iloc[va])[:, 1]
            m = classification_metrics(y.iloc[va], p, threshold)
        oof[va] = p
        rows.append({"fold": k, "n_val": len(va), **m, "seconds": round(time.perf_counter() - t0, 2)})
    return pd.DataFrame(rows), oof


def summarize(per_fold: pd.DataFrame, name: str) -> pd.Series:
    """Mean and SD across folds of every metric column, as one row named `name`."""
    cols = [c for c in per_fold.columns if c not in ("fold", "n_val", "seconds")]
    out = {f"{c}_mean": per_fold[c].mean() for c in cols}
    out.update({f"{c}_sd": per_fold[c].std(ddof=1) for c in cols})
    out["seconds_total"] = per_fold["seconds"].sum() if "seconds" in per_fold else np.nan
    return pd.Series(out, name=name)


def paired_comparison(scores_a, scores_b) -> dict:
    """Fold-wise paired difference a - b: mean, SD, SE = SD/sqrt(k) (basis of the 1-SE rule, REVIEW #11)."""
    d = np.asarray(scores_a, float) - np.asarray(scores_b, float)
    sd = d.std(ddof=1)
    return {"mean_diff": float(d.mean()), "sd_diff": float(sd), "se_diff": float(sd / np.sqrt(len(d))), "k": len(d)}
