# NYC Airbnb – End-to-End Data Science Project

Design Project course work. An end-to-end data science project on New York City Airbnb listings, following the structure of the course *Python A-Z: Veri Bilimi ve Machine Learning* (M. Vahit Keskin): data literacy → Pandas → EDA → statistics → preprocessing → machine learning (regression, classification, unsupervised).

## Business question
How accurately can we estimate the historical (early-2021) nightly price of a New York City Airbnb listing from characteristics known **before guests stay**, and which characteristics drive it?
Secondary question: which listing and host characteristics are associated with Superhost status?

## Targets and metrics
- **Regression:** `price` (USD), modeled as `log(1 + price)`. Primary metric: cross-validated RMSE on the log scale; MAE and median absolute error in USD for interpretation; median baseline (`DummyRegressor`).
- **Classification:** `host_is_superhost`. Primary metric: ROC-AUC; PR-AUC, recall and F1 for the Superhost class.
- Review scores are created after a stay, so they are excluded from the primary price model.

## Data
- **Source:** Kaggle – [Airbnb Listings & Reviews](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews) (mysarahmadbhat), originally from Inside Airbnb via Maven Analytics.
- **License:** CC0 Public Domain.
- **Scope:** New York City subset – 37,012 listings × 33 columns. Snapshot is historical (last review 2021-02-05, COVID period).

Raw data is **not** committed. To reproduce:
1. Download the dataset zip from the Kaggle link above.
2. Extract `Listings.csv`, `Reviews.csv` and the two data dictionaries into `data/raw/`.

## Project structure
```
data/raw/            raw Kaggle files (not in git)
data/processed/      cleaned data produced by notebooks (not in git)
notebooks/           one notebook per stage, run in order
src/                 reusable Python functions
configs/             settings (random_state, paths)
tests/               light checks for src/ functions
reports/figures/     saved figures
reports/tables/      saved comparison tables
artifacts/           saved models and pipelines (not in git)
```

## Notebooks
| # | Notebook | Stage | Status |
|---|---|---|---|
| 01 | `01_first_look.ipynb` | Data check + first look (S1/S4) | done |
| 02 | `02_problem_definition.ipynb` | Problem definition, variable dictionary, target & metrics (S2) | done |
| 03 | `03_python_toolkit.ipynb` | Python toolkit (`src/`), tests, data-quality report, clean-clone check (S3) | done |

## Environment
```bash
conda env create -f environment.yml
conda activate dsproj
jupyter lab
```
Python 3.11, pandas 3.0, scikit-learn 1.9. Random seed: `42`.

Run the tests (from the project root, no raw data needed):
```bash
python -m tests.test_src
```

## Method notes
- The train/test split is made at the start of preprocessing; the test set is used **once**, at the end.
- All learned preprocessing steps are fitted inside scikit-learn Pipelines to avoid data leakage.
