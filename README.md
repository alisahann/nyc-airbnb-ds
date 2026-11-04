# NYC Airbnb – End-to-End Data Science Project

Design Project course work. An end-to-end data science project on New York City Airbnb listings, following the structure of the course *Python A-Z: Veri Bilimi ve Machine Learning* (M. Vahit Keskin): data literacy → Pandas → EDA → statistics → preprocessing → machine learning (regression, classification, unsupervised).

## Business question (draft)
What drives the nightly price of an Airbnb listing in New York City, and how accurately can we estimate a listing's (historical) price from its characteristics?

## Data
- **Source:** Kaggle – [Airbnb Listings & Reviews](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews) (mysarahmadbhat), originally from Inside Airbnb via Maven Analytics.
- **License:** CC0 Public Domain.
- **Scope:** New York City subset – 37,012 listings × 33 columns. Snapshot is historical (last review 2021-02-05, COVID period).
- **Target:** `price` (USD per night). Classification candidate: `host_is_superhost`.

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

## Environment
```bash
conda env create -f environment.yml
conda activate dsproj
jupyter lab
```
Python 3.11, pandas 3.0, scikit-learn 1.9. Random seed: `42`.

## Method notes
- The train/test split is made at the start of preprocessing; the test set is used **once**, at the end.
- All learned preprocessing steps are fitted inside scikit-learn Pipelines to avoid data leakage.
