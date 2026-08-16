# Heavy Equipment Price Prediction

A machine learning project that predicts the resale price of used heavy machinery (bulldozers, excavators, loaders, and similar equipment) from historical transaction data. Built for a Kaggle competition, scored on **RMSLE**.

## Problem

Used heavy equipment doesn't have a fixed price like new machinery — its value depends on age, usage, configuration, region, and market conditions. Historically this has been priced by human appraisers, which is slow and inconsistent. This project learns the pricing pattern from a history of past sales (specs + actual sale price) and applies it to new, unpriced machines.

## Data

| File | Description |
|---|---|
| `train.csv` | ~138,700 historical transactions with full specs and known sale price (`TargetValue`) |
| `test.csv` | ~15,000 transactions with specs only, price hidden |
| `sample_submission.csv` | Expected output format (`TransactionID` + predicted `TargetValue`) |
| `metadata.csv` | Data dictionary for the ~50 columns (many generically named, e.g. `col1`, `Spec_BaseClass`) |

Key data notes:
- 138,701 rows, 50 columns
- `OperationalHoursMeter` populated in only ~59% of rows; `UtilizationTier` in only ~36%
- `TargetValue` ranges from $7,500 to $142,000 (mean ≈ $41,522)

## Approach

1. **EDA first** — missingness, target distribution (raw vs. log), categorical-vs-price medians, numeric scatter plots, and a correlation heatmap, all used to justify downstream decisions rather than assume them.
2. **Regression framing** — `TargetValue` is continuous, so this is a regression problem evaluated with RMSLE, which penalizes *relative* error. The target is trained in `log1p` space to align the model's objective with the evaluation metric from the start.
3. **Progressive modeling** — each new model tests a specific hypothesis:
   - **RandomForest** (bagging) — simple, honest baseline inside a proper `sklearn` preprocessing `Pipeline`
   - **XGBoost** (gradient boosting) — does boosting beat bagging?
   - **LightGBM** (native categorical handling) — does removing the one-hot bottleneck help on high-cardinality columns?
   - **LightGBM + CatBoost blend** — does combining diverse boosted models reduce error further?
4. **Feature engineering** — `MachineAge` (transaction year − manufacture year) and `HoursPerYear` (usage intensity relative to age), motivated directly by EDA findings.
5. **Hyperparameter tuning** — `RandomizedSearchCV` on a reduced sample to find tuning direction, then scaled up for the final full-data model with added regularization.
6. **K-fold target encoding** — for the highest-cardinality `Spec_` columns, category-to-average-price mappings are computed strictly within K-fold splits to prevent target leakage.
7. **Honest validation** — a held-out 20% split with hand-written RMSLE/MAE/RMSE/R² alongside the Kaggle leaderboard as the ultimate check.

## Results

Validation metrics on a held-out 20% split (log-trained, evaluated in dollar space):

| Model | MAE | RMSE | RMSLE | R² |
|---|---|---|---|---|
| **LightGBM** | 6,246.12 | 9,139.87 | 0.2255 | 0.8781 |
| CatBoost | 6,897.24 | 9,945.99 | 0.2433 | 0.8556 |
| XGBoost | 7,993.76 | 11,482.71 | 0.2757 | 0.8076 |
| RandomForest | 9,660.99 | 14,041.54 | 0.3266 | 0.7123 |

**Best Kaggle leaderboard score: 0.198 RMSLE**, from a weighted blend (80% LightGBM / 20% CatBoost) of the final K-fold-target-encoded models.

## Repo Contents

- `heavy-equipment-price-prediction.ipynb` — full notebook: EDA → RandomForest → XGBoost → LightGBM → LightGBM+CatBoost blend → hyperparameter tuning → K-fold target encoding → validation comparison
- `train.csv`, `test.csv`, `sample_submission.csv`, `metadata.csv` — competition data
