"""
Trains the LightGBM model (native categorical handling, no K-fold target
encoding) from the notebook and saves it to model/ for the Streamlit app.

This mirrors the "Model 3" cell in heavy-equipment-price-prediction.ipynb.
Run once locally: python train_model.py
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

CAT_COLS = [
    "UtilizationTier", "DrivetrainType", "CabinType", "AssetScaleFactor",
    "RegionCode", "FunctionalClassification", "Spec_BaseClass", "Spec_SubClass",
    "InventoryGroupCategory", "VendorPartnerID", "DataOriginCode",
]
NUM_COLS = ["ManufactureYear", "OperationalHoursMeter", "MachineAge", "TransactionYear", "HoursPerYear"]

MODEL_DIR = "model"


def main():
    print("Loading train.csv...")
    train = pd.read_csv("train.csv", low_memory=False)

    train["TransactionYear"] = pd.to_datetime(train["TransactionDate"]).dt.year
    train["MachineAge"] = (train["TransactionYear"] - train["ManufactureYear"]).clip(lower=0)
    train["HoursPerYear"] = train["OperationalHoursMeter"] / (train["MachineAge"] + 1)
    train["UtilizationTier"] = train["UtilizationTier"].fillna("Unknown")
    for c in CAT_COLS:
        train[c] = train[c].fillna("Missing").astype("category")

    X = train[CAT_COLS + NUM_COLS]
    y_log = np.log1p(train["TargetValue"])

    print("Training LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=2000, max_depth=9, learning_rate=0.025, num_leaves=110,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y_log, categorical_feature=CAT_COLS)
    print("Done training.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "lgb_model.joblib"))

    # Category choices + numeric ranges, so the Streamlit form only offers real values
    choices = {c: sorted(train[c].astype(str).unique().tolist()) for c in CAT_COLS}
    ranges = {
        "ManufactureYear": [int(train["ManufactureYear"].quantile(0.01)), int(train["ManufactureYear"].max())],
        "OperationalHoursMeter": [0, int(train["OperationalHoursMeter"].quantile(0.99))],
        "TargetValue": [float(train["TargetValue"].min()), float(train["TargetValue"].max())],
    }
    with open(os.path.join(MODEL_DIR, "form_options.json"), "w") as f:
        json.dump({"choices": choices, "ranges": ranges}, f)

    print("Saved model/lgb_model.joblib and model/form_options.json")


if __name__ == "__main__":
    main()
