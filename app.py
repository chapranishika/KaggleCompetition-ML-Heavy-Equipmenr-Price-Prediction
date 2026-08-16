"""
Streamlit demo for the Heavy Equipment Price Prediction project.

Loads the LightGBM model trained by train_model.py (run that first) and lets
you enter machine specs to get a predicted resale price, mirroring the
"Model 3" LightGBM pipeline in heavy-equipment-price-prediction.ipynb.
"""
import datetime
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = "model/lgb_model.joblib"
OPTIONS_PATH = "model/form_options.json"

CAT_COLS = [
    "UtilizationTier", "DrivetrainType", "CabinType", "AssetScaleFactor",
    "RegionCode", "FunctionalClassification", "Spec_BaseClass", "Spec_SubClass",
    "InventoryGroupCategory", "VendorPartnerID", "DataOriginCode",
]
NUM_COLS = ["ManufactureYear", "OperationalHoursMeter", "MachineAge", "TransactionYear", "HoursPerYear"]

st.set_page_config(page_title="Heavy Equipment Price Predictor", page_icon="🚜", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(OPTIONS_PATH) as f:
        options = json.load(f)
    return model, options


model, options = load_model()

st.title("🚜 Heavy Equipment Price Predictor")
st.caption(
    "Predicts resale price for used heavy machinery (bulldozers, excavators, loaders, etc.) "
    "using the LightGBM model from this project's notebook, trained on ~138,700 historical transactions."
)

if model is None:
    st.error("No trained model found. Run `python train_model.py` first to generate `model/lgb_model.joblib`.")
    st.stop()

choices = options["choices"]
ranges = options["ranges"]

with st.form("prediction_form"):
    st.subheader("Machine specs")
    col1, col2 = st.columns(2)
    with col1:
        asset_scale = st.selectbox("Size class", choices["AssetScaleFactor"])
        cabin_type = st.selectbox("Cabin type", choices["CabinType"])
        drivetrain = st.selectbox("Drivetrain type", choices["DrivetrainType"])
    with col2:
        spec_base = st.selectbox("Base model / class (Spec_BaseClass)", choices["Spec_BaseClass"])
        spec_sub = st.selectbox("Sub-class (Spec_SubClass)", choices["Spec_SubClass"])
        inventory_group = st.selectbox("Inventory group", choices["InventoryGroupCategory"])

    st.subheader("Usage & age")
    col3, col4 = st.columns(2)
    with col3:
        manufacture_year = st.slider(
            "Manufacture year", min_value=1950, max_value=int(ranges["ManufactureYear"][1]), value=2005,
        )
        utilization_tier = st.selectbox("Utilization tier", choices["UtilizationTier"])
    with col4:
        hours = st.number_input(
            "Operational hours (meter reading)", min_value=0,
            max_value=int(ranges["OperationalHoursMeter"][1]) * 3, value=1600, step=50,
        )
        transaction_date = st.date_input("Transaction date", value=datetime.date(2010, 6, 1))

    st.subheader("Transaction context")
    col5, col6 = st.columns(2)
    with col5:
        region = st.selectbox("Region", choices["RegionCode"])
        vendor = st.selectbox("Vendor / partner ID", choices["VendorPartnerID"])
    with col6:
        functional_class = st.selectbox("Functional classification", choices["FunctionalClassification"])
        data_origin = st.selectbox("Data origin code", choices["DataOriginCode"])

    submitted = st.form_submit_button("Predict price", use_container_width=True)

if submitted:
    transaction_year = transaction_date.year
    machine_age = max(transaction_year - manufacture_year, 0)
    hours_per_year = hours / (machine_age + 1)

    row = {
        "UtilizationTier": utilization_tier,
        "DrivetrainType": drivetrain,
        "CabinType": cabin_type,
        "AssetScaleFactor": asset_scale,
        "RegionCode": region,
        "FunctionalClassification": functional_class,
        "Spec_BaseClass": spec_base,
        "Spec_SubClass": spec_sub,
        "InventoryGroupCategory": inventory_group,
        "VendorPartnerID": vendor,
        "DataOriginCode": data_origin,
        "ManufactureYear": manufacture_year,
        "OperationalHoursMeter": hours,
        "MachineAge": machine_age,
        "TransactionYear": transaction_year,
        "HoursPerYear": hours_per_year,
    }
    X = pd.DataFrame([row])
    for c in CAT_COLS:
        X[c] = X[c].astype("category")

    pred_log = model.predict(X[CAT_COLS + NUM_COLS])[0]
    pred_price = float(np.clip(np.expm1(pred_log), 0, None))

    st.success(f"### Predicted price: ${pred_price:,.0f}")
    st.caption(
        f"Machine age: {machine_age} years · Hours/year: {hours_per_year:,.0f} · "
        f"Training data price range: ${ranges['TargetValue'][0]:,.0f} – ${ranges['TargetValue'][1]:,.0f}"
    )

st.divider()
st.caption(
    "Model: LightGBM (native categorical handling), trained on log1p(price), evaluated via RMSLE. "
    "See the full notebook and validation results in this repo's README."
)
