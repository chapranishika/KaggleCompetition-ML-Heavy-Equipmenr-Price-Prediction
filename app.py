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

VALIDATION_TABLE = pd.DataFrame(
    {
        "Model": ["LightGBM", "CatBoost", "XGBoost", "RandomForest"],
        "MAE ($)": [6246.12, 6897.24, 7993.76, 9660.99],
        "RMSE ($)": [9139.87, 9945.99, 11482.71, 14041.54],
        "RMSLE": [0.2255, 0.2433, 0.2757, 0.3266],
        "R²": [0.8781, 0.8556, 0.8076, 0.7123],
    }
).set_index("Model")

st.set_page_config(page_title="Heavy Equipment Price Predictor", page_icon="🚜", layout="wide")

# ----------------------------------------------------------------------------
# Styling — Streamlit's defaults are flat; this adds real elevation, a
# construction-equipment-appropriate palette, and clear section structure.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .hero {
        background: linear-gradient(135deg, #1F2937 0%, #2D3748 55%, #B4590E 130%);
        border-radius: 20px;
        padding: 2.25rem 2.5rem;
        color: #F9FAFB;
        box-shadow: 0 20px 40px -18px rgba(31, 41, 55, 0.55);
        margin-bottom: 1.75rem;
    }
    .hero h1 { margin: 0 0 .5rem 0; font-size: 2rem; font-weight: 800; color: #FFFFFF; }
    .hero p { margin: 0; font-size: 1.02rem; line-height: 1.55; color: #E5E7EB; max-width: 760px; }
    .hero .badge {
        display: inline-block; background: rgba(232, 135, 30, 0.18); color: #FBBF6B;
        border: 1px solid rgba(251, 191, 107, 0.35); border-radius: 999px;
        padding: .2rem .75rem; font-size: .78rem; font-weight: 600; letter-spacing: .02em;
        margin-bottom: .9rem;
    }

    .card h3 { margin-top: 0; }

    /* Native st.container(border=True) → give it the same elevated-card look */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px !important;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 10px 24px -10px rgba(16,24,40,0.10);
        border: 1px solid rgba(16,24,40,0.04) !important;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 16px !important; }

    .stat-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: .5rem 0 0 0; }
    .stat-pill {
        background: #F3F4F6; border-radius: 12px; padding: .7rem 1rem; flex: 1; min-width: 140px;
        border: 1px solid rgba(16,24,40,0.05);
    }
    .stat-pill .label { font-size: .74rem; color: #6B7280; text-transform: uppercase; letter-spacing: .04em; }
    .stat-pill .value { font-size: 1.25rem; font-weight: 700; color: #1F2937; margin-top: .1rem; }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.75rem 1.9rem 1.1rem 1.9rem;
        box-shadow: 0 1px 2px rgba(16,24,40,0.04), 0 10px 24px -10px rgba(16,24,40,0.10);
        border: 1px solid rgba(16,24,40,0.04);
    }
    .section-label {
        font-weight: 700; font-size: 1.02rem; color: #1F2937; margin: .2rem 0 .9rem 0;
        display: flex; align-items: center; gap: .5rem;
    }
    .section-sub { color: #6B7280; font-size: .85rem; margin: -.6rem 0 1rem 0; }

    div[data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #E8871E, #B4590E);
        color: white; border: none; font-weight: 700; padding: .7rem 0;
        box-shadow: 0 10px 20px -8px rgba(180, 89, 14, 0.55);
        transition: transform .12s ease;
    }
    div[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); }

    .result-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFF7ED 100%);
        border: 1px solid #FDE0B8;
        border-radius: 18px;
        padding: 1.75rem 2rem;
        box-shadow: 0 18px 34px -18px rgba(180, 89, 14, 0.35);
        margin-top: 1.4rem;
    }
    .result-price { font-size: 2.6rem; font-weight: 800; color: #B4590E; margin: .1rem 0 .2rem 0; }
    .result-label { color: #6B7280; font-size: .85rem; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }

    .range-track {
        position: relative; height: 10px; border-radius: 999px; margin: 1rem 0 .3rem 0;
        background: linear-gradient(90deg, #DCEFE4, #FFF3D6, #FDE1E1);
    }
    .range-marker {
        position: absolute; top: -6px; width: 3px; height: 22px; background: #1F2937;
        border-radius: 2px; box-shadow: 0 0 0 3px rgba(31,41,55,0.12);
    }
    .range-caption { display: flex; justify-content: space-between; color: #6B7280; font-size: .78rem; }

    .step-item { display: flex; gap: .8rem; margin-bottom: .7rem; align-items: flex-start; }
    .step-num {
        background: #1F2937; color: #fff; width: 22px; height: 22px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center; font-size: .74rem; font-weight: 700;
        flex-shrink: 0; margin-top: .1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    with open(OPTIONS_PATH) as f:
        options = json.load(f)
    return model, options


model, options = load_model()

# ----------------------------------------------------------------------------
# Hero — states the problem up front so nothing below needs guessing at.
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="badge">🚜 REGRESSION MODEL · RMSLE 0.198 ON KAGGLE LEADERBOARD</div>
        <h1>Heavy Equipment Price Predictor</h1>
        <p>
        Used bulldozers, excavators, and loaders don't have a fixed resale price like new
        machinery — value depends on age, usage, configuration, and region. This tool replaces
        manual appraisal with a LightGBM model trained on <b>138,700 historical transactions</b>,
        so you can enter a machine's specs below and get an instant estimated sale price.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error("No trained model found. Run `python train_model.py` first to generate `model/lgb_model.joblib`.")
    st.stop()

choices = options["choices"]
ranges = options["ranges"]

tab_predict, tab_problem, tab_model = st.tabs(["🔮  Predict a Price", "📖  Problem & Data", "📊  Model Performance"])

# ----------------------------------------------------------------------------
# TAB: Predict
# ----------------------------------------------------------------------------
with tab_predict:
    with st.form("prediction_form"):
        st.markdown('<div class="section-label">🔧 Machine specs</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">What the machine physically is — its size class, model line, and build.</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            asset_scale = st.selectbox(
                "Size class", choices["AssetScaleFactor"],
                help="Overall size category, from Mini/Compact up to Large. Bigger machines generally sell for more.",
            )
            cabin_type = st.selectbox(
                "Cabin type", choices["CabinType"],
                help="EROPS = enclosed cab (w/ or w/o AC); OROPS = open cab. Enclosed cabs typically resell higher.",
            )
            drivetrain = st.selectbox(
                "Drivetrain type", choices["DrivetrainType"],
                help="How power reaches the wheels/tracks — e.g. Powershift, Hydrostatic, Direct Drive.",
            )
        with col2:
            spec_base = st.selectbox(
                "Base model / class", choices["Spec_BaseClass"],
                help="Manufacturer's base model code (e.g. '950', '310', 'PC200') — which product line this machine is.",
            )
            spec_sub = st.selectbox(
                "Sub-class / trim", choices["Spec_SubClass"],
                help="A finer-grained variant or trim code within the base model.",
            )
            inventory_group = st.selectbox(
                "Inventory group", choices["InventoryGroupCategory"],
                help="Broad equipment category, e.g. WL = Wheel Loader, BL = Backhoe Loader, TEX = Track Excavator.",
            )

        st.divider()
        st.markdown('<div class="section-label">⏱️ Usage & age</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">How old the machine is and how hard it has been worked — the strongest '
            'price drivers found in our EDA.</div>',
            unsafe_allow_html=True,
        )
        col3, col4 = st.columns(2)
        with col3:
            manufacture_year = st.slider(
                "Manufacture year", min_value=1950, max_value=int(ranges["ManufactureYear"][1]), value=2005,
                help="The year the machine rolled off the factory line.",
            )
            utilization_tier = st.selectbox(
                "Utilization tier", choices["UtilizationTier"],
                help="A coarse usage-intensity label (Low / Medium / High / Unknown) assigned at sale time.",
            )
        with col4:
            hours = st.number_input(
                "Operational hours (meter reading)", min_value=0,
                max_value=int(ranges["OperationalHoursMeter"][1]) * 3, value=1600, step=50,
                help="Total hours on the machine's built-in hour meter — the equivalent of an odometer.",
            )
            transaction_date = st.date_input(
                "Transaction date", value=datetime.date(2010, 6, 1),
                help="The sale date. Combined with manufacture year, this determines the machine's age at sale.",
            )

        st.divider()
        st.markdown('<div class="section-label">📍 Transaction context</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Where and through whom the sale happened.</div>',
            unsafe_allow_html=True,
        )
        col5, col6 = st.columns(2)
        with col5:
            region = st.selectbox(
                "Region", choices["RegionCode"],
                help="US state where the sale took place. Regional demand/supply shifts typical prices.",
            )
            vendor = st.selectbox(
                "Vendor / partner ID", choices["VendorPartnerID"],
                help="Code identifying which vendor or reseller handled the transaction.",
            )
        with col6:
            functional_class = st.selectbox(
                "Functional classification", choices["FunctionalClassification"],
                help="Machine type + capacity sub-range, e.g. 'Wheel Loader - 150.0 to 175.0 Horsepower'.",
            )
            data_origin = st.selectbox(
                "Data origin code", choices["DataOriginCode"],
                help="Internal code marking which data source/auction this record came from.",
            )

        submitted = st.form_submit_button("Predict price →", use_container_width=True)

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

        lo, hi = ranges["TargetValue"]
        pct = min(max((pred_price - lo) / (hi - lo), 0), 1) * 100

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Predicted resale price</div>
                <div class="result-price">${pred_price:,.0f}</div>
                <div class="range-track"><div class="range-marker" style="left: calc({pct:.1f}% - 2px);"></div></div>
                <div class="range-caption"><span>${lo:,.0f} (cheapest in training data)</span><span>${hi:,.0f} (priciest in training data)</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("#### What just happened")
            st.markdown(
                f"""
                <div class="step-item"><div class="step-num">1</div>
                    <div>Your 11 spec/context choices were combined with 2 <b>auto-computed</b> features:
                    <b>Machine Age</b> = {machine_age} years (transaction year − manufacture year) and
                    <b>Hours/Year</b> = {hours_per_year:,.0f} (operational hours ÷ (age + 1)) — usage intensity
                    turned out to matter more than raw cumulative hours in our EDA.</div>
                </div>
                <div class="step-item"><div class="step-num">2</div>
                    <div>All 16 features were handed to the trained <b>LightGBM</b> model, which predicts price in
                    log-space (it was trained on <code>log1p(price)</code> to align with the RMSLE metric used to
                    score this competition — see the Model Performance tab).</div>
                </div>
                <div class="step-item"><div class="step-num">3</div>
                    <div>The log-space output was inverted with <code>expm1</code> and clipped at $0 to get the
                    final dollar estimate above, and plotted against the real training-data price range
                    ($7,500–$142,000) so you can see how extreme or typical this estimate is.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------
# TAB: Problem & Data
# ----------------------------------------------------------------------------
with tab_problem:
    with st.container(border=True):
        st.markdown("### The problem")
        st.markdown(
            """
            Heavy machinery — bulldozers, excavators, loaders — has a large, active resale market.
            When a company finishes using a machine, or refreshes its fleet, that machine gets sold to
            someone else. Unlike new equipment with a fixed manufacturer price, a **used machine's value
            depends on a tangle of factors**: age, usage, exact configuration, where it's sold, and market
            conditions.

            This has historically been priced by human appraisers using experience and judgment — slow,
            inconsistent between appraisers, and impossible to scale to thousands of transactions. With a
            large enough history of past sales (specs paired with actual sale price), a model can learn the
            underlying pricing pattern and apply it automatically, consistently, and instantly.
            """
        )

    with st.container(border=True):
        st.markdown("### The data")
        st.markdown(
            "~138,700 historical transactions, ~50 spec/usage/transaction columns, and one target: "
            "**TargetValue** — the actual sale price."
        )
        st.markdown(
            """
            <div class="stat-row">
                <div class="stat-pill"><div class="label">Rows × Columns</div><div class="value">138,701 × 50</div></div>
                <div class="stat-pill"><div class="label">Price range</div><div class="value">$7.5K – $142K</div></div>
                <div class="stat-pill"><div class="label">Mean price</div><div class="value">~$41,522</div></div>
                <div class="stat-pill"><div class="label">Hours meter populated</div><div class="value">~59%</div></div>
                <div class="stat-pill"><div class="label">Utilization tier populated</div><div class="value">~36%</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            """
            A lot of columns are missing values on purpose (real-world messiness), and several — like
            `UtilizationTier` — are missing on the majority of rows. That's part of why the notebook spends
            real effort on EDA and imputation strategy before any modeling starts, rather than assuming a
            one-size-fits-all fix.
            """
        )

# ----------------------------------------------------------------------------
# TAB: Model Performance
# ----------------------------------------------------------------------------
with tab_model:
    with st.container(border=True):
        st.markdown("### Why RMSLE, and why log-transform the target")
        st.markdown(
            """
            The competition is scored on **RMSLE** (Root Mean Squared Log Error), which penalizes
            *relative* error rather than absolute error — being off by \\$5,000 on a \\$10,000 machine is
            judged much worse than being off by \\$5,000 on a \\$150,000 machine. Training directly on
            `log1p(TargetValue)` aligns the model's own loss function with how it's actually graded,
            instead of optimizing for a different objective and hoping it correlates.
            """
        )

    with st.container(border=True):
        st.markdown("### Model comparison (held-out 20% validation split)")
        st.dataframe(
            VALIDATION_TABLE.style.format(
                {"MAE ($)": "{:,.2f}", "RMSE ($)": "{:,.2f}", "RMSLE": "{:.4f}", "R²": "{:.4f}"}
            ),
            use_container_width=True,
        )
        st.markdown(
            """
            **LightGBM wins on every metric** — its native categorical handling (no one-hot encoding
            needed) let it use high-cardinality columns like `Spec_BaseClass` (1,249 unique values) that
            the other pipelines couldn't practically include. This ranking matches the real Kaggle
            leaderboard exactly, which is why this app uses the LightGBM model rather than the full
            LightGBM+CatBoost blend used for the final competition submission (0.198 RMSLE) — it's
            standalone, simpler to ship, and nearly as accurate.
            """
        )

st.markdown(
    "<div style='text-align:center; color:#9CA3AF; font-size:.8rem; margin-top:1rem;'>"
    "Built on a LightGBM regressor trained on log1p(price) · Full notebook and validation details in this repo's README."
    "</div>",
    unsafe_allow_html=True,
)
