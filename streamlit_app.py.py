import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Load artifacts ─────────────────────────────────────────────────────────────
model         = joblib.load('lr_model.joblib')
scaler        = joblib.load('scaler.joblib')
feature_names = joblib.load('feature_names.joblib')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maternal Mortality Risk — KDHS 2022",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Maternal Mortality Risk Prediction")
st.markdown("**KDHS 2022 | Logistic Regression Model | AUC-ROC: 0.699**")
st.warning(
    "⚠️ This tool is for research and programme planning only. "
    "Risk scores should inform triage prioritisation, not replace clinical judgment."
)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Individual Risk", "🗺️ County Risk Map", "📊 Model Performance"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INDIVIDUAL RISK
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Individual-Level Risk Assessment")
    st.markdown("Enter patient and county details to generate a mortality risk score.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Individual Factors")
        age               = st.slider("Age (years)", 15, 49, 25)
        parity            = st.slider("Parity (number of births)", 0, 10, 2)
        anc_visits        = st.slider("ANC Visits", 0, 12, 4)
        wealth_quintile   = st.selectbox("Wealth Quintile", [1, 2, 3, 4, 5],
                                          format_func=lambda x: f"{x} — {'Poorest' if x==1 else 'Richest' if x==5 else ''}")
        education         = st.selectbox("Education Level",
                                          ["Unknown", "Primary", "Secondary", "Higher"])
        skilled_attendant = st.selectbox("Skilled Birth Attendant", ["Not Skilled", "Skilled"])
        delivery_place    = st.selectbox("Place of Delivery", ["Home", "Private Facility", "Public Facility"])

    with col2:
        st.markdown("##### County-Level Factors")
        distance_facility = st.slider("Distance to Facility (km)", 0, 100, 20)
        poverty_rate      = st.slider("County Poverty Rate (%)", 0, 100, 35)
        sba_coverage      = st.slider("SBA Coverage (%)", 0, 100, 60)
        county_edu_index  = st.slider("County Education Index", 0.0, 100.0, 50.0)
        hiv_prev          = st.slider("HIV Prevalence (%)", 0.0, 20.0, 5.0)

    # ── Encode inputs ──────────────────────────────────────────────────────────
    education_enc    = {"Unknown": 0, "Primary": 1, "Secondary": 2, "Higher": 3}[education]
    skilled_enc      = 0 if skilled_attendant == "Not Skilled" else 1
    delivery_private = 1 if delivery_place == "Private Facility" else 0
    delivery_public  = 1 if delivery_place == "Public Facility" else 0

    input_data = pd.DataFrame([[
        age, parity, anc_visits, wealth_quintile,
        distance_facility, poverty_rate, sba_coverage,
        county_edu_index, hiv_prev,
        education_enc, skilled_enc,
        delivery_private, delivery_public
    ]], columns=feature_names)

    input_scaled = scaler.transform(input_data)
    prob         = model.predict_proba(input_scaled)[0][1]

    # ── Risk tier — thresholds derived from KDHS test set death distribution ───
    if prob < 0.43:
        tier = "🟢 Low Risk"
    elif prob < 0.57:
        tier = "🟡 Moderate Risk"
    else:
        tier = "🔴 High Risk"

    st.divider()
    st.subheader("Risk Score")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Predicted Mortality Probability", f"{prob:.1%}")
    with col_b:
        st.markdown(f"### {tier}")

    st.caption(
        "Risk tiers derived from KDHS test set death distribution: "
        "Low < 43% | Moderate 43–57% | High > 57%. "
        "Thresholds correspond to the 25th percentile and median of actual death probabilities. "
        "These should be validated against local clinical capacity before operational use."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COUNTY RISK MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("County-Level Risk Overview")
    st.markdown(
        "Risk scores are generated for a representative woman profile "
        "(age 25, parity 2, 4 ANC visits, secondary education, skilled attendant, "
        "public facility delivery) varying only by county-level contextual factors. "
        "County aggregates derived from KDHS 2022 survey data."
    )

    county_data = {
        "County": [
            "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet",
            "Embu", "Garissa", "Homa Bay", "Isiolo", "Kajiado",
            "Kakamega", "Kericho", "Kiambu", "Kilifi", "Kirinyaga",
            "Kisii", "Kisumu", "Kitui", "Kwale", "Laikipia",
            "Lamu", "Machakos", "Makueni", "Mandera", "Marsabit",
            "Meru", "Migori", "Mombasa", "Murang'a", "Nairobi",
            "Nakuru", "Nandi", "Narok", "Nyamira", "Nyandarua",
            "Nyeri", "Samburu", "Siaya", "Taita Taveta", "Tana River",
            "Tharaka-Nithi", "Trans Nzoia", "Turkana", "Uasin Gishu",
            "Vihiga", "Wajir", "West Pokot"
        ],
        "poverty_rate": [
            35.57, 33.83, 34.91, 33.28, 38.00,
            35.12, 35.52, 33.56, 35.86, 35.03,
            34.53, 34.19, 36.26, 33.21, 36.66,
            35.23, 36.22, 35.22, 35.02, 34.65,
            34.62, 35.58, 36.34, 34.97, 33.93,
            35.22, 35.47, 36.49, 36.03, 34.62,
            32.37, 34.48, 35.93, 36.28, 33.85,
            33.54, 38.95, 34.66, 35.99, 34.89,
            33.97, 36.38, 34.45, 37.27, 34.58,
            35.79, 34.46
        ],
        "sba_coverage": [
            75.02, 76.29, 75.67, 73.82, 75.42,
            75.56, 74.40, 75.74, 75.14, 76.85,
            77.81, 74.90, 75.52, 74.97, 76.41,
            74.82, 76.14, 74.37, 75.83, 74.82,
            74.96, 71.97, 74.10, 75.39, 74.24,
            74.82, 74.50, 74.49, 73.69, 75.01,
            74.79, 74.47, 77.14, 75.29, 77.05,
            77.34, 75.42, 74.10, 75.60, 75.87,
            75.83, 75.19, 76.78, 75.67, 72.83,
            74.24, 76.24
        ],
        "county_edu_index": [
            68.46, 72.40, 70.98, 72.84, 73.11,
            71.87, 73.36, 75.09, 71.99, 72.66,
            73.48, 71.88, 72.34, 72.68, 72.13,
            71.50, 71.74, 72.71, 74.25, 71.90,
            73.16, 72.90, 71.71, 74.68, 73.42,
            72.15, 72.77, 73.08, 74.02, 72.77,
            72.28, 72.19, 70.22, 72.57, 70.74,
            72.89, 72.34, 72.08, 72.72, 73.77,
            71.94, 73.89, 71.27, 72.73, 71.31,
            73.55, 70.67
        ],
        "hiv_prev": [
            6.00, 5.92, 6.02, 6.11, 6.24,
            6.04, 6.06, 5.86, 6.00, 5.94,
            5.93, 6.28, 6.06, 6.18, 5.96,
            5.66, 6.15, 5.84, 6.25, 5.90,
            5.95, 6.11, 6.05, 5.99, 5.78,
            6.28, 6.07, 5.98, 5.87, 5.74,
            5.81, 6.36, 5.83, 6.19, 5.94,
            6.20, 5.85, 6.02, 5.88, 6.00,
            6.37, 5.91, 6.13, 5.88, 6.27,
            6.00, 6.12
        ],
        "distance_facility": [
            29.29, 29.67, 28.63, 30.15, 29.45,
            30.93, 27.26, 27.92, 25.27, 25.66,
            26.44, 28.11, 24.80, 26.83, 26.48,
            25.52, 29.50, 27.13, 27.36, 27.60,
            26.55, 26.19, 29.87, 26.70, 27.95,
            25.70, 27.43, 26.57, 27.84, 27.50,
            26.65, 28.33, 26.92, 27.82, 27.81,
            27.01, 28.04, 27.22, 28.47, 25.68,
            26.81, 26.73, 28.97, 30.16, 28.33,
            27.12, 27.13
        ]
    }

    county_df = pd.DataFrame(county_data)

    scores = []
    for _, row in county_df.iterrows():
        profile = pd.DataFrame([[
            25, 2, 4, 3,
            row["distance_facility"], row["poverty_rate"],
            row["sba_coverage"], row["county_edu_index"],
            row["hiv_prev"], 2, 1, 0, 1
        ]], columns=feature_names)
        p = model.predict_proba(scaler.transform(profile))[0][1]
        scores.append(round(p * 100, 1))

    county_df["risk_score_%"] = scores
    county_df = county_df.sort_values("risk_score_%", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 14))
    colors = ["#e76f51" if s > 57 else "#f4a261" if s > 43 else "#2a9d8f"
              for s in county_df["risk_score_%"]]
    ax.barh(county_df["County"], county_df["risk_score_%"], color=colors)
    ax.axvline(x=43, color="gray", linestyle="--", linewidth=1, label="Moderate threshold (43%)")
    ax.axvline(x=57, color="red",  linestyle="--", linewidth=1, label="High threshold (57%)")
    ax.set_xlabel("Predicted Mortality Risk (%)")
    ax.set_title("County-Level Maternal Mortality Risk\n(Fixed individual profile, varying county context)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        county_df[["County", "poverty_rate", "sba_coverage", "county_edu_index", "risk_score_%"]],
        use_container_width=True
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", "0.699")
    col2.metric("AUC-PR", "0.200")
    col3.metric("Baseline AUC-PR", "~0.100", delta="2x lift")

    st.markdown("""
    **Model:** Logistic Regression (L1/L2 regularisation, class_weight='balanced')

    **Training:** Stratified 80/20 split, 5-fold StratifiedKFold, RandomizedSearchCV

    **Top predictors:** poverty_rate, education_enc (social determinants dominate)

    **Risk thresholds:** Derived from the KDHS test set death probability distribution.
    Low < 43% | Moderate 43–57% | High > 57%.
    Thresholds correspond to the 25th percentile and median of actual death probabilities.

    **County data source:** County-level aggregates derived directly from KDHS 2022 survey data
    (all 47 counties).

    **Limitation:** KDHS survey data is designed for population-level estimates.
    AUC-PR of 0.200 represents a 2x lift over random baseline but is insufficient
    for individual clinical screening. The model is best used as a population-level
    planning tool to identify high-burden subgroups and counties.
    """)

    st.info(
        "Full methodology, feature selection rationale, and threshold sensitivity "
        "analysis are documented in the project notebook."
    )
