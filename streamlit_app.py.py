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
        county_edu_index  = st.slider("County Education Index", 0.0, 1.0, 0.5)
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
    # Low < Q25 of deaths (0.43) | Moderate Q25–Median (0.43–0.57) | High > Median (0.57)
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
        "public facility delivery) varying only by county-level contextual factors."
    )

    county_data = {
        "County":            ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret/Uasin Gishu",
                               "Meru", "Kilifi", "Turkana", "Mandera", "Garissa"],
        "poverty_rate":      [17, 22, 35, 28, 24, 30, 55, 79, 72, 65],
        "sba_coverage":      [95, 88, 80, 82, 84, 75, 60, 35, 28, 40],
        "county_edu_index":  [0.78, 0.65, 0.62, 0.60, 0.63, 0.55, 0.42, 0.22, 0.18, 0.28],
        "hiv_prev":          [4.5, 3.8, 14.2, 5.1, 4.2, 3.5, 2.8, 1.2, 0.8, 1.5],
        "distance_facility": [5, 8, 12, 15, 10, 20, 35, 55, 60, 45]
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
    county_df = county_df.sort_values("risk_score_%", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
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
    Low < 0.43 (below Q25 of deaths) | Moderate 0.43–0.57 (Q25 to median) | High > 0.57 (above median of deaths).

    **Limitation:** KDHS survey data is designed for population-level estimates.
    AUC-PR of 0.200 represents a 2x lift over random baseline but is insufficient
    for individual clinical screening. The model is best used as a population-level
    planning tool to identify high-burden subgroups and counties.
    """)

    st.info(
        "Full methodology, feature selection rationale, and threshold sensitivity "
        "analysis are documented in the project notebook."
    )