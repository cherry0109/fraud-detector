"""
Healthcare Claims Fraud Detection — Streamlit Dashboard
========================================================
The face of the project. A recruiter can open this,
submit a fake claim, and see a fraud score plus AI
explanation in real time.

Usage:
    streamlit run dashboard.py
"""

import pickle
import random
import pandas as pd
import streamlit as st
from ai_explainer import explain_claim

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Claims Fraud Detector",
    page_icon="🔍",
    layout="wide",
)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    with open("fraud_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_claims():
    return pd.read_csv("claims.csv")

payload       = load_model()
model         = payload["model"]
feature_names = payload["feature_names"]
metrics       = payload["metrics"]
df            = load_claims()

# Score all claims once
X = df[feature_names].fillna(0)
df["fraud_prob"] = model.predict_proba(X)[:, 1]
df["risk_level"]  = df["fraud_prob"].apply(
    lambda p: "HIGH" if p >= 0.70 else ("MEDIUM" if p >= 0.40 else "LOW")
)

# ── STYLES ────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.risk-high   { background:#fee2e2; color:#991b1b; padding:6px 16px; border-radius:20px; font-weight:600; font-size:14px; display:inline-block; }
.risk-medium { background:#fef3c7; color:#92400e; padding:6px 16px; border-radius:20px; font-weight:600; font-size:14px; display:inline-block; }
.risk-low    { background:#d1fae5; color:#065f46; padding:6px 16px; border-radius:20px; font-weight:600; font-size:14px; display:inline-block; }
.explain-box { background:#f8fafc; border-left:4px solid #6366f1; padding:1rem 1.25rem; border-radius:0 8px 8px 0; font-size:15px; line-height:1.7; color:#1e293b; margin-top:1rem; }
.metric-card { background:#f1f5f9; border-radius:10px; padding:1rem; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────

st.title("🔍 Real-Time Claims Fraud Detector")
st.markdown("Built by **Cherry Anem** · Azure · XGBoost · GPT-4o · Python")
st.markdown("---")

# ── SIDEBAR — MODEL STATS ─────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Model performance")
    st.metric("Precision", f"{metrics['precision']:.1%}")
    st.metric("Recall",    f"{metrics['recall']:.1%}")
    st.metric("F1 Score",  f"{metrics['f1']:.1%}")
    st.metric("ROC AUC",   f"{metrics['auc']:.1%}")

    st.markdown("---")
    st.markdown("### Dataset")
    total     = len(df)
    fraud_n   = df["is_fraud"].sum()
    st.metric("Total claims",    f"{total:,}")
    st.metric("Fraud cases",     f"{fraud_n:,}")
    st.metric("Fraud rate",      f"{fraud_n/total:.1%}")

    st.markdown("---")
    st.markdown("### Risk breakdown")
    for risk, color in [("HIGH","🔴"), ("MEDIUM","🟡"), ("LOW","🟢")]:
        count = (df["risk_level"] == risk).sum()
        st.markdown(f"{color} **{risk}** — {count:,} claims")

    st.markdown("---")
    st.caption("Note: perfect scores are expected on synthetic data. Real-world production models typically achieve 85–95% recall.")

# ── TABS ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Submit a claim", "Browse fraud cases", "How it works"])

# ══ TAB 1 — SUBMIT A CLAIM ═══════════════════════════════════════════════════

with tab1:
    st.markdown("### Submit a claim for fraud scoring")
    st.markdown("Fill in the claim details below or load a random example to see the model in action.")

    col_load1, col_load2, col_load3 = st.columns(3)
    load_fraud  = col_load1.button("Load a HIGH risk example")
    load_medium = col_load2.button("Load a MEDIUM risk example")
    load_legit  = col_load3.button("Load a LOW risk example")

    # Pick example claim
    if load_fraud:
        example = df[df["risk_level"] == "HIGH"].iloc[0].to_dict()
        st.session_state["example"] = example
    elif load_medium:
        example = df[df["risk_level"] == "MEDIUM"].iloc[0].to_dict()
        st.session_state["example"] = example
    elif load_legit:
        example = df[df["risk_level"] == "LOW"].iloc[0].to_dict()
        st.session_state["example"] = example

    ex = st.session_state.get("example", df.iloc[0].to_dict())

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Claim info**")
        claim_id      = st.text_input("Claim ID",        value=ex.get("claim_id", "CLM-DEMO-001"))
        claim_date    = st.text_input("Claim date",      value=str(ex.get("claim_date", "2025-01-15")))
        plan_type     = st.selectbox("Plan type",        ["PPO","HMO","EPO","HDHP","Medicare Advantage","Medicaid"],
                                     index=["PPO","HMO","EPO","HDHP","Medicare Advantage","Medicaid"].index(ex.get("plan_type","PPO")) if ex.get("plan_type","PPO") in ["PPO","HMO","EPO","HDHP","Medicare Advantage","Medicaid"] else 0)
        place_of_svc  = st.text_input("Place of service", value=str(ex.get("place_of_service", "11")))

    with col2:
        st.markdown("**Provider & patient**")
        provider_name = st.text_input("Provider name",  value=ex.get("provider_name", ""))
        provider_npi  = st.text_input("Provider NPI",   value=str(ex.get("provider_npi", "")))
        specialty     = st.text_input("Specialty",      value=ex.get("provider_specialty", ""))
        patient_id    = st.text_input("Patient ID",     value=ex.get("patient_id", ""))
        patient_dob   = st.text_input("Patient DOB",    value=str(ex.get("patient_dob", "")))
        patient_gender= st.selectbox("Gender",          ["M","F"],
                                     index=0 if ex.get("patient_gender","M")=="M" else 1)

    with col3:
        st.markdown("**Coding & amounts**")
        diag_code     = st.text_input("Diagnosis code",  value=ex.get("diagnosis_code", "Z00.00"))
        diag_desc     = st.text_input("Diagnosis desc",  value=ex.get("diagnosis_desc", "General exam"))
        proc_code     = st.text_input("Procedure code",  value=ex.get("procedure_code", "99213"))
        proc_desc     = st.text_input("Procedure desc",  value=ex.get("procedure_desc", "Office visit")[:40])
        billed        = st.number_input("Billed amount ($)",  value=float(ex.get("billed_amount", 200)), min_value=0.0, step=50.0)
        allowed       = st.number_input("Allowed amount ($)", value=float(ex.get("allowed_amount", 160)), min_value=0.0, step=50.0)
        paid          = st.number_input("Paid amount ($)",    value=float(ex.get("paid_amount", 130)),   min_value=0.0, step=50.0)
        units         = st.number_input("Units billed",       value=int(ex.get("units_billed", 1)),      min_value=1,   step=1)

    st.markdown("---")

    if st.button("Score this claim", type="primary", use_container_width=True):

        # Build feature dict
        prov_count  = int(df[df["provider_npi"] == str(provider_npi)].shape[0]) or 1
        pat_count   = int(df[df["patient_id"]   == str(patient_id)].shape[0])   or 1
        proc_mean   = df[df["procedure_code"]   == str(proc_code)]["billed_amount"].mean()
        proc_std    = df[df["procedure_code"]   == str(proc_code)]["billed_amount"].std()
        zscore      = round((billed - proc_mean) / max(proc_std, 1), 3) if proc_mean else 0.0

        surgical    = {"27447","43239","29827"}
        mismatch    = int(proc_code in surgical and diag_code == "Z00.00")
        high_units  = int(units > 3)
        paid_ratio  = round(paid / max(billed, 1), 3)

        feature_dict = {
            "amount_zscore":         zscore,
            "provider_claim_count":  prov_count,
            "patient_claim_count":   pat_count,
            "diag_proc_mismatch":    mismatch,
            "high_units":            high_units,
            "paid_ratio":            paid_ratio,
            "billed_amount":         billed,
            "allowed_amount":        allowed,
            "units_billed":          units,
        }

        X_input = pd.DataFrame([feature_dict])[feature_names].fillna(0)
        prob    = float(model.predict_proba(X_input)[0][1])
        risk    = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.40 else "LOW")

        score_result = {
            "fraud_probability": round(prob, 3),
            "is_fraud":          prob >= 0.50,
            "risk_level":        risk,
        }

        claim_dict = {
            "claim_id":            claim_id,
            "claim_date":          claim_date,
            "provider_name":       provider_name,
            "provider_npi":        provider_npi,
            "provider_specialty":  specialty,
            "patient_id":          patient_id,
            "patient_dob":         patient_dob,
            "patient_gender":      patient_gender,
            "plan_type":           plan_type,
            "place_of_service":    place_of_svc,
            "diagnosis_code":      diag_code,
            "diagnosis_desc":      diag_desc,
            "procedure_code":      proc_code,
            "procedure_desc":      proc_desc,
            "billed_amount":       billed,
            "allowed_amount":      allowed,
            "paid_amount":         paid,
            "units_billed":        units,
            "amount_zscore":       zscore,
            "provider_claim_count":prov_count,
            "patient_claim_count": pat_count,
            "diag_proc_mismatch":  mismatch,
            "high_units":          high_units,
            "paid_ratio":          paid_ratio,
        }

        # Results
        st.markdown("---")
        st.markdown("### Scoring result")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Fraud probability", f"{prob:.1%}")
        r2.metric("Risk level",        risk)
        r3.metric("Amount z-score",    f"{zscore:.2f}σ")
        r4.metric("Units billed",      units)

        risk_class = {"HIGH":"risk-high","MEDIUM":"risk-medium","LOW":"risk-low"}[risk]
        st.markdown(f'<span class="{risk_class}">{risk} RISK</span>', unsafe_allow_html=True)

        # Progress bar
        st.markdown(f"**Fraud probability: {prob:.1%}**")
        st.progress(min(prob, 1.0))

        # AI explanation
        st.markdown("### Investigator summary")
        with st.spinner("Generating AI explanation..."):
            explanation = explain_claim(claim_dict, score_result)
        st.markdown(f'<div class="explain-box">{explanation}</div>', unsafe_allow_html=True)

        # Recommended action
        st.markdown("### Recommended action")
        if risk == "HIGH":
            st.error("🚨 Hold payment — escalate to Special Investigations Unit immediately")
        elif risk == "MEDIUM":
            st.warning("⚠️ Route to secondary review queue before releasing payment")
        else:
            st.success("✅ Clear for payment — no significant fraud signals detected")


# ══ TAB 2 — BROWSE FRAUD CASES ═══════════════════════════════════════════════

with tab2:
    st.markdown("### Fraud case browser")
    st.markdown("Browse all flagged claims in the dataset.")

    col_f1, col_f2 = st.columns(2)
    risk_filter = col_f1.multiselect(
        "Filter by risk level",
        ["HIGH","MEDIUM","LOW"],
        default=["HIGH","MEDIUM"],
    )
    fraud_filter = col_f2.checkbox("Show only confirmed fraud (labelled)", value=True)

    filtered = df[df["risk_level"].isin(risk_filter)]
    if fraud_filter:
        filtered = filtered[filtered["is_fraud"] == True]

    display_cols = [
        "claim_id","claim_date","provider_name","procedure_code",
        "billed_amount","fraud_prob","risk_level","fraud_type",
    ]
    available_cols = [c for c in display_cols if c in filtered.columns]

    st.markdown(f"Showing **{len(filtered):,}** claims")
    st.dataframe(
        filtered[available_cols].sort_values("fraud_prob", ascending=False).head(100),
        use_container_width=True,
        hide_index=True,
    )

    # Download
    csv = filtered[available_cols].to_csv(index=False)
    st.download_button(
        "Download filtered claims as CSV",
        data=csv,
        file_name="flagged_claims.csv",
        mime="text/csv",
    )


# ══ TAB 3 — HOW IT WORKS ═════════════════════════════════════════════════════

with tab3:
    st.markdown("### How this works")

    st.markdown("""
**The pipeline has four layers:**

**1. Data ingestion**
Claims arrive via API, EHR feed, or batch export. In production this runs through Azure Event Hubs with Kafka protocol, ingesting up to 18,000 transactions per second.

**2. Feature engineering**
Each claim is transformed into ML signals — billing amount z-score against procedure averages, provider claim velocity, diagnosis-to-procedure match, units billed, and paid ratio. These are the patterns that rules engines miss.

**3. Fraud scoring**
An XGBoost classifier trained on 2,000 synthetic claims scores each one from 0 to 1. Claims above 0.70 are HIGH risk. Claims between 0.40 and 0.70 are MEDIUM. Below 0.40 clears automatically.

**4. AI explanation**
Every HIGH and MEDIUM risk claim is sent to Azure OpenAI GPT-4o with the full claim context and ML signals. GPT-4o writes a 3-sentence investigator summary in plain English — no data science jargon, just actionable intelligence.
""")

    st.markdown("---")
    st.markdown("### Tech stack")

    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown("**Data & ML**")
        st.markdown("Python · Pandas · XGBoost · scikit-learn")
    with col_t2:
        st.markdown("**Cloud (Azure)**")
        st.markdown("Event Hubs · Databricks · ADLS Gen2 · Azure OpenAI")
    with col_t3:
        st.markdown("**Interface**")
        st.markdown("Streamlit · FastAPI · Power BI")

    st.markdown("---")
    st.markdown("### Model evaluation")
    try:
        st.image("model_evaluation.png", caption="Confusion matrix and feature importance", use_column_width=True)
    except:
        st.info("Run fraud_model.py to generate the evaluation chart.")
