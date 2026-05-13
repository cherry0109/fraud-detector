"""
Healthcare Claims Fraud Detection — AI Explainer
=================================================
Takes a flagged claim and its fraud score and asks
Azure OpenAI GPT-4o to write a plain English explanation
that an investigator can act on in 30 seconds.

Usage:
    python ai_explainer.py                  # runs a demo explanation
    python ai_explainer.py --demo           # same, explicit

Requires:
    AZURE_OPENAI_KEY        in .env or Replit Secrets
    AZURE_OPENAI_ENDPOINT   in .env or Replit Secrets
    AZURE_OPENAI_DEPLOYMENT in .env or Replit Secrets (e.g. gpt-4o)

If no Azure credentials are set, the file runs in DEMO MODE
and returns a realistic hardcoded explanation so you can
test the dashboard and API without an Azure account.
"""

import os
import json
import pickle
import argparse
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

AZURE_KEY        = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_ENDPOINT   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
MODEL_PATH       = "fraud_model.pkl"
DATA_PATH        = "claims.csv"

DEMO_MODE = not AZURE_KEY or not AZURE_ENDPOINT


# ── PROMPT BUILDER ────────────────────────────────────────────────────────────

def build_prompt(claim, score_result):
    """
    Build the prompt that gets sent to GPT-4o.
    Structured so the model writes like a senior investigator
    briefing a colleague — not like a data scientist.
    """
    risk  = score_result["risk_level"]
    prob  = score_result["fraud_probability"]

    prompt = f"""
You are a senior healthcare fraud investigator reviewing a flagged insurance claim.
Write a concise 3-sentence case summary for another investigator.

Your summary must:
1. State what specific pattern triggered the flag and why it is suspicious
2. Reference the actual numbers from the claim (amounts, codes, counts)
3. Give a clear recommended action (approve, escalate, hold for review)

Do not use bullet points. Write in plain professional English.
Do not start with "This claim" — vary your opening.
Do not explain what fraud is in general — focus only on this specific claim.

CLAIM DATA:
Claim ID         : {claim.get('claim_id', 'N/A')}
Provider         : {claim.get('provider_name', 'N/A')} (NPI: {claim.get('provider_npi', 'N/A')})
Provider Specialty: {claim.get('provider_specialty', 'N/A')}
Patient          : {claim.get('patient_id', 'N/A')} | {claim.get('patient_gender', 'N/A')} | DOB {claim.get('patient_dob', 'N/A')}
Diagnosis        : {claim.get('diagnosis_code', 'N/A')} — {claim.get('diagnosis_desc', 'N/A')}
Procedure        : {claim.get('procedure_code', 'N/A')} — {claim.get('procedure_desc', 'N/A')}
Billed Amount    : ${claim.get('billed_amount', 0):,.2f}
Allowed Amount   : ${claim.get('allowed_amount', 0):,.2f}
Paid Amount      : ${claim.get('paid_amount', 0):,.2f}
Units Billed     : {claim.get('units_billed', 1)}
Place of Service : {claim.get('place_of_service', 'N/A')}
Claim Date       : {claim.get('claim_date', 'N/A')}
Plan Type        : {claim.get('plan_type', 'N/A')}

ML SIGNALS:
Fraud Probability : {prob:.1%}
Risk Level        : {risk}
Amount Z-Score    : {claim.get('amount_zscore', 0):.2f} standard deviations above average for this procedure
Provider Volume   : {int(claim.get('provider_claim_count', 0))} total claims from this provider
Patient Volume    : {int(claim.get('patient_claim_count', 0))} total claims for this patient
Units Flag        : {"Yes — unusually high" if claim.get('high_units', 0) else "No"}
Diagnosis Mismatch: {"Yes — procedure does not match diagnosis" if claim.get('diag_proc_mismatch', 0) else "No"}
Paid Ratio        : {claim.get('paid_ratio', 0):.2f} (paid / billed)

Write your 3-sentence investigator summary now:
""".strip()

    return prompt


# ── AZURE OPENAI CALL ─────────────────────────────────────────────────────────

def call_azure_openai(prompt):
    """Call Azure OpenAI and return the explanation text."""
    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=AZURE_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version="2024-02-01",
        )

        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior healthcare fraud investigator. "
                        "You write clear, evidence-based case summaries. "
                        "You never speculate — you reference only what the data shows."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"[Azure OpenAI error: {e}]"


# ── DEMO EXPLANATIONS ─────────────────────────────────────────────────────────

DEMO_EXPLANATIONS = {
    "HIGH": [
        "Provider NPI {npi} billed procedure {proc} {units} times in a single encounter on {date}, which is {zscore:.1f} standard deviations above the average billing volume for this specialty. The billed amount of ${amount:,.2f} against a {diag} diagnosis is inconsistent with the documented level of care. Recommend immediate hold and escalation to the Special Investigations Unit.",

        "The billed amount of ${amount:,.2f} for procedure {proc} is {zscore:.1f} standard deviations above the expected range for this procedure code, and the diagnosis code {diag_code} does not clinically support a procedure of this complexity or cost. Provider {provider} has submitted {prov_count} claims in this period, which is unusually high for their specialty. Escalate to senior investigator before releasing payment.",

        "A {diag} diagnosis paired with procedure {proc} at ${amount:,.2f} represents a clinically implausible combination — routine checkup diagnoses do not support high-cost surgical billing. The fraud model scored this claim at {prob:.0%} confidence with a {zscore:.1f} standard deviation anomaly in the billed amount. Hold payment and open a formal case review.",
    ],
    "MEDIUM": [
        "Provider {provider} billed ${amount:,.2f} for procedure {proc}, which sits {zscore:.1f} standard deviations above the average for this code — elevated but not immediately conclusive. The claim warrants secondary review against the provider's recent billing history before payment is released. Flag for secondary review queue.",

        "The billed amount of ${amount:,.2f} is higher than expected for procedure {proc} on a {plan} plan, and the provider volume of {prov_count} claims is above average for this specialty. No single signal is definitive but the combination warrants a closer look. Route to secondary review before approving.",
    ],
    "LOW": [
        "Claim {claim_id} scored {prob:.0%} fraud probability with no individual signals crossing critical thresholds. Billed amount and procedure coding are within normal ranges for this provider and diagnosis. Approve for payment.",
    ],
}


def get_demo_explanation(claim, score_result):
    """Return a realistic hardcoded explanation for demo mode."""
    import random
    risk = score_result["risk_level"]
    templates = DEMO_EXPLANATIONS.get(risk, DEMO_EXPLANATIONS["LOW"])
    template  = random.choice(templates)

    try:
        explanation = template.format(
            claim_id   = claim.get("claim_id", "N/A"),
            npi        = claim.get("provider_npi", "N/A"),
            provider   = claim.get("provider_name", "N/A"),
            proc       = claim.get("procedure_code", "N/A"),
            diag       = claim.get("diagnosis_desc", "N/A"),
            diag_code  = claim.get("diagnosis_code", "N/A"),
            date       = claim.get("claim_date", "N/A"),
            amount     = claim.get("billed_amount", 0),
            units      = claim.get("units_billed", 1),
            zscore     = claim.get("amount_zscore", 0),
            prov_count = int(claim.get("provider_claim_count", 0)),
            pat_count  = int(claim.get("patient_claim_count", 0)),
            plan       = claim.get("plan_type", "N/A"),
            prob       = score_result["fraud_probability"],
        )
    except KeyError:
        explanation = (
            f"Claim scored {score_result['fraud_probability']:.0%} fraud probability "
            f"with risk level {risk}. Review recommended."
        )

    return explanation


# ── MAIN EXPLAINER FUNCTION ───────────────────────────────────────────────────

def explain_claim(claim, score_result):
    """
    Main function called by dashboard.py and api.py.
    Pass a claim dict and a score_result dict.
    Returns a plain English explanation string.

    Example:
        from ai_explainer import explain_claim
        explanation = explain_claim(claim_dict, score_result)
    """
    if DEMO_MODE:
        return get_demo_explanation(claim, score_result)

    prompt = build_prompt(claim, score_result)
    return call_azure_openai(prompt)


# ── DEMO RUNNER ───────────────────────────────────────────────────────────────

def run_demo():
    """
    Load real claims from claims.csv, pick 3 examples
    (high, medium, low risk) and show what the AI explanation
    looks like for each one.
    """
    print("\n" + "═" * 60)
    print("  AI EXPLAINER — DEMO MODE")
    if DEMO_MODE:
        print("  (Running with demo explanations — add Azure keys for live GPT-4o)")
    else:
        print("  (Running with Azure OpenAI GPT-4o)")
    print("═" * 60)

    # Load model
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    model         = payload["model"]
    feature_names = payload["feature_names"]

    # Load claims
    df = pd.read_csv(DATA_PATH)

    # Score all claims
    X = df[feature_names].fillna(0)
    df["fraud_prob"] = model.predict_proba(X)[:, 1]
    df["risk_level"] = df["fraud_prob"].apply(
        lambda p: "HIGH" if p >= 0.70 else ("MEDIUM" if p >= 0.40 else "LOW")
    )

    # Pick examples
    examples = []
    for risk in ["HIGH", "MEDIUM", "LOW"]:
        subset = df[df["risk_level"] == risk]
        if len(subset):
            examples.append(subset.iloc[0])

    for row in examples:
        claim = row.to_dict()
        score_result = {
            "fraud_probability": round(float(claim["fraud_prob"]), 3),
            "is_fraud": float(claim["fraud_prob"]) >= 0.50,
            "risk_level": claim["risk_level"],
        }

        explanation = explain_claim(claim, score_result)

        print(f"\n{'─' * 60}")
        print(f"  CLAIM ID  : {claim.get('claim_id', 'N/A')}")
        print(f"  PROVIDER  : {claim.get('provider_name', 'N/A')}")
        print(f"  PROCEDURE : {claim.get('procedure_code', 'N/A')} — {claim.get('procedure_desc', 'N/A')[:40]}")
        print(f"  BILLED    : ${claim.get('billed_amount', 0):,.2f}")
        print(f"  RISK      : {score_result['risk_level']} ({score_result['fraud_probability']:.1%})")
        print(f"\n  AI EXPLANATION:")
        print(f"  {explanation}")
        print(f"{'─' * 60}")

    print("\n  Done. Next step: run python dashboard.py")
    print("═" * 60 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Claim Explainer")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample claims")
    args = parser.parse_args()
    run_demo()
