"""
Healthcare Claims Fraud Detection — REST API
=============================================
A FastAPI endpoint that accepts a claim as JSON,
scores it for fraud, and returns the probability,
risk level, and AI explanation.

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8001 --reload

Then test it:
    curl -X POST http://localhost:8001/score \
      -H "Content-Type: application/json" \
      -d '{"procedure_code":"27447","diagnosis_code":"Z00.00","billed_amount":14000,"units_billed":1}'

Or open the interactive docs at:
    http://localhost:8001/docs
"""

import pickle
import pandas as pd
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ai_explainer import explain_claim

# ── APP SETUP ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Healthcare Claims Fraud Detector",
    description="Real-time fraud scoring API. Submit a claim and get a fraud probability, risk level, and plain English explanation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── LOAD MODEL ONCE AT STARTUP ────────────────────────────────────────────────

MODEL_PATH = "fraud_model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    model         = payload["model"]
    feature_names = payload["feature_names"]
    model_metrics = payload["metrics"]
    print(f"Model loaded — F1: {model_metrics['f1']:.3f} | AUC: {model_metrics['auc']:.3f}")
except FileNotFoundError:
    raise RuntimeError("fraud_model.pkl not found. Run fraud_model.py first.")

# Load claims for context lookup
try:
    df_claims = pd.read_csv("claims.csv")
except FileNotFoundError:
    df_claims = pd.DataFrame()


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────

class ClaimRequest(BaseModel):
    claim_id:           Optional[str]   = Field(None,    example="CLM-TEST-001")
    claim_date:         Optional[str]   = Field(None,    example="2025-06-01")
    provider_npi:       Optional[str]   = Field(None,    example="1234567890")
    provider_name:      Optional[str]   = Field(None,    example="Dr. Smith, John")
    provider_specialty: Optional[str]   = Field(None,    example="Internal Medicine")
    patient_id:         Optional[str]   = Field(None,    example="PAT-001")
    patient_dob:        Optional[str]   = Field(None,    example="1965-04-12")
    patient_gender:     Optional[str]   = Field(None,    example="M")
    plan_type:          Optional[str]   = Field("PPO",   example="PPO")
    diagnosis_code:     Optional[str]   = Field("Z00.00",example="Z00.00")
    diagnosis_desc:     Optional[str]   = Field(None,    example="General adult medical exam")
    procedure_code:     Optional[str]   = Field("99213", example="99213")
    procedure_desc:     Optional[str]   = Field(None,    example="Office visit, established patient")
    billed_amount:      float           = Field(200.0,   example=14000.0)
    allowed_amount:     Optional[float] = Field(None,    example=11000.0)
    paid_amount:        Optional[float] = Field(None,    example=9500.0)
    units_billed:       Optional[int]   = Field(1,       example=1)
    place_of_service:   Optional[str]   = Field("11",    example="11")
    explain:            bool            = Field(True,    description="Set to false to skip AI explanation and return faster")


class FraudScore(BaseModel):
    claim_id:           str
    scored_at:          str
    fraud_probability:  float
    is_fraud:           bool
    risk_level:         str
    explanation:        Optional[str]
    signals: dict


class ModelInfo(BaseModel):
    precision:  float
    recall:     float
    f1:         float
    auc:        float
    features:   list
    trained_on: str


# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────

def engineer_features(req: ClaimRequest) -> dict:
    """
    Build the feature vector from a raw claim request.
    Mirrors what happens in the Databricks streaming job.
    """
    billed  = req.billed_amount
    allowed = req.allowed_amount or billed * 0.80
    paid    = req.paid_amount    or allowed * 0.85
    units   = req.units_billed   or 1

    # Amount z-score vs procedure average
    if not df_claims.empty and req.procedure_code:
        subset   = df_claims[df_claims["procedure_code"] == req.procedure_code]["billed_amount"]
        proc_mean = subset.mean() if len(subset) else billed
        proc_std  = subset.std()  if len(subset) > 1 else 1.0
    else:
        proc_mean = billed
        proc_std  = 1.0

    zscore = round((billed - proc_mean) / max(proc_std, 1.0), 3)

    # Provider and patient claim counts from historical data
    if not df_claims.empty and req.provider_npi:
        prov_count = int(df_claims[df_claims["provider_npi"] == req.provider_npi].shape[0]) or 1
    else:
        prov_count = 1

    if not df_claims.empty and req.patient_id:
        pat_count = int(df_claims[df_claims["patient_id"] == req.patient_id].shape[0]) or 1
    else:
        pat_count = 1

    # Rule signals
    surgical  = {"27447", "43239", "29827"}
    mismatch  = int(
        (req.procedure_code or "") in surgical and
        (req.diagnosis_code or "") == "Z00.00"
    )
    high_units = int(units > 3)
    paid_ratio = round(paid / max(billed, 1.0), 3)

    return {
        "amount_zscore":          zscore,
        "provider_claim_count":   prov_count,
        "patient_claim_count":    pat_count,
        "diag_proc_mismatch":     mismatch,
        "high_units":             high_units,
        "paid_ratio":             paid_ratio,
        "billed_amount":          billed,
        "allowed_amount":         allowed,
        "units_billed":           units,
        # signals returned for transparency
        "_proc_mean":             round(proc_mean, 2),
        "_proc_std":              round(proc_std, 2),
    }


# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
def root():
    return {
        "service": "Healthcare Claims Fraud Detector",
        "status":  "running",
        "docs":    "/docs",
        "score":   "POST /score",
    }


@app.get("/health", tags=["health"])
def health():
    return {
        "status":    "healthy",
        "model_f1":  model_metrics["f1"],
        "model_auc": model_metrics["auc"],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/model", response_model=ModelInfo, tags=["model"])
def model_info():
    return ModelInfo(
        precision  = model_metrics["precision"],
        recall     = model_metrics["recall"],
        f1         = model_metrics["f1"],
        auc        = model_metrics["auc"],
        features   = feature_names,
        trained_on = payload.get("trained_on", "unknown"),
    )


@app.post("/score", response_model=FraudScore, tags=["scoring"])
def score_claim(req: ClaimRequest):
    """
    Score a single claim for fraud.

    Returns:
    - fraud_probability: 0.0 to 1.0
    - risk_level: LOW / MEDIUM / HIGH
    - explanation: plain English investigator summary (set explain=false to skip)
    - signals: the ML features used to make the decision
    """
    try:
        features = engineer_features(req)

        # Build model input — only the features the model knows about
        X = pd.DataFrame([features])[feature_names].fillna(0)
        prob = float(model.predict_proba(X)[0][1])

        risk = "HIGH" if prob >= 0.70 else ("MEDIUM" if prob >= 0.40 else "LOW")

        score_result = {
            "fraud_probability": round(prob, 3),
            "is_fraud":          prob >= 0.50,
            "risk_level":        risk,
        }

        # Build claim dict for explainer
        claim_dict = req.dict()
        claim_dict.update(features)

        # Get AI explanation
        explanation = None
        if req.explain:
            explanation = explain_claim(claim_dict, score_result)

        # Signals to return (human readable)
        signals = {
            "amount_zscore":        features["amount_zscore"],
            "provider_claim_count": features["provider_claim_count"],
            "patient_claim_count":  features["patient_claim_count"],
            "diag_proc_mismatch":   bool(features["diag_proc_mismatch"]),
            "high_units":           bool(features["high_units"]),
            "paid_ratio":           features["paid_ratio"],
            "procedure_avg_cost":   features["_proc_mean"],
        }

        return FraudScore(
            claim_id          = req.claim_id or "N/A",
            scored_at         = datetime.utcnow().isoformat(),
            fraud_probability = round(prob, 3),
            is_fraud          = prob >= 0.50,
            risk_level        = risk,
            explanation       = explanation,
            signals           = signals,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", tags=["scoring"])
def score_batch(claims: list[ClaimRequest]):
    """
    Score multiple claims at once. Returns a list of fraud scores.
    Maximum 100 claims per request.
    """
    if len(claims) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 claims per batch request")

    results = []
    for req in claims:
        try:
            result = score_claim(req)
            results.append(result)
        except Exception as e:
            results.append({"claim_id": req.claim_id, "error": str(e)})

    return {
        "total":    len(results),
        "high":     sum(1 for r in results if isinstance(r, FraudScore) and r.risk_level == "HIGH"),
        "medium":   sum(1 for r in results if isinstance(r, FraudScore) and r.risk_level == "MEDIUM"),
        "low":      sum(1 for r in results if isinstance(r, FraudScore) and r.risk_level == "LOW"),
        "results":  results,
    }


@app.get("/stats", tags=["data"])
def dataset_stats():
    """Return summary statistics about the claims dataset."""
    if df_claims.empty:
        return {"error": "claims.csv not found"}

    return {
        "total_claims":     int(len(df_claims)),
        "fraud_cases":      int(df_claims["is_fraud"].sum()),
        "fraud_rate":       round(float(df_claims["is_fraud"].mean()), 3),
        "total_billed":     round(float(df_claims["billed_amount"].sum()), 2),
        "fraud_amount":     round(float(df_claims[df_claims["is_fraud"]]["billed_amount"].sum()), 2),
        "fraud_by_type":    df_claims["fraud_type"].value_counts().dropna().to_dict(),
        "risk_breakdown":   {
            "HIGH":   int((df_claims["fraud_prob"] >= 0.70).sum()) if "fraud_prob" in df_claims else "run dashboard first",
            "MEDIUM": int(((df_claims["fraud_prob"] >= 0.40) & (df_claims["fraud_prob"] < 0.70)).sum()) if "fraud_prob" in df_claims else "run dashboard first",
            "LOW":    int((df_claims["fraud_prob"] < 0.40).sum()) if "fraud_prob" in df_claims else "run dashboard first",
        },
    }
