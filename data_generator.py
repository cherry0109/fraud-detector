"""
Healthcare Claims Fraud Detection — Synthetic Data Generator
============================================================
Generates realistic but entirely fake insurance claims data.
About 10% of claims contain injected fraud patterns.

Usage:
    python data_generator.py                    # generates 1000 claims → claims.csv
    python data_generator.py --n 5000           # generates 5000 claims
    python data_generator.py --n 1000 --stream  # streams claims one by one (simulates live feed)

Install dependencies:
    pip install faker pandas numpy
"""

import argparse
import json
import random
import time
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ── REFERENCE DATA ────────────────────────────────────────────────────────────

# ICD-10 diagnosis codes (code → description)
DIAGNOSIS_CODES = {
    "E11.9":  "Type 2 diabetes without complications",
    "I10":    "Essential hypertension",
    "J18.9":  "Pneumonia, unspecified",
    "M54.5":  "Low back pain",
    "F32.9":  "Major depressive disorder, single episode",
    "Z00.00": "General adult medical exam",
    "K21.0":  "GERD with esophagitis",
    "J06.9":  "Acute upper respiratory infection",
    "N39.0":  "Urinary tract infection",
    "I25.10": "Atherosclerotic heart disease",
    "G43.909":"Migraine, unspecified",
    "M17.11": "Primary osteoarthritis, right knee",
    "R05":    "Cough",
    "R51":    "Headache",
    "Z23":    "Encounter for immunization",
}

# CPT procedure codes (code → description, base_cost)
PROCEDURE_CODES = {
    "99213": ("Office visit, established patient, low complexity", 150),
    "99214": ("Office visit, established patient, moderate complexity", 220),
    "99215": ("Office visit, established patient, high complexity", 310),
    "99203": ("Office visit, new patient, low complexity", 180),
    "99204": ("Office visit, new patient, moderate complexity", 260),
    "93000": ("Electrocardiogram, routine", 95),
    "71046": ("Chest X-ray, 2 views", 185),
    "80053": ("Comprehensive metabolic panel", 75),
    "85025": ("Complete blood count", 55),
    "90837": ("Psychotherapy, 60 min", 175),
    "27447": ("Total knee replacement", 12000),
    "43239": ("Upper GI endoscopy with biopsy", 2800),
    "29827": ("Arthroscopy, shoulder", 4500),
    "70553": ("MRI brain with contrast", 1800),
    "99232": ("Subsequent hospital care", 140),
}

# Insurance plan types
PLAN_TYPES = ["HMO", "PPO", "EPO", "HDHP", "Medicare Advantage", "Medicaid"]

# US States
STATES = ["TX", "CA", "FL", "NY", "IL", "PA", "OH", "GA", "NC", "MI"]

# Fraud pattern types (for labelling)
FRAUD_TYPES = [
    "upcoding",            # billing higher code than service rendered
    "duplicate_billing",   # same service billed multiple times
    "unbundling",          # splitting bundled services to bill more
    "phantom_billing",     # billing for services never rendered
    "provider_ring",       # coordinated fraud across provider network
]


# ── PROVIDER POOL ─────────────────────────────────────────────────────────────

def generate_providers(n=80):
    """Generate a pool of fake healthcare providers."""
    specialties = [
        "Internal Medicine", "Family Medicine", "Cardiology",
        "Orthopedics", "Psychiatry", "Gastroenterology",
        "Radiology", "Surgery", "Neurology", "Emergency Medicine",
    ]
    providers = []
    for _ in range(n):
        providers.append({
            "npi": str(random.randint(1000000000, 1999999999)),
            "name": f"Dr. {fake.last_name()}, {fake.first_name()}",
            "specialty": random.choice(specialties),
            "state": random.choice(STATES),
            "is_fraud_ring": False,  # flagged later
        })

    # Mark 5 providers as part of a fraud ring
    fraud_ring_indices = random.sample(range(n), 5)
    for i in fraud_ring_indices:
        providers[i]["is_fraud_ring"] = True

    return providers


# ── PATIENT POOL ──────────────────────────────────────────────────────────────

def generate_patients(n=300):
    """Generate a pool of fake patients."""
    patients = []
    for _ in range(n):
        dob = fake.date_of_birth(minimum_age=18, maximum_age=85)
        patients.append({
            "patient_id": str(uuid.uuid4())[:8].upper(),
            "name": fake.name(),
            "dob": dob.strftime("%Y-%m-%d"),
            "gender": random.choice(["M", "F"]),
            "state": random.choice(STATES),
            "plan_type": random.choice(PLAN_TYPES),
            "member_id": f"MBR{random.randint(100000, 999999)}",
        })
    return patients


# ── CLAIM GENERATORS ──────────────────────────────────────────────────────────

def make_legitimate_claim(patient, provider, claim_date):
    """Generate a normal, legitimate insurance claim."""
    diag_code = random.choice(list(DIAGNOSIS_CODES.keys()))
    proc_code = random.choice(list(PROCEDURE_CODES.keys()))
    proc_desc, base_cost = PROCEDURE_CODES[proc_code]

    # Realistic cost variation ±20%
    billed_amount = round(base_cost * random.uniform(0.85, 1.20), 2)
    allowed_amount = round(billed_amount * random.uniform(0.70, 0.90), 2)
    paid_amount = round(allowed_amount * random.uniform(0.75, 0.95), 2)

    return {
        "claim_id":        f"CLM{str(uuid.uuid4())[:10].upper()}",
        "claim_date":      claim_date.strftime("%Y-%m-%d"),
        "patient_id":      patient["patient_id"],
        "member_id":       patient["member_id"],
        "patient_name":    patient["name"],
        "patient_dob":     patient["dob"],
        "patient_gender":  patient["gender"],
        "patient_state":   patient["state"],
        "plan_type":       patient["plan_type"],
        "provider_npi":    provider["npi"],
        "provider_name":   provider["name"],
        "provider_specialty": provider["specialty"],
        "provider_state":  provider["state"],
        "diagnosis_code":  diag_code,
        "diagnosis_desc":  DIAGNOSIS_CODES[diag_code],
        "procedure_code":  proc_code,
        "procedure_desc":  proc_desc,
        "billed_amount":   billed_amount,
        "allowed_amount":  allowed_amount,
        "paid_amount":     paid_amount,
        "units_billed":    1,
        "place_of_service": random.choice(["11", "21", "22", "23"]),
        "is_fraud":        False,
        "fraud_type":      None,
        "fraud_notes":     None,
    }


def inject_upcoding(claim):
    """Fraud: bill a higher complexity code than service warranted."""
    claim["procedure_code"] = "99215"
    claim["procedure_desc"] = PROCEDURE_CODES["99215"][0]
    claim["billed_amount"] = round(PROCEDURE_CODES["99215"][1] * random.uniform(1.1, 1.4), 2)
    claim["allowed_amount"] = round(claim["billed_amount"] * 0.80, 2)
    claim["paid_amount"] = round(claim["allowed_amount"] * 0.85, 2)
    claim["is_fraud"] = True
    claim["fraud_type"] = "upcoding"
    claim["fraud_notes"] = "High complexity code billed for routine visit"
    return claim


def inject_duplicate_billing(claim):
    """Fraud: same service billed multiple times (high units)."""
    claim["units_billed"] = random.randint(4, 12)
    claim["billed_amount"] = round(claim["billed_amount"] * claim["units_billed"], 2)
    claim["allowed_amount"] = round(claim["billed_amount"] * 0.80, 2)
    claim["paid_amount"] = round(claim["allowed_amount"] * 0.85, 2)
    claim["is_fraud"] = True
    claim["fraud_type"] = "duplicate_billing"
    claim["fraud_notes"] = f"Service billed {claim['units_billed']} times in single encounter"
    return claim


def inject_phantom_billing(claim):
    """Fraud: service never rendered — abnormally high bill, no clinical context."""
    expensive_proc = random.choice(["27447", "43239", "29827", "70553"])
    proc_desc, base_cost = PROCEDURE_CODES[expensive_proc]
    claim["procedure_code"] = expensive_proc
    claim["procedure_desc"] = proc_desc
    claim["billed_amount"] = round(base_cost * random.uniform(1.3, 1.9), 2)
    claim["allowed_amount"] = round(claim["billed_amount"] * 0.75, 2)
    claim["paid_amount"] = round(claim["allowed_amount"] * 0.90, 2)
    claim["diagnosis_code"] = "Z00.00"  # general checkup — mismatched with surgery
    claim["diagnosis_desc"] = DIAGNOSIS_CODES["Z00.00"]
    claim["is_fraud"] = True
    claim["fraud_type"] = "phantom_billing"
    claim["fraud_notes"] = "High-cost procedure billed against routine checkup diagnosis"
    return claim


def inject_unbundling(claim):
    """Fraud: split a bundled procedure into separate line items to inflate payment."""
    claim["billed_amount"] = round(claim["billed_amount"] * random.uniform(2.2, 3.5), 2)
    claim["allowed_amount"] = round(claim["billed_amount"] * 0.78, 2)
    claim["paid_amount"] = round(claim["allowed_amount"] * 0.88, 2)
    claim["units_billed"] = random.randint(2, 5)
    claim["is_fraud"] = True
    claim["fraud_type"] = "unbundling"
    claim["fraud_notes"] = "Bundled procedure split across multiple line items"
    return claim


def inject_provider_ring(claim, ring_providers):
    """Fraud: coordinated billing — same patient across multiple fraud-ring providers."""
    ring_provider = random.choice(ring_providers)
    claim["provider_npi"] = ring_provider["npi"]
    claim["provider_name"] = ring_provider["name"]
    claim["provider_specialty"] = ring_provider["specialty"]
    claim["billed_amount"] = round(claim["billed_amount"] * random.uniform(1.5, 2.5), 2)
    claim["allowed_amount"] = round(claim["billed_amount"] * 0.80, 2)
    claim["paid_amount"] = round(claim["allowed_amount"] * 0.85, 2)
    claim["is_fraud"] = True
    claim["fraud_type"] = "provider_ring"
    claim["fraud_notes"] = "Provider part of coordinated billing ring — patient seen across multiple ring members"
    return claim


# ── DERIVED FEATURES ──────────────────────────────────────────────────────────

def add_features(df):
    """
    Add engineered features that the ML model will use.
    These mirror what you'd compute in your Databricks streaming job.
    """
    # Claim amount z-score per procedure code
    df["proc_mean"] = df.groupby("procedure_code")["billed_amount"].transform("mean")
    df["proc_std"]  = df.groupby("procedure_code")["billed_amount"].transform("std").fillna(1)
    df["amount_zscore"] = ((df["billed_amount"] - df["proc_mean"]) / df["proc_std"]).round(3)

    # Claims per provider (volume signal)
    df["provider_claim_count"] = df.groupby("provider_npi")["claim_id"].transform("count")

    # Claims per patient (utilisation signal)
    df["patient_claim_count"] = df.groupby("patient_id")["claim_id"].transform("count")

    # Diagnosis-procedure mismatch flag (simple rule)
    # Surgical procedures billed against Z00.00 (routine checkup) = suspicious
    surgical = {"27447", "43239", "29827"}
    df["diag_proc_mismatch"] = (
        (df["procedure_code"].isin(surgical)) &
        (df["diagnosis_code"] == "Z00.00")
    ).astype(int)

    # High units flag
    df["high_units"] = (df["units_billed"] > 3).astype(int)

    # Paid ratio (paid / billed — unusually high suggests manipulation)
    df["paid_ratio"] = (df["paid_amount"] / df["billed_amount"]).round(3)

    # Drop helper columns
    df.drop(columns=["proc_mean", "proc_std"], inplace=True)

    return df


# ── MAIN GENERATOR ────────────────────────────────────────────────────────────

def generate_claims(n=1000, fraud_rate=0.10):
    """
    Generate n claims with ~fraud_rate proportion being fraudulent.
    Returns a pandas DataFrame.
    """
    providers = generate_providers(80)
    patients  = generate_patients(300)
    ring_providers = [p for p in providers if p["is_fraud_ring"]]

    fraud_injectors = [
        inject_upcoding,
        inject_duplicate_billing,
        inject_phantom_billing,
        inject_unbundling,
    ]

    claims = []
    start_date = datetime.now() - timedelta(days=180)

    print(f"Generating {n} claims ({int(n * fraud_rate)} fraudulent)...")

    for i in range(n):
        patient  = random.choice(patients)
        provider = random.choice(providers)
        claim_date = start_date + timedelta(days=random.randint(0, 180),
                                            hours=random.randint(6, 22),
                                            minutes=random.randint(0, 59))

        claim = make_legitimate_claim(patient, provider, claim_date)

        # Inject fraud
        if random.random() < fraud_rate:
            fraud_choice = random.random()
            if fraud_choice < 0.30:
                claim = inject_upcoding(claim)
            elif fraud_choice < 0.55:
                claim = inject_duplicate_billing(claim)
            elif fraud_choice < 0.70:
                claim = inject_phantom_billing(claim)
            elif fraud_choice < 0.85:
                claim = inject_unbundling(claim)
            else:
                claim = inject_provider_ring(claim, ring_providers)

        claims.append(claim)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n} claims generated...")

    df = pd.DataFrame(claims)
    df = add_features(df)

    return df


def stream_claims(n=100, delay_seconds=0.5):
    """
    Simulate a live claims feed — yields one claim at a time.
    Use this to test your Event Hubs producer.
    """
    providers = generate_providers(80)
    patients  = generate_patients(300)
    ring_providers = [p for p in providers if p["is_fraud_ring"]]

    print(f"Streaming {n} claims with {delay_seconds}s delay between each...")

    for i in range(n):
        patient    = random.choice(patients)
        provider   = random.choice(providers)
        claim_date = datetime.now()
        claim      = make_legitimate_claim(patient, provider, claim_date)

        if random.random() < 0.10:
            injector = random.choice([
                inject_upcoding, inject_duplicate_billing,
                inject_phantom_billing, inject_unbundling,
            ])
            claim = injector(claim)

        print(f"[{i+1:04d}] {claim['claim_id']} | "
              f"${claim['billed_amount']:>9.2f} | "
              f"{'🚨 FRAUD' if claim['is_fraud'] else '✓ legit':10s} | "
              f"{claim['fraud_type'] or ''}")

        yield claim
        time.sleep(delay_seconds)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Healthcare Claims Data Generator")
    parser.add_argument("--n",      type=int,   default=1000,  help="Number of claims to generate")
    parser.add_argument("--fraud",  type=float, default=0.10,  help="Fraud rate (0.0–1.0)")
    parser.add_argument("--out",    type=str,   default="claims.csv", help="Output CSV filename")
    parser.add_argument("--stream", action="store_true",       help="Stream mode (simulates live feed)")
    parser.add_argument("--delay",  type=float, default=0.3,   help="Delay between streamed claims (seconds)")
    args = parser.parse_args()

    if args.stream:
        all_claims = []
        for claim in stream_claims(n=args.n, delay_seconds=args.delay):
            all_claims.append(claim)
        df = pd.DataFrame(all_claims)
    else:
        df = generate_claims(n=args.n, fraud_rate=args.fraud)

    # Save to CSV
    df.to_csv(args.out, index=False)

    # Print summary
    fraud_count = df["is_fraud"].sum()
    print(f"\n{'─'*55}")
    print(f"  Claims generated : {len(df):,}")
    print(f"  Fraudulent       : {fraud_count:,} ({fraud_count/len(df)*100:.1f}%)")
    print(f"  Total billed     : ${df['billed_amount'].sum():,.2f}")
    print(f"  Fraud amount     : ${df[df['is_fraud']]['billed_amount'].sum():,.2f}")
    print(f"  Fraud types      : {df[df['is_fraud']]['fraud_type'].value_counts().to_dict()}")
    print(f"  Output saved to  : {args.out}")
    print(f"{'─'*55}\n")

    # Preview
    print("Sample fraudulent claims:")
    fraud_cols = ["claim_id", "billed_amount", "fraud_type", "fraud_notes"]
    print(df[df["is_fraud"]][fraud_cols].head(5).to_string(index=False))

    return df


if __name__ == "__main__":
    main()
