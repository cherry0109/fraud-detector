"""
Healthcare Claims Fraud Detection — ML Model
=============================================
Trains an XGBoost classifier on synthetic claims data,
evaluates performance, and saves the model for use in
the dashboard and API.

Usage:
    python fraud_model.py

Requirements:
    claims.csv must exist in the same folder (run data_generator.py first)
"""

import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_PATH  = "claims.csv"
MODEL_PATH = "fraud_model.pkl"

FEATURES = [
    "amount_zscore",
    "provider_claim_count",
    "patient_claim_count",
    "diag_proc_mismatch",
    "high_units",
    "paid_ratio",
    "billed_amount",
    "allowed_amount",
    "units_billed",
]

TARGET = "is_fraud"


# ── LOAD DATA ─────────────────────────────────────────────────────────────────

def load_data(path):
    print(f"\nLoading data from {path}...")
    df = pd.read_csv(path)
    print(f"  {len(df):,} claims loaded")
    print(f"  {df[TARGET].sum():,} fraudulent ({df[TARGET].mean()*100:.1f}%)")
    return df


# ── PREPARE FEATURES ──────────────────────────────────────────────────────────

def prepare_features(df):
    """
    Select and clean features for model training.
    All features are numeric — no encoding needed.
    """
    available = [f for f in FEATURES if f in df.columns]
    missing   = [f for f in FEATURES if f not in df.columns]

    if missing:
        print(f"  Warning: missing features {missing} — skipping them")

    X = df[available].copy()
    y = df[TARGET].astype(int).copy()

    # Fill any NaN values with column median
    X = X.fillna(X.median())

    print(f"  Features used: {available}")
    return X, y


# ── TRAIN MODEL ───────────────────────────────────────────────────────────────

def train_model(X_train, y_train):
    """
    Train XGBoost classifier.
    scale_pos_weight handles class imbalance —
    set to ratio of negative to positive examples.
    """
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale = round(neg / pos, 2)

    print(f"\nTraining XGBoost model...")
    print(f"  Training samples : {len(X_train):,}")
    print(f"  Fraud cases      : {pos:,}")
    print(f"  scale_pos_weight : {scale} (handles class imbalance)")

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )

    model.fit(X_train, y_train)
    print("  Model trained successfully")
    return model


# ── EVALUATE ──────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, feature_names):
    """
    Print a clean evaluation summary and plot confusion matrix
    plus feature importance.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_proba)

    print("\n" + "═" * 55)
    print("  MODEL EVALUATION RESULTS")
    print("═" * 55)
    print(f"  Test claims      : {len(y_test):,}")
    print(f"  Fraud in test    : {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
    print(f"  Precision        : {precision:.3f}  (of flagged claims, how many are real fraud)")
    print(f"  Recall           : {recall:.3f}  (of all fraud, how many did we catch)")
    print(f"  F1 Score         : {f1:.3f}  (balance of precision and recall)")
    print(f"  ROC AUC          : {auc:.3f}  (overall discrimination ability)")
    print("═" * 55)

    # Feature importance
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    print("\n  TOP FEATURES (by importance to the model):")
    for feat, score in importance.head(5).items():
        bar = "█" * int(score * 40)
        print(f"  {feat:<28} {bar} {score:.3f}")

    print("═" * 55)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion matrix:")
    print(f"  True negatives  (correctly cleared) : {tn:,}")
    print(f"  False positives (wrongly flagged)   : {fp:,}")
    print(f"  False negatives (fraud we missed)   : {fn:,}")
    print(f"  True positives  (fraud caught)      : {tp:,}")
    print("═" * 55)

    # Save confusion matrix plot
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Confusion matrix heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=axes[0],
        )
        axes[0].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Actual")
        axes[0].set_xlabel("Predicted")

        # Feature importance bar chart
        importance.head(8).plot(
            kind="barh",
            ax=axes[1],
            color="#4a90e2",
        )
        axes[1].set_title("Feature Importance", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Importance Score")
        axes[1].invert_yaxis()

        plt.tight_layout()
        plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
        print("\n  Chart saved to model_evaluation.png")
    except Exception as e:
        print(f"\n  (Chart skipped: {e})")

    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
        "auc":       round(auc, 3),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


# ── SAVE MODEL ────────────────────────────────────────────────────────────────

def save_model(model, feature_names, metrics, path):
    """
    Save the trained model plus metadata so the dashboard
    and API can load it without retraining.
    """
    payload = {
        "model":         model,
        "feature_names": feature_names,
        "metrics":       metrics,
        "trained_on":    pd.Timestamp.now().isoformat(),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Model saved to {path}")
    print("  (Load it later with: pickle.load(open('fraud_model.pkl', 'rb')))")


# ── SCORE A SINGLE CLAIM ──────────────────────────────────────────────────────

def score_claim(claim_dict, model_path=MODEL_PATH):
    """
    Score a single claim dict and return fraud probability.
    Used by the dashboard and API.

    Example:
        result = score_claim({
            "amount_zscore": 3.2,
            "provider_claim_count": 45,
            "patient_claim_count": 2,
            "diag_proc_mismatch": 1,
            "high_units": 0,
            "paid_ratio": 0.95,
            "billed_amount": 12500,
            "allowed_amount": 9800,
            "units_billed": 1,
        })
        print(result)
        # {"fraud_probability": 0.94, "is_fraud": True, "risk_level": "HIGH"}
    """
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    model         = payload["model"]
    feature_names = payload["feature_names"]

    X = pd.DataFrame([claim_dict])[feature_names].fillna(0)
    prob = float(model.predict_proba(X)[0][1])

    if prob >= 0.70:
        risk = "HIGH"
    elif prob >= 0.40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "fraud_probability": round(prob, 3),
        "is_fraud":          prob >= 0.50,
        "risk_level":        risk,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═" * 55)
    print("  HEALTHCARE CLAIMS FRAUD DETECTOR — ML TRAINING")
    print("═" * 55)

    # Load
    df = load_data(DATA_PATH)

    # Prepare
    X, y = prepare_features(df)
    feature_names = list(X.columns)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, feature_names)

    # Save
    save_model(model, feature_names, metrics, MODEL_PATH)

    print("\n  All done. Next step: run python ai_explainer.py")
    print("═" * 55 + "\n")


if __name__ == "__main__":
    main()
