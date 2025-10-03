# app/app.py
from __future__ import annotations

import os
import json
import math
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import joblib

# -----------------------------------------------------------------------------
# Paths (relative & portable)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
METRICS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "results", "metrics.json"))

# -----------------------------------------------------------------------------
# App & Model
# -----------------------------------------------------------------------------
app = Flask(__name__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Try to pull expected training columns from the fitted ColumnTransformer
# so we can construct a row with ALL columns (missing ones imputed by the pipeline).
try:
    preprocessor = model.named_steps["preprocessor"]
    NUMERIC_FEATURES = list(preprocessor.transformers_[0][2])
    CATEGORICAL_FEATURES = list(preprocessor.transformers_[1][2])
    EXPECTED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
except Exception:
    # Fallback: if the model doesn't expose, assume the classic Kaggle loan columns.
    EXPECTED_COLUMNS = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area",
        # engineered features we created in the notebook:
        "Income", "Loan_to_Income",
    ]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def coerce_float(x, default=np.nan):
    """Safely convert to float."""
    try:
        if x is None or str(x).strip() == "":
            return default
        return float(x)
    except Exception:
        return default

def map_property_area(code_or_text):
    """
    Accepts:
      - numeric codes from your current HTML (0,1,2),
      - or strings like 'Rural', 'Urban', 'Semiurban'.
    Returns the canonical string used during training.
    """
    if x := (None if code_or_text is None else str(code_or_text).strip()):
        # numeric mapping from your current form:
        if x in {"0", "1", "2"}:
            return {"0": "Rural", "1": "Urban", "2": "Semiurban"}[x]
        # free-text safe mapping
        s = x.lower()
        if "rural" in s:
            return "Rural"
        if "semi" in s:  # semiurban variations
            return "Semiurban"
        if "urban" in s:
            return "Urban"
    return None  # imputer will handle

def load_metrics_snippet():
    """Load a compact metrics summary if results/metrics.json exists."""
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                records = json.load(f)
            # Pick the first row (already sorted in the notebook) or best by F1
            if isinstance(records, list) and records:
                best = sorted(
                    records,
                    key=lambda r: (r.get("f1", 0.0), r.get("roc_auc", 0.0)),
                    reverse=True,
                )[0]
                return {
                    "best_model": best.get("model"),
                    "f1": round(best.get("f1", float("nan")), 4)
                    if not math.isnan(best.get("f1", float("nan"))) else None,
                    "roc_auc": round(best.get("roc_auc", float("nan")), 4)
                    if not math.isnan(best.get("roc_auc", float("nan"))) else None,
                    "accuracy": round(best.get("accuracy", float("nan")), 4)
                    if not math.isnan(best.get("accuracy", float("nan"))) else None,
                }
    except Exception:
        pass
    return None

def build_input_row(form) -> pd.DataFrame:
    """
    Build a single-row DataFrame containing ALL columns the pipeline expects.
    We fill columns not present in the HTML form with NaN (imputer handles them).
    - Uses your current minimal form fields:
        Credit_History (0/1)
        Property_Area (0=Rural, 1=Urban, 2=Semiurban)
        Income (numeric)
    - We also synthesize ApplicantIncome = Income and CoapplicantIncome = 0
      to satisfy common schemas.
    """
    # Read values from form
    credit_history = coerce_float(form.get("Credit_History"), default=np.nan)
    income = coerce_float(form.get("Income"), default=np.nan)
    property_area = map_property_area(form.get("Property_Area"))

    # Construct a dict with ALL expected columns set to NaN by default
    row = {col: np.nan for col in EXPECTED_COLUMNS}

    # Populate the fields we know
    if "Credit_History" in row:
        row["Credit_History"] = credit_history
    if "Property_Area" in row:
        row["Property_Area"] = property_area

    # Many classic versions have ApplicantIncome/CoapplicantIncome and our engineered features:
    if "Income" in row:
        row["Income"] = income
    if "ApplicantIncome" in row:
        row["ApplicantIncome"] = income
    if "CoapplicantIncome" in row:
        row["CoapplicantIncome"] = 0.0

    # Loan_to_Income if both present (else NaN; imputer handles)
    if "Loan_to_Income" in row and "LoanAmount" in row and "Income" in row:
        try:
            row["Loan_to_Income"] = (
                (row.get("LoanAmount") or np.nan) / (income if income not in (0, None, np.nan) else np.nan)
            )
        except Exception:
            row["Loan_to_Income"] = np.nan

    # Return as DataFrame
    return pd.DataFrame([row])

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    metrics = load_metrics_snippet()
    return render_template("index.html", prediction=None, proba=None, metrics=metrics)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Build a single-row DataFrame with all expected columns
        X = build_input_row(request.form)

        # Predict
        pred = model.predict(X)[0]
        # Try to get probability (if available)
        try:
            proba = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            proba = None

        # Human-friendly label
        label = "Loan Approved" if int(pred) == 1 else "Loan Rejected"

        # Load metrics snippet (optional)
        metrics = load_metrics_snippet()

        return render_template("index.html", prediction=label, proba=proba, metrics=metrics)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}", proba=None, metrics=None)

@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Use 0.0.0.0 for container platforms; change port if needed via env.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
