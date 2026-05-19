import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI

from fraud_api.schemas import BatchPredictionResponse, BatchTransactionRequest, PredictionResponse, TransactionRequest

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CONFIG_PATH = BASE_DIR / "config.yaml"

app = FastAPI(title="Nepal Mobile Wallet Fraud API", version="0.1.0")

MODEL = None
PREPROCESSOR = None
THRESHOLD = None
METADATA = {}
BACKGROUND_SAMPLE = None


def load_artifacts():
    global MODEL, PREPROCESSOR, THRESHOLD, METADATA, BACKGROUND_SAMPLE
    if MODEL is not None:
        return

    try:
        MODEL = joblib.load(MODELS_DIR / "final_model.pkl")
        PREPROCESSOR = joblib.load(MODELS_DIR / "pipeline.pkl")
        with open(MODELS_DIR / "threshold.json", "r", encoding="utf-8") as fp:
            THRESHOLD = json.load(fp)["threshold"]
        with open(MODELS_DIR / "metadata.json", "r", encoding="utf-8") as fp:
            METADATA = json.load(fp)
    except FileNotFoundError as exc:
        raise RuntimeError("Model artifacts are not available. Run train_pipeline.py first.") from exc

    try:
        sample_data = pd.read_csv(BASE_DIR / "data" / "transactions.csv")
        BACKGROUND_SAMPLE = PREPROCESSOR.transform(sample_data.sample(100, random_state=42)[[
            "amount_npr",
            "city",
            "merchant_category",
            "merchant_name",
            "device_type",
            "channel",
            "is_new_merchant",
        ]])
    except Exception:
        BACKGROUND_SAMPLE = None


def build_shap_summary(features: pd.DataFrame):
    if MODEL is None or PREPROCESSOR is None:
        raise RuntimeError("Model artifacts are not loaded.")
    transformed = PREPROCESSOR.transform(features)
    model = MODEL.named_steps["classifier"]
    feature_names = PREPROCESSOR.get_feature_names_out()

    try:
        if BACKGROUND_SAMPLE is not None:
            background = BACKGROUND_SAMPLE[:10]
        else:
            background = transformed

        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer(transformed)
    except Exception:
        return []

    values = shap_values.values
    if isinstance(values, np.ndarray) and values.ndim == 3:
        if values.shape[1] == 2:
            shap_values_array = values[0, 1]
        else:
            shap_values_array = values[0, -1]
    else:
        shap_values_array = values[0]

    contributions = []
    for name, value, shap_value in zip(feature_names, transformed[0].tolist(), shap_values_array):
        contributions.append({
            "feature": name,
            "value": str(value),
            "shap_value": float(shap_value),
        })
    contributions = sorted(contributions, key=lambda item: abs(item["shap_value"]), reverse=True)
    return contributions[:6]


def business_reason(shap_summary):
    reasons = []
    for item in shap_summary:
        readable = item["feature"].replace("__", " ").replace("merchant_name", "merchant").replace("amount npr", "amount")
        polarity = "increased" if item["shap_value"] > 0 else "reduced"
        reasons.append(f"{readable.capitalize()} {polarity} fraud risk.")
    return " ".join(reasons)


def classify_risk(probability: float):
    if probability >= 0.8:
        return "High"
    if probability >= 0.45:
        return "Medium"
    return "Low"


def predict_transaction(features: dict):
    if MODEL is None or PREPROCESSOR is None:
        load_artifacts()

    row_df = pd.DataFrame([features])
    probabilities = MODEL.predict_proba(row_df)
    proba_array = np.asarray(probabilities)
    if proba_array.ndim == 1:
        probability = float(proba_array[1])
    else:
        probability = float(proba_array[0, 1])

    shap_summary = build_shap_summary(row_df)
    return {
        "probability": round(probability, 5),
        "risk_label": classify_risk(probability),
        "threshold": THRESHOLD,
        "shap_summary": shap_summary,
        "business_reason": business_reason(shap_summary),
        "model_metadata": METADATA,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: TransactionRequest):
    prediction = predict_transaction(request.dict())
    return prediction


@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchTransactionRequest):
    predictions: List[PredictionResponse] = []
    for item in request.transactions:
        predictions.append(PredictionResponse(**predict_transaction(item.dict())))
    return BatchPredictionResponse(predictions=predictions)
