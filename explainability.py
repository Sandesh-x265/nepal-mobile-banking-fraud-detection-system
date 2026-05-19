import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

FEATURE_MAP = {
    "amount_npr": "transaction amount",
    "city": "transaction city",
    "merchant_category": "merchant category",
    "merchant_name": "merchant name",
    "device_type": "device type",
    "channel": "transaction channel",
    "is_new_merchant": "new merchant flag",
}


def load_model_artifacts():
    model = joblib.load(MODELS_DIR / "final_model.pkl")
    preprocessor = joblib.load(MODELS_DIR / "pipeline.pkl")
    with open(MODELS_DIR / "threshold.json", "r", encoding="utf-8") as fp:
        threshold = json.load(fp)["threshold"]
    return model, preprocessor, threshold


def explain_global(model, preprocessor, data_sample):
    explainer = shap.Explainer(model.named_steps["classifier"], preprocessor.transform(data_sample))
    shap_values = explainer(preprocessor.transform(data_sample))
    importance = np.abs(shap_values.values).mean(axis=0)
    feature_names = preprocessor.get_feature_names_out()
    ranking = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": float(v)} for f, v in ranking[:10]]


def explain_local(model, preprocessor, row):
    if isinstance(row, dict):
        row_df = pd.DataFrame([row])
    else:
        row_df = pd.DataFrame(row)
    explainer = shap.Explainer(model.named_steps["classifier"], preprocessor.transform(row_df))
    shap_values = explainer(preprocessor.transform(row_df))
    feature_names = preprocessor.get_feature_names_out()
    values = row_df.iloc[0].to_dict()
    contributions = []
    for feature, value, shap_value in zip(feature_names, row_df.iloc[0].tolist(), shap_values.values[0]):
        contributions.append(
            {"feature": feature, "value": str(value), "shap_value": float(shap_value)}
        )
    contributions = sorted(contributions, key=lambda item: abs(item["shap_value"]), reverse=True)
    return contributions[:8]


def business_explanation(shap_summary):
    reasons = []
    for item in shap_summary:
        feature = item["feature"].split("__")[-1]
        readable = FEATURE_MAP.get(feature, feature.replace("_", " "))
        direction = "increased" if item["shap_value"] > 0 else "decreased"
        reasons.append(f"{readable.capitalize()} {direction} risk by {abs(item['shap_value']):.3f}.")
    return " ".join(reasons)


def main():
    model, preprocessor, threshold = load_model_artifacts()
    sample_path = BASE_DIR / "data" / "transactions.csv"
    sample_df = pd.read_csv(sample_path, parse_dates=["timestamp"]).sample(200, random_state=42)
    global_explanation = explain_global(model, preprocessor, sample_df)
    print("Global SHAP feature importance:")
    print(json.dumps(global_explanation, indent=2))

    first_row = sample_df.iloc[0][["amount_npr", "city", "merchant_category", "merchant_name", "device_type", "channel", "is_new_merchant"]].to_dict()
    local = explain_local(model, preprocessor, first_row)
    print("\nLocal explanation:")
    print(json.dumps(local, indent=2))
    print("\nBusiness text:")
    print(business_explanation(local))


if __name__ == "__main__":
    main()
