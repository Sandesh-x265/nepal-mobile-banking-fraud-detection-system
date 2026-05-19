import inspect
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "transactions.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_config():
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def evaluate_threshold(y_true, y_scores, precision_target=0.75):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    candidates = [(p, r, t) for p, r, t in zip(precision[:-1], recall[:-1], thresholds) if p >= precision_target]
    if candidates:
        best = max(candidates, key=lambda item: item[1])
        return best[2], best[0], best[1]
    best_f1_idx = np.argmax(2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9))
    return thresholds[best_f1_idx], precision[best_f1_idx], recall[best_f1_idx]


def score_model(model, X_test, y_test, threshold):
    y_scores = model.predict_proba(X_test)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    return {
        "average_precision": average_precision_score(y_test, y_scores),
        "recall": recall_score(y_test, y_pred),
        "precision": (y_test.sum() and (y_test & y_pred).sum() / y_pred.sum()) or 0.0,
    }


def build_preprocessor(df):
    numeric_features = ["amount_npr", "is_new_merchant"]
    categorical_features = ["city", "merchant_category", "merchant_name", "device_type", "channel"]
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    ohe_kwargs = {"handle_unknown": "ignore"}
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False

    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(**ohe_kwargs))]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ], remainder="drop"
    )


def build_models(config):
    return {
        "logistic_regression": LogisticRegression(random_state=config["model"]["random_seed"], max_iter=600, class_weight="balanced"),
        "random_forest": RandomForestClassifier(random_state=config["model"]["random_seed"], n_estimators=180, class_weight="balanced"),
        "xgboost": XGBClassifier(
            random_state=config["model"]["random_seed"],
            n_estimators=config["model"]["classifier"]["xgboost"]["n_estimators"],
            max_depth=config["model"]["classifier"]["xgboost"]["max_depth"],
            learning_rate=config["model"]["classifier"]["xgboost"]["learning_rate"],
            subsample=config["model"]["classifier"]["xgboost"]["subsample"],
            colsample_bytree=config["model"]["classifier"]["xgboost"]["colsample_bytree"],
            use_label_encoder=False,
            eval_metric="logloss",
        ),
    }


def resample(X, y, method, rng):
    if method == "smote":
        return SMOTE(random_state=rng).fit_resample(X, y)
    if method == "smote_tomek":
        return SMOTETomek(random_state=rng).fit_resample(X, y)
    return X, y


def prepare_dataset(df):
    df = df.copy()
    df = df.dropna(subset=["amount_npr", "city", "merchant_category", "merchant_name", "device_type", "channel"])
    X = df[["amount_npr", "city", "merchant_category", "merchant_name", "device_type", "channel", "is_new_merchant"]]
    y = df["is_fraud"].astype(int)
    return X, y


def main():
    config = load_config()
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    X, y = prepare_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_seed"],
        stratify=y,
    )

    preprocessor = build_preprocessor(df)
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    models = build_models(config)

    mlflow.set_experiment("nepal_mobile_fraud_detection")
    best_classifier = None
    best_score = -1
    best_artifacts = {}

    for model_name, classifier in models.items():
        for strategy in config["model"]["imbalanced_methods"]:
            with mlflow.start_run(run_name=f"{model_name}_{strategy}"):
                X_res, y_res = resample(X_train_transformed, y_train, strategy, config["model"]["random_seed"])
                classifier_clone = clone(classifier)
                classifier_clone.fit(X_res, y_res)
                threshold, precision_val, recall_val = evaluate_threshold(
                    y_train,
                    classifier_clone.predict_proba(X_train_transformed)[:, 1],
                    precision_target=config["model"]["precision_target"],
                )
                test_metrics = score_model(classifier_clone, X_test_transformed, y_test, threshold)

                mlflow.log_params({
                    "model": model_name,
                    "strategy": strategy,
                    "threshold": threshold,
                    "precision_target": config["model"]["precision_target"],
                })
                mlflow.log_metrics({
                    "train_precision": precision_val,
                    "train_recall": recall_val,
                    "test_average_precision": test_metrics["average_precision"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                })

                if test_metrics["average_precision"] > best_score:
                    best_score = test_metrics["average_precision"]
                    best_classifier = classifier_clone
                    best_artifacts = {
                        "model_name": model_name,
                        "strategy": strategy,
                        "threshold": threshold,
                        "metrics": test_metrics,
                    }

    if best_classifier is None:
        raise RuntimeError("No model was trained successfully.")

    best_run = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", best_classifier)])
    dump(best_run, MODELS_DIR / "final_model.pkl")
    dump(preprocessor, MODELS_DIR / "pipeline.pkl")
    with open(MODELS_DIR / "threshold.json", "w", encoding="utf-8") as fp:
        json.dump({"threshold": float(best_artifacts["threshold"])}, fp, indent=2)
    with open(MODELS_DIR / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump({
            "best_model": best_artifacts["model_name"],
            "strategy": best_artifacts["strategy"],
            "metrics": best_artifacts["metrics"],
        }, fp, indent=2)

    print("Training complete. Best model saved to models/final_model.pkl")


if __name__ == "__main__":
    main()
