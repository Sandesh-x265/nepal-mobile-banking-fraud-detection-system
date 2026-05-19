from fastapi.testclient import TestClient

import fraud_api.app as app_mod


class DummyClassifier:
    def predict_proba(self, X):
        return [[0.15, 0.85]]


class DummyPreprocessor:
    def transform(self, X):
        return [[0.0]]

    def get_feature_names_out(self):
        return ["amount_npr"]


def fake_load_artifacts():
    app_mod.MODEL = DummyClassifier()
    app_mod.PREPROCESSOR = DummyPreprocessor()
    app_mod.THRESHOLD = 0.5
    app_mod.METADATA = {"source": "test"}


def fake_shap_summary(features):
    return [{"feature": "amount_npr", "value": "100.0", "shap_value": 0.42}]


def test_health_endpoint():
    client = TestClient(app_mod.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(app_mod, "load_artifacts", fake_load_artifacts)
    monkeypatch.setattr(app_mod, "build_shap_summary", fake_shap_summary)
    client = TestClient(app_mod.app)

    payload = {
        "customer_id": "CUS00001",
        "amount_npr": 100.0,
        "city": "Kathmandu",
        "merchant_category": "grocery",
        "merchant_name": "FreshMart",
        "device_type": "android",
        "channel": "app",
        "is_new_merchant": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["probability"] == 0.85
    assert data["risk_label"] == "High"
    assert data["threshold"] == 0.5
    assert data["model_metadata"]["source"] == "test"
