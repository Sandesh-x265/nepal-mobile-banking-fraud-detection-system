import pandas as pd

from generate_data import build_rule_based_data, load_config


def test_generate_data_schema_and_fraud_ratio():
    config = load_config()
    config["data"]["n_transactions"] = 200
    config["data"]["fraud_ratio"] = 0.12
    config["data"]["ctgan_augment"] = False

    df = build_rule_based_data(config)
    assert not df.empty
    assert set(["transaction_id", "customer_id", "amount_npr", "city", "merchant_category", "merchant_name", "is_fraud", "fraud_type"]).issubset(df.columns)
    fraud_fraction = df["is_fraud"].mean()
    assert 0.0 <= fraud_fraction <= 1.0
    assert df["amount_npr"].min() > 0
