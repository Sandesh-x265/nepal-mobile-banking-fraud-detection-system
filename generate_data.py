import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CTGAN_MODEL = None
CTGAN_MODEL_NAME = None
try:
    from ctgan import CTGANSynthesizer as _CTGANModel
    CTGAN_MODEL = _CTGANModel
    CTGAN_MODEL_NAME = "CTGANSynthesizer"
except ImportError:
    try:
        from ctgan.synthesizers.ctgan import CTGANSynthesizer as _CTGANModel
        CTGAN_MODEL = _CTGANModel
        CTGAN_MODEL_NAME = "CTGANSynthesizer"
    except ImportError:
        try:
            from ctgan import CTGAN as _CTGANModel
            CTGAN_MODEL = _CTGANModel
            CTGAN_MODEL_NAME = "CTGAN"
        except ImportError:
            CTGAN_MODEL = None
            CTGAN_MODEL_NAME = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

NEPALI_CITIES = [
    "Kathmandu", "Lalitpur", "Bhaktapur", "Pokhara", "Biratnagar",
    "Chitwan", "Birgunj", "Dhangadhi", "Butwal", "Bharatpur"
]
MERCHANT_CATEGORIES = [
    "grocery", "fuel", "utilities", "food_delivery", "pharmacy",
    "mobile_recharge", "education", "travel", "entertainment", "apparel"
]
MERCHANTS = {
    "grocery": ["BhatBhandar", "FreshMart", "Nirvana Grocers"],
    "fuel": ["Ncell Station", "Reliance Fuel", "Himal Fuel"],
    "utilities": ["NEA BillPay", "KTM Water", "Smart Meter"],
    "food_delivery": ["Foodmandu", "SastoFood", "KhanaKhajana"],
    "pharmacy": ["HealthPlus", "MediCare", "NepalPharma"],
    "mobile_recharge": ["NTC Recharge", "Ncell TopUp", "SmartSIM"],
    "education": ["EduPay", "Campus Fees", "Online Tuition"],
    "travel": ["Yatayat", "Nepal Airlines", "TourVista"],
    "entertainment": ["Cineplex", "LiveMusic", "GameZone"],
    "apparel": ["StyleSquare", "TrendStreet", "FashionHub"],
}
DEVICE_TYPES = ["android", "ios", "web"]
CHANNELS = ["app", "ussd", "web"]

FRAUD_REASONS = ["velocity", "geo_anomaly", "amount_anomaly", "merchant_inconsistency"]


def load_config():
    config_path = BASE_DIR / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def sample_customer_profiles(n_customers, rng):
    profiles = []
    for customer_id in range(1, n_customers + 1):
        home_city = rng.choice(NEPALI_CITIES)
        preferred_category = rng.choice(MERCHANT_CATEGORIES)
        profiles.append(
            {
                "customer_id": f"CUS{customer_id:05d}",
                "home_city": home_city,
                "preferred_category": preferred_category,
                "average_amount": float(rng.normal(1500, 600)),
                "velocity_baseline": rng.integers(1, 4),
            }
        )
    return pd.DataFrame(profiles)


def pick_merchant(category, rng):
    return rng.choice(MERCHANTS[category])


def generate_transaction(customer, transaction_id, current_time, rng):
    city = customer["home_city"] if rng.random() > 0.1 else rng.choice(NEPALI_CITIES)
    category = customer["preferred_category"] if rng.random() > 0.2 else rng.choice(MERCHANT_CATEGORIES)
    merchant = pick_merchant(category, rng)
    amount = max(25, float(rng.normal(customer["average_amount"], 800)))
    if category == "fuel":
        amount = max(50, float(rng.normal(2500, 900)))
    if category == "education":
        amount = max(100, float(rng.normal(4200, 2100)))
    return {
        "transaction_id": f"TXN{transaction_id:07d}",
        "customer_id": customer["customer_id"],
        "timestamp": current_time,
        "city": city,
        "merchant_category": category,
        "merchant_name": merchant,
        "amount_npr": round(amount, 2),
        "device_type": rng.choice(DEVICE_TYPES),
        "channel": rng.choice(CHANNELS),
        "previous_city": customer["home_city"],
        "is_new_merchant": int(rng.random() < 0.12),
    }


def inject_fraud(transaction, customer, rng):
    fraud_type = rng.choice(FRAUD_REASONS)
    transaction["fraud_type"] = fraud_type
    transaction["is_fraud"] = 1
    if fraud_type == "velocity":
        transaction["amount_npr"] *= 0.8 + rng.random() * 0.5
        transaction["channel"] = "app"
        transaction["device_type"] = rng.choice(["android", "ios"])
        transaction["merchant_category"] = customer["preferred_category"]
    elif fraud_type == "geo_anomaly":
        transaction["city"] = rng.choice([c for c in NEPALI_CITIES if c != customer["home_city"]])
        transaction["is_new_merchant"] = 1
    elif fraud_type == "amount_anomaly":
        transaction["amount_npr"] *= 3 + rng.random() * 2
        transaction["merchant_category"] = rng.choice(["travel", "electronics", "education"])
    elif fraud_type == "merchant_inconsistency":
        transaction["merchant_category"] = rng.choice([c for c in MERCHANT_CATEGORIES if c != customer["preferred_category"]])
        transaction["is_new_merchant"] = 1
    return transaction


def build_rule_based_data(config):
    rng = np.random.default_rng(config["data"]["random_seed"])
    customers = sample_customer_profiles(config["data"]["n_customers"], rng)
    start_dt = datetime.fromisoformat(config["data"]["transaction_start"])
    end_dt = datetime.fromisoformat(config["data"]["transaction_end"])
    n_transactions = config["data"]["n_transactions"]
    fraud_target = int(n_transactions * config["data"]["fraud_ratio"])

    records = []
    fraud_count = 0
    for transaction_id in range(1, n_transactions + 1):
        customer = customers.sample(n=1, random_state=int(rng.integers(1_000_000))).iloc[0]
        interval = timedelta(minutes=int(rng.integers(3, 240)))
        current_time = start_dt + timedelta(seconds=int(rng.random() * (end_dt - start_dt).total_seconds()))
        record = generate_transaction(customer, transaction_id, current_time, rng)
        if fraud_count < fraud_target and rng.random() < 0.16:
            record = inject_fraud(record, customer, rng)
            fraud_count += 1
        else:
            record["fraud_type"] = "legitimate"
            record["is_fraud"] = 0
        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def augment_with_ctgan(df, config):
    if CTGAN_MODEL is None:
        print("CTGAN not installed or unsupported version; skipping augmentation.")
        return df

    numeric_columns = ["amount_npr", "is_new_merchant"]
    categorical_columns = ["city", "merchant_category", "merchant_name", "device_type", "channel", "fraud_type"]
    synth_columns = ["customer_id", "timestamp"] + numeric_columns + categorical_columns
    nonfraud_df = df[df["is_fraud"] == 0].copy()
    if nonfraud_df.empty:
        return df

    synthesizer = CTGAN_MODEL(
        epochs=config["data"].get("ctgan_epochs", 40),
        batch_size=config["data"].get("ctgan_batch_size", 128),
        verbose=False,
    )

    try:
        if CTGAN_MODEL_NAME == "CTGANSynthesizer":
            synthesizer.fit(nonfraud_df[synth_columns], categorical_columns)
        else:
            synthesizer.fit(nonfraud_df[synth_columns], discrete_columns=categorical_columns)

        generated = synthesizer.sample(config["data"]["ctgan_samples"])
        generated["fraud_type"] = "legitimate"
        generated["is_fraud"] = 0
        generated["transaction_id"] = [f"SYN{idx:07d}" for idx in range(1, len(generated) + 1)]
        generated["timestamp"] = pd.to_datetime(generated["timestamp"], errors="coerce")
        generated = generated.dropna(subset=["timestamp"])
        return pd.concat([df, generated], ignore_index=True).reset_index(drop=True)
    except Exception as exc:
        print(f"CTGAN augmentation failed: {exc}")
        return df


def save_dataset(df):
    path = DATA_DIR / "transactions.csv"
    df.to_csv(path, index=False)
    print(f"Saved synthetic data to {path}")


def main():
    config = load_config()
    df = build_rule_based_data(config)
    if config["data"].get("ctgan_augment", False):
        df = augment_with_ctgan(df, config)
    save_dataset(df)


if __name__ == "__main__":
    main()
