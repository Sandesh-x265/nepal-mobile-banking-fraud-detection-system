import os
import requests
import streamlit as st
import pandas as pd

API_URL = os.environ.get("API_URL", "http://localhost:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"

FEATURES = {
    "customer_id": "CUS00001",
    "amount_npr": 1200.0,
    "city": "Kathmandu",
    "merchant_category": "grocery",
    "merchant_name": "FreshMart",
    "device_type": "android",
    "channel": "app",
    "is_new_merchant": 0,
}

st.set_page_config(page_title="Nepal Mobile Wallet Fraud Dashboard", layout="wide")
st.title("Nepal Mobile Wallet Fraud Detection")
st.write("Use the form to submit a synthetic transaction and inspect the predicted fraud risk.")

with st.form("transaction_form"):
    st.subheader("Transaction input")
    amount_npr = st.number_input("Amount (NPR)", min_value=1.0, value=1200.0, step=50.0)
    city = st.selectbox("City", ["Kathmandu", "Lalitpur", "Pokhara", "Biratnagar", "Chitwan"])
    merchant_category = st.selectbox("Merchant category", ["grocery", "fuel", "utilities", "food_delivery", "pharmacy", "mobile_recharge", "education", "travel", "entertainment", "apparel"])
    merchant_name = st.text_input("Merchant name", "FreshMart")
    device_type = st.selectbox("Device type", ["android", "ios", "web"])
    channel = st.selectbox("Channel", ["app", "ussd", "web"])
    is_new_merchant = st.radio("Is new merchant?", [0, 1], index=0)
    submit = st.form_submit_button("Predict fraud risk")

if submit:
    payload = {
        "customer_id": "CUS00001",
        "amount_npr": amount_npr,
        "city": city,
        "merchant_category": merchant_category,
        "merchant_name": merchant_name,
        "device_type": device_type,
        "channel": channel,
        "is_new_merchant": is_new_merchant,
    }
    
    with st.spinner("Sending prediction request to API..."):
        try:
            response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=15)
            response.raise_for_status()
            result = response.json()
            
            st.success("✅ Prediction received!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud Probability", f"{result['probability']:.4f}")
            with col2:
                risk_color = "🔴" if result['risk_label'] == "High" else "🟡" if result['risk_label'] == "Medium" else "🟢"
                st.metric("Risk Label", f"{risk_color} {result['risk_label']}")
            with col3:
                st.metric("Threshold", f"{result['threshold']:.4f}")
            
            st.subheader("Business Explanation")
            if result.get("business_reason"):
                st.info(result["business_reason"])
            else:
                st.info("No business explanation available for this prediction.")
            
            st.subheader("SHAP Feature Contributions")
            if result.get("shap_summary") and len(result["shap_summary"]) > 0:
                shap_df = pd.DataFrame(result["shap_summary"])
                st.dataframe(shap_df, use_container_width=True)
            else:
                st.info("SHAP contributions not available for this prediction.")
            
            st.subheader("Model Metadata")
            if result.get("model_metadata"):
                st.json(result["model_metadata"])
                
        except requests.exceptions.Timeout:
            st.error("❌ API request timed out. The server may be processing a slow request.")
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Could not connect to API at {PREDICT_ENDPOINT}. Ensure the FastAPI server is running.")
        except requests.exceptions.HTTPError as e:
            st.error(f"❌ API error: {e.response.status_code} - {e.response.text}")
        except Exception as exc:
            st.error(f"❌ Unexpected error: {exc}")

st.sidebar.header("About")
st.sidebar.write(
    "This dashboard calls the FastAPI fraud detection service and returns a risk label, probability, and explainable feature contributions."
)
st.sidebar.write(f"API endpoint: {PREDICT_ENDPOINT}")
st.sidebar.write(
    "If the API is running locally, keep the default URL. For Docker Compose, set API_URL=https://api:8000 or the service host name."
)
