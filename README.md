# Nepal Mobile Banking Fraud Detection System

A portfolio-grade machine learning and MLOps project for mobile wallet fraud detection in the Nepal fintech context.

This repository includes:
- Synthetic Nepali mobile wallet transaction generation
- Rule-based and CTGAN data augmentation
- End-to-end training pipeline with model benchmarking and threshold optimization
- SHAP explainability and business-readable fraud reasoning
- FastAPI serving with single and batch prediction endpoints
- Streamlit dashboard for interactive prediction and analysis
- Docker and CI-ready configuration for deployment

## Features

- Synthetic transaction dataset with Nepal-specific merchants, cities, device types, and channels
- Fraud labels covering velocity anomalies, geo anomalies, amount spikes, and merchant irregularities
- Model artifacts stored under `models/` and threshold metadata in `models/threshold.json`
- FastAPI endpoints:
  - `GET /health`
  - `POST /predict`
  - `POST /batch_predict`
- Streamlit dashboard available via `streamlit run streamlit_app/dashboard.py`

## Repository Structure

- `generate_data.py` — Generate and optionally augment transaction data
- `train_pipeline.py` — Train model, evaluate performance, and save artifacts
- `explainability.py` — Build SHAP explainability summaries
- `fraud_api/app.py` — API service implementation
- `streamlit_app/dashboard.py` — Dashboard UI
- `config.yaml` / `params.yaml` — Configuration and parameter defaults
- `Dockerfile` / `docker-compose.yml` — Local container setup
- `tests/` — Unit tests for data generation, configuration, and API behavior

## Getting Started

1. Create and activate a virtual environment:

   Windows PowerShell:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS / Linux:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Generate synthetic data:

   ```bash
   python generate_data.py
   ```

4. Train the model and save artifacts:

   ```bash
   python train_pipeline.py
   ```

5. Run the API service:

   ```bash
   uvicorn fraud_api.app:app --host 0.0.0.0 --port 8000
   ```

6. Run the Streamlit dashboard:

   ```bash
   streamlit run streamlit_app/dashboard.py
   ```

## Running Tests

```bash
python -m pytest -q
```

## Docker

To run the app with Docker Compose:

```bash
docker compose up --build
```

## Notes

- This project is designed for demonstration and evaluation purposes.
- The dataset is synthetic and should not be used for production fraud detection without additional validation.
- Use `config.yaml` and `params.yaml` to tune data generation, model training, and API behavior.
