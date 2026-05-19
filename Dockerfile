FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000 8501 5000

CMD ["uvicorn", "fraud_api.app:app", "--host", "0.0.0.0", "--port", "8000"]
