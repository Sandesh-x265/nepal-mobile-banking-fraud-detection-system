from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    customer_id: str
    amount_npr: float
    city: str
    merchant_category: str
    merchant_name: str
    device_type: str
    channel: str
    is_new_merchant: int = Field(0, ge=0, le=1)


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionRequest]


class ShapContribution(BaseModel):
    feature: str
    value: str
    shap_value: float


class PredictionResponse(BaseModel):
    probability: float
    risk_label: str
    threshold: float
    shap_summary: List[ShapContribution]
    business_reason: str
    model_metadata: Optional[Dict[str, object]] = None


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
