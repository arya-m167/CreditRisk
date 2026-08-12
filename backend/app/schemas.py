from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ApplicantInput(BaseModel):
    LIMIT_BAL: float = Field(..., description="Credit limit (NT$)")
    SEX: int = Field(..., ge=1, le=2, description="1=male, 2=female")
    EDUCATION: int = Field(..., ge=1, le=4, description="1=grad school,2=university,3=high school,4=other")
    MARRIAGE: int = Field(..., ge=1, le=3, description="1=married,2=single,3=other")
    AGE: int
    PAY_1: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

    class Config:
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 120000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 2, "AGE": 26,
                "PAY_1": -1, "PAY_2": 2, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 2,
                "BILL_AMT1": 2682, "BILL_AMT2": 1725, "BILL_AMT3": 2682, "BILL_AMT4": 3272,
                "BILL_AMT5": 3455, "BILL_AMT6": 3261,
                "PAY_AMT1": 0, "PAY_AMT2": 1000, "PAY_AMT3": 1000, "PAY_AMT4": 1000,
                "PAY_AMT5": 0, "PAY_AMT6": 2000
            }
        }

class PredictionOut(BaseModel):
    id: int
    risk_score: float
    risk_band: str
    predicted_default: int
    created_at: datetime

    class Config:
        from_attributes = True

class ExplanationOut(BaseModel):
    id: int
    risk_score: float
    base_value: float
    top_factors: list[dict]  # [{"feature": "PAY_1", "value": 2, "shap_value": 0.12}, ...]

class PortfolioStats(BaseModel):
    total_scored: int
    overall_default_rate: float
    by_risk_band: dict
    recent: list[PredictionOut]
