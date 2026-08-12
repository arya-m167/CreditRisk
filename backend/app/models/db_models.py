from sqlalchemy import Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base

class Prediction(Base):
    """Every scored applicant, kept as an audit trail (mirrors what a real
    credit-risk system needs for regulatory review)."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Raw input features (subset shown for brevity, all 23 stored as JSON too)
    limit_bal = Column(Float)
    sex = Column(Integer)
    education = Column(Integer)
    marriage = Column(Integer)
    age = Column(Integer)
    features_json = Column(JSON)  # full feature dict, for reproducibility

    # Model output
    risk_score = Column(Float)       # probability of default, 0-1
    risk_band = Column(String)       # low / medium / high
    predicted_default = Column(Integer)  # 0 or 1
    model_version = Column(String, default="xgboost-v1")
