from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.db_models import Prediction
from app.schemas import ApplicantInput, PredictionOut, ExplanationOut, PortfolioStats
from app.ml.inference import score_applicant, explain_applicant

router = APIRouter()

@router.post("/predict", response_model=PredictionOut)
def predict(applicant: ApplicantInput, db: Session = Depends(get_db)):
    features = applicant.model_dump()
    result = score_applicant(features)

    record = Prediction(
        limit_bal=features["LIMIT_BAL"],
        sex=features["SEX"],
        education=features["EDUCATION"],
        marriage=features["MARRIAGE"],
        age=features["AGE"],
        features_json=features,
        risk_score=result["risk_score"],
        risk_band=result["risk_band"],
        predicted_default=result["predicted_default"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

@router.get("/explain/{prediction_id}", response_model=ExplanationOut)
def explain(prediction_id: int, db: Session = Depends(get_db)):
    record = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")
    result = explain_applicant(record.features_json)
    return {
        "id": record.id,
        "risk_score": result["risk_score"],
        "base_value": result["base_value"],
        "top_factors": result["top_factors"],
    }

@router.get("/portfolio/stats", response_model=PortfolioStats)
def portfolio_stats(db: Session = Depends(get_db)):
    total = db.query(Prediction).count()
    if total == 0:
        return {"total_scored": 0, "overall_default_rate": 0.0, "by_risk_band": {}, "recent": []}

    avg_default = db.query(func.avg(Prediction.predicted_default)).scalar() or 0.0

    band_counts = {}
    for band in ["low", "medium", "high"]:
        band_counts[band] = db.query(Prediction).filter(Prediction.risk_band == band).count()

    recent = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )

    return {
        "total_scored": total,
        "overall_default_rate": round(float(avg_default), 4),
        "by_risk_band": band_counts,
        "recent": recent,
    }
