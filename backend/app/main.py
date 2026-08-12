from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app.routers import predictions

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Explainable Credit Risk API",
    description="Scores credit card applicants for default risk (XGBoost + SHAP), "
                 "based on Yeh & Lien (2009). Every prediction is logged for audit review.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router, tags=["predictions"])

@app.get("/health")
def health():
    return {"status": "ok"}
