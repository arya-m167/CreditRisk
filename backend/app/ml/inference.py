import os
import joblib
import shap
import pandas as pd

_ML_DIR = os.path.dirname(__file__)

_model = joblib.load(os.path.join(_ML_DIR, "model.joblib"))
_background = joblib.load(os.path.join(_ML_DIR, "background.joblib"))
_feature_names = joblib.load(os.path.join(_ML_DIR, "feature_names.joblib"))
_explainer = shap.TreeExplainer(_model, feature_perturbation="tree_path_dependent")

def risk_band(score: float) -> str:
    if score < 0.2:
        return "low"
    if score < 0.5:
        return "medium"
    return "high"

def score_applicant(features: dict) -> dict:
    """features: dict matching _feature_names order/keys -> returns score + band."""
    X = pd.DataFrame([features])[_feature_names]
    proba = float(_model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return {"risk_score": proba, "risk_band": risk_band(proba), "predicted_default": pred}

def explain_applicant(features: dict, top_n: int = 6) -> dict:
    X = pd.DataFrame([features])[_feature_names]
    shap_values = _explainer(X)
    base_value = float(shap_values.base_values[0])
    values = shap_values.values[0]

    contributions = sorted(
        zip(_feature_names, X.iloc[0].tolist(), values.tolist()),
        key=lambda t: abs(t[2]),
        reverse=True,
    )[:top_n]

    top_factors = [
        {"feature": f, "value": v, "shap_value": round(s, 4)}
        for f, v, s in contributions
    ]
    proba = float(_model.predict_proba(X)[0, 1])
    return {"risk_score": proba, "base_value": base_value, "top_factors": top_factors}
