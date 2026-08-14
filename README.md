# Explainable Credit Risk Scoring

A full-stack credit default risk model with per-decision explanations, built on a real
published dataset — not a toy Kaggle exercise. Reproduces and extends:

> Yeh, I-Cheng & Lien, Che-hui (2009). *The comparisons of data mining techniques for the
> predictive accuracy of probability of default of credit card clients.* Expert Systems with
> Applications, 36(2), 2473–2480.

**Stack:** React (frontend) · FastAPI (backend) · PostgreSQL (audit log) · XGBoost + SHAP (model)

## Why this project

Most "credit risk" portfolio projects stop at a notebook and an accuracy number. This one asks
the question a real risk desk — or a regulator — actually asks: **why** did the model say what
it said, and is it treating people fairly? Every prediction is explained with SHAP and logged
to an audit trail, and the model is checked for disparate impact across gender before being
called done. That's the standard financial institutions in regulated markets (Hong Kong's HKMA
included) are expected to meet for AI-assisted credit decisions.

## Architecture

```
React frontend  →  FastAPI backend  →  PostgreSQL (applicants, predictions, audit log)
                          ↓
                   XGBoost + SHAP  (risk score + feature attributions)
```

- `POST /predict` — score an applicant, log the result
- `GET /explain/{id}` — SHAP breakdown of a stored prediction
- `GET /portfolio/stats` — aggregate default rate, risk-band distribution, recent activity

## Model

| Model | Test AUC |
|---|---|
| Logistic regression (baseline) | 0.716 |
| **XGBoost (used in the app)** | **0.778** |

Trained on the full UCI *Default of Credit Card Clients* dataset — 30,000 Taiwanese credit
card accounts, April–September 2005, 23 features (demographics, six months of repayment status,
bill amounts, and payment amounts). Class-imbalance handled via `scale_pos_weight` rather than
resampling, to keep SHAP attributions on real (not synthetic) data.

Metrics on held-out test set (25% split, stratified): F1 = 0.536, precision = 0.467, recall = 0.630.
An AUC around 0.78 is consistent with the published literature on this dataset — treat anything
claiming 0.95+ on this data as a leakage bug, not a better model.

## Fairness check

Before shipping, the model's false-positive rate was checked across the `SEX` field:

| Group | False positive rate |
|---|---|
| Male | 23.2% |
| Female | 18.7% |

The model is meaningfully more likely to flag male applicants as high-risk when they will not
actually default. This is a real finding, not a caveat pasted in for show — it's exactly the
kind of gap a model-risk review would flag, and worth investigating further (e.g. whether it's
driven by a proxy variable like `EDUCATION` or `MARRIAGE` correlating with `SEX`) before any
such model went near a production lending decision.

## Running locally

python3 -m venv venv
source venv/bin/activate

**Backend**
```bash
cd backend
pip install -r requirements.txt
python app/ml/train.py      # only needed if you want to retrain; artifacts are already committed
uvicorn app.main:app --reload
```

**Frontend**
```bash
cd frontend
npm install
npm run dev
```

**Or, with Docker Compose** (Postgres + backend + frontend):
```bash
docker compose up --build
```
Frontend on `:5173`, API on `:8000`, Postgres on `:5432`.

## Dataset

[UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
(Yeh & Lien, 2009), retrieved via GitHub mirror since UCI's direct download was unreachable
from the build environment. Raw CSV included under `backend/data/`.

## What's not here (yet)

- Model monitoring / drift detection over time
- Authentication on the API (fine for a demo, not for production)
- A held-out temporal validation split (the original data only spans 6 months, so there's no
  later period to validate against)
