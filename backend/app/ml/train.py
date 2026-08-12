"""
Train a credit default risk model on the UCI 'Default of Credit Card Clients' dataset.
Paper: Yeh, I.C. & Lien, C.H. (2009), Expert Systems with Applications 36(2), 2473-2480.

Produces:
  - model.joblib          (trained XGBoost classifier)
  - background.joblib     (sample of training data, used by SHAP at inference time)
  - feature_names.joblib  (ordered list of feature columns the model expects)
  - metrics.json          (test-set metrics + a simple fairness check by SEX)
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

DATA_PATH = "data/credit_card_default.csv"
OUT_DIR = "app/ml"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"PAY_0": "PAY_1", "default.payment.next.month": "DEFAULT"})
    df = df.drop(columns=["ID"])
    # Clean undocumented categories, per the original paper's follow-up analyses
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df

def fairness_check(model, X_test, y_test, sex_test):
    """Simple disparate-impact style check: false positive rate by SEX (1=male, 2=female)."""
    preds = model.predict(X_test)
    out = {}
    for group, label in [(1, "male"), (2, "female")]:
        mask = sex_test == group
        if mask.sum() == 0:
            continue
        y_g, p_g = y_test[mask], preds[mask]
        fp = ((p_g == 1) & (y_g == 0)).sum()
        neg = (y_g == 0).sum()
        out[label] = {"n": int(mask.sum()), "false_positive_rate": round(float(fp / neg), 4) if neg else None}
    return out

def main():
    df = load_data()
    y = df["DEFAULT"]
    X = df.drop(columns=["DEFAULT"])
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # Baseline
    lr = LogisticRegression(max_iter=2000, C=0.1)
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    # Main model
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    proba = xgb.predict_proba(X_test)[:, 1]
    preds = xgb.predict(X_test)

    metrics = {
        "baseline_logistic_regression_auc": round(float(lr_auc), 4),
        "xgboost_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "xgboost_f1": round(float(f1_score(y_test, preds)), 4),
        "xgboost_precision": round(float(precision_score(y_test, preds)), 4),
        "xgboost_recall": round(float(recall_score(y_test, preds)), 4),
        "fairness_false_positive_rate_by_sex": fairness_check(
            xgb, X_test, y_test.values, X_test["SEX"].values
        ),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "default_rate_test": round(float(y_test.mean()), 4),
    }

    print(json.dumps(metrics, indent=2))

    joblib.dump(xgb, f"{OUT_DIR}/model.joblib")
    joblib.dump(X_train.sample(200, random_state=42), f"{OUT_DIR}/background.joblib")
    joblib.dump(feature_names, f"{OUT_DIR}/feature_names.joblib")
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved model.joblib, background.joblib, feature_names.joblib, metrics.json to", OUT_DIR)

if __name__ == "__main__":
    main()
