"""
Rigorous model evaluation: 5-fold cross-validation (not just one train/test split),
ROC curve, confusion matrix, and a calibration check (are predicted probabilities
actually trustworthy as probabilities, not just useful for ranking).

Run after train.py. Produces:
  - app/ml/plots/roc_curve.png
  - app/ml/plots/confusion_matrix.png
  - app/ml/plots/calibration_curve.png
  - app/ml/eval_metrics.json
"""
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

DATA_PATH = "data/credit_card_default.csv"
OUT_DIR = "app/ml"
PLOTS_DIR = f"{OUT_DIR}/plots"

import os
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"PAY_0": "PAY_1", "default.payment.next.month": "DEFAULT"})
    df = df.drop(columns=["ID"])
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    return df

def main():
    df = load_data()
    y = df["DEFAULT"]
    X = df.drop(columns=["DEFAULT"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc", random_state=42, n_jobs=1,
    )

    # --- 1. Cross-validation: is the single train/test AUC a fluke, or stable? ---
    print("Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"CV AUC scores: {[round(s, 4) for s in cv_scores]}")
    print(f"CV AUC mean +/- std: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # --- 2. Fit final model on full training set, evaluate on held-out test set ---
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    test_auc = roc_auc_score(y_test, proba)
    print(f"\nHeld-out test AUC: {test_auc:.4f}")
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, preds, target_names=["no default", "default"]))

    # --- 3. ROC curve plot ---
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="#E8543D", linewidth=2, label=f"XGBoost (AUC = {test_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#565E68", linestyle="--", linewidth=1, label="Random guess (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Credit Default Prediction")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # --- 4. Confusion matrix plot ---
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (threshold=0.5)")
    plt.colorbar()
    classes = ["No default", "Default"]
    plt.xticks([0, 1], classes)
    plt.yticks([0, 1], classes)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # --- 5. Calibration curve: when the model says "70% risk", does ~70% of
    # that group actually default? This matters because a risk score is only
    # useful for real decisions if the probabilities themselves are trustworthy,
    # not just useful for ranking applicants relative to each other. ---
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", color="#2FBF8F", label="XGBoost")
    plt.plot([0, 1], [0, 1], color="#565E68", linestyle="--", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of actual positives")
    plt.title("Calibration Curve")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/calibration_curve.png", dpi=150)
    plt.close()

    eval_metrics = {
        "cv_auc_scores": [round(float(s), 4) for s in cv_scores],
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
        "test_auc": round(float(test_auc), 4),
        "confusion_matrix": cm.tolist(),
        "n_test": len(y_test),
    }
    with open(f"{OUT_DIR}/eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    print(f"\nSaved plots to {PLOTS_DIR}/ and eval_metrics.json to {OUT_DIR}/")

if __name__ == "__main__":
    main()
