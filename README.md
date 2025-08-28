import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import joblib
import os

# ------------------------------
# Utility: Create Synthetic Dataset
# ------------------------------
def make_synthetic_medical_dataset(name, n_samples=1200):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        weights=[0.7, 0.3],
        class_sep=1.2,
        random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(X.shape[1])])

    if name == "heart":
        feature_names = [
            "age","sex","chest_pain_type","resting_bp","cholesterol","fasting_bs",
            "rest_ecg","max_heart_rate","exercise_angina","oldpeak","slope","ca"
        ]
    elif name == "lung":
        feature_names = [
            "age","sex","smoking_history","cough_days","blood_in_sputum","weight_loss",
            "breathlessness","chest_pain","history_cancer","tumor_marker_1","tumor_marker_2","tumor_marker_3"
        ]
    elif name == "thyroid":
        feature_names = [
            "age","sex","neck_swelling","fatigue","weight_change","cold_intolerance",
            "heat_intolerance","heart_rate","tsh_level","t3_level","t4_level","autoantibodies"
        ]
    else:
        feature_names = df.columns.tolist()

    df.columns = feature_names
    df["target"] = y
    return df

# ------------------------------
# Train + Evaluate Models
# ------------------------------
def train_and_evaluate(df, name, output_dir="disease_models"):
    os.makedirs(output_dir, exist_ok=True)

    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Models
    lr = LogisticRegression(max_iter=1000, solver="liblinear")
    rf = RandomForestClassifier(n_estimators=200, random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # Predictions + Evaluation
    def evaluate(model, model_name):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        print(f"\n{name.upper()} - {model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

    # Evaluate both
    plt.figure(figsize=(6,5))
    evaluate(lr, "Logistic Regression")
    evaluate(rf, "Random Forest")

    # Final ROC curve
    plt.plot([0,1],[0,1],'--')
    plt.title(f"{name.capitalize()} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save models
    joblib.dump(lr, os.path.join(output_dir, f"{name}_logreg.pkl"))
    joblib.dump(rf, os.path.join(output_dir, f"{name}_rf.pkl"))
    print(f"\nModels saved to {output_dir}/")

    # ✅ FIXED: Predictions before adding columns
    sample = X_test.iloc[:6].copy()
    probs = rf.predict_proba(sample)[:,1].round(3)
    preds = rf.predict(sample)
    sample["predicted_probability"] = probs
    sample["predicted_label"] = preds

    print(f"\n{name.upper()} - Sample Predictions:")
    print(sample.head())

# ------------------------------
# Run for all 3 diseases
# ------------------------------
for disease in ["heart", "lung", "thyroid"]:
    df = make_synthetic_medical_dataset(disease)
    train_and_evaluate(df, disease)
