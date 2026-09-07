import os
import joblib
import pandas as pd

from ucimlrepo import fetch_ucirepo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "liver_model.joblib"
RESULTS_FILE = "model_results.txt"

FEATURES = ["Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"]

def load_data():
    print("Downloading/loading UCI ILPD dataset...")
    dataset = fetch_ucirepo(id=225)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    # Make column names predictable.
    X.columns = FEATURES
    target_name = y.columns[0]

    # UCI labels: 1 = liver disease, 2 = no liver disease.
    y = y[target_name].astype(int).map({1: 1, 2: 0})

    return X, y

def build_preprocessor():
    numeric_features = ["Age", "TB", "DB", "Alkphos", "Sgpt", "Sgot", "TP", "ALB", "A/G Ratio"]
    categorical_features = ["Gender"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

def train_and_save_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        )
    }

    results = []
    best_name = None
    best_pipeline = None
    best_f1 = -1

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator)
        ])

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, pred)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)

        results.append(
            f"{name}\n"
            f"Accuracy : {accuracy:.4f}\n"
            f"Precision: {precision:.4f}\n"
            f"Recall   : {recall:.4f}\n"
            f"F1-score : {f1:.4f}\n"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_pipeline = pipeline

    joblib.dump(best_pipeline, MODEL_FILE)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("LIVER DISEASE SCREENING MODEL RESULTS\n")
        f.write("=====================================\n\n")
        f.write("\n".join(results))
        f.write(f"\nSelected model: {best_name}\n")

    print("\nTraining completed.")
    print("\n".join(results))
    print(f"Selected model: {best_name}")
    print(f"Saved model to: {MODEL_FILE}")
    print(f"Saved metrics to: {RESULTS_FILE}")

    return best_pipeline

if __name__ == "__main__":
    train_and_save_model()
