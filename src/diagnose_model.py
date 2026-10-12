
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "movie_success_ml_ready.csv"
MODEL_PATH = BASE_DIR / "models" / "movie_success_model.pkl"

if not DATA_PATH.exists() or not MODEL_PATH.exists():
    raise FileNotFoundError("Check the dataset and saved model paths.")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)

df.columns = df.columns.astype(str).str.strip()

numeric_features = [
    "budget", "runtime", "release_year", "release_month",
    "release_day", "is_friday_release", "is_weekend_release",
    "director_hit_rate", "lead_actor_hit_rate"
]

categorical_features = [
    "genre", "original_language",
    "production_country", "release_season"
]

features = numeric_features + categorical_features
required = features + ["movie_success"]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

df = df.dropna(subset=["movie_success"]).copy()
df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)
df = df.dropna(subset=["release_year"])
df = df.sort_values("release_year").reset_index(drop=True)

years = sorted(df["release_year"].unique())

if len(years) < 2:
    raise ValueError("Not enough different release years for a time split.")

cut_index = max(1, int(len(years) * 0.8))

if cut_index >= len(years):
    cut_index = len(years) - 1

cut_year = years[cut_index]

test_df = df[df["release_year"] >= cut_year].copy()

X_test = test_df[features]
y_test = test_df["movie_success"].astype(int)

predictions = model.predict(X_test).astype(int)

print("\n========== TEST SET ==========")
print("Test starts in release year:", cut_year)
print("Test movies:", len(test_df))

print("\nActual class distribution:")
print(y_test.value_counts().rename(index={0: "FLOP", 1: "HIT"}))

print("\nPredicted class distribution:")
print(
    pd.Series(predictions)
    .value_counts()
    .rename(index={0: "FLOP", 1: "HIT"})
)

print("\n========== CONFUSION MATRIX ==========")
print("Rows = actual; columns = predicted")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))
print("\nClass order: FLOP, HIT")

print("\n========== CLASSIFICATION REPORT ==========")
print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["FLOP", "HIT"],
        zero_division=0
    )
)

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_test)
    classes = list(model.classes_)

    if 1 in classes and y_test.nunique() == 2:
        auc = roc_auc_score(
            y_test,
            probabilities[:, classes.index(1)]
        )
        print("ROC-AUC:", round(auc, 4))

print("\n========== DIAGNOSTICS COMPLETE ==========")
