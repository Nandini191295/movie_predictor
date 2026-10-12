
from pathlib import Path

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data/processed/movie_success_ml_ready.csv"
MODEL_PATH = BASE_DIR / "models/movie_success_model_v3.pkl"
INFO_PATH = BASE_DIR / "models/model_info_v3.json"

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

numeric = [
    "budget", "runtime", "release_year", "release_month",
    "release_day", "is_friday_release", "is_weekend_release"
]
categorical = [
    "genre", "original_language",
    "production_country", "release_season"
]
features = numeric + categorical
target = "movie_success"

df[target] = pd.to_numeric(df[target], errors="coerce")
df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)
df = df.dropna(subset=[target, "release_year"]).copy()
df[target] = df[target].astype(int)
df = df.sort_values("release_year").reset_index(drop=True)

years = sorted(df["release_year"].unique())
if len(years) < 5:
    raise ValueError("Not enough distinct years for train/validation/test.")

# 70% earlier years for training, next 15% for validation,
# and final 15% for the untouched test period.
train_cut = years[max(1, int(len(years) * 0.70))]
valid_cut = years[max(2, int(len(years) * 0.85))]

if valid_cut <= train_cut:
    raise ValueError("Could not create distinct chronological splits.")

train = df[df["release_year"] < train_cut]
valid = df[
    (df["release_year"] >= train_cut)
    & (df["release_year"] < valid_cut)
]
test = df[df["release_year"] >= valid_cut]

if min(len(train), len(valid), len(test)) == 0:
    raise ValueError("One of the chronological splits is empty.")

X_train, y_train = train[features], train[target]
X_valid, y_valid = valid[features], valid[target]
X_test, y_test = test[features], test[target]

if any(y.nunique() != 2 for y in [y_train, y_valid, y_test]):
    raise ValueError(
        "Each split must contain both classes. Review the year boundaries."
    )

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_transformer, numeric),
    ("categorical", categorical_transformer, categorical)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print("Training years before:", train_cut)
print("Validation years:", train_cut, "to", valid_cut - 1)
print("Test starts in:", valid_cut)
print("Rows:", len(train), len(valid), len(test))

model.fit(X_train, y_train)

# Select threshold on validation data only.
valid_prob = model.predict_proba(X_valid)[:, 1]

threshold_results = []
for threshold in np.arange(0.20, 0.81, 0.02):
    valid_pred = (valid_prob >= threshold).astype(int)
    threshold_results.append({
        "threshold": float(threshold),
        "f1": f1_score(y_valid, valid_pred, zero_division=0),
        "precision": precision_score(
            y_valid, valid_pred, zero_division=0
        ),
        "recall": recall_score(
            y_valid, valid_pred, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)
best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
best_threshold = float(best_row["threshold"])

print("\nBest validation threshold:", round(best_threshold, 2))
print("Validation F1:", round(best_row["f1"], 4))
print("Validation precision:", round(best_row["precision"], 4))
print("Validation recall:", round(best_row["recall"], 4))

# Evaluate only once on the later test period.
test_prob = model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\n========== FINAL TEST RESULTS ==========")
print("Test accuracy:", round(accuracy_score(y_test, test_pred), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, test_pred), 4)
)
print("Precision:", round(
    precision_score(y_test, test_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, test_pred, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_test, test_pred, zero_division=0), 4
))
print("ROC-AUC:", round(
    roc_auc_score(y_test, test_prob), 4
))
print("Confusion matrix (actual rows, predicted columns):")
print(confusion_matrix(y_test, test_pred, labels=[0, 1]))

joblib.dump(model, MODEL_PATH)

with open(INFO_PATH, "w", encoding="utf-8") as file:
    json.dump({
        "model_name": "Extra Trees",
        "threshold": best_threshold,
        "features": features,
        "target": target,
        "train_end_year": int(train_cut) - 1,
        "validation_end_year": int(valid_cut) - 1,
        "test_start_year": int(valid_cut)
    }, file, indent=2)

print("\nSaved model:", MODEL_PATH)
print("Saved model settings:", INFO_PATH)
