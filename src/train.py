
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading ML-ready dataset...")

data_path = Path("data/processed/movie_success_ml_ready.csv")
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
    raise FileNotFoundError(
        "ML-ready dataset not found. Run "
        "python src/feature_engineering.py first."
    )

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)


# ==========================================
# 2. DEFINE FEATURES AND TARGET
# ==========================================

target = "movie_success"

numeric_features = [
    "budget",
    "runtime",
    "release_year",
    "release_month",
    "release_day",
    "is_friday_release",
    "is_weekend_release",
    "director_hit_rate",
    "lead_actor_hit_rate",
]

categorical_features = [
    "genre",
    "original_language",
    "production_country",
    "release_season",
]

feature_columns = numeric_features + categorical_features

missing = [
    col for col in feature_columns + [target]
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df.dropna(subset=[target]).copy()
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df[df[target].isin([0, 1])].copy()
df[target] = df[target].astype(int)

df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)

# Require a valid year for a chronological split.
df = df.dropna(subset=["release_year"]).copy()
df["release_year"] = df["release_year"].astype(int)

X = df[feature_columns]
y = df[target]


# ==========================================
# 3. CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================

print("\nPreparing chronological split...")

years = sorted(df["release_year"].unique())

if len(years) < 2:
    raise ValueError(
        "Not enough distinct release years for a chronological split."
    )

# Hold out the latest 20% of distinct years.
cut_index = max(1, int(len(years) * 0.8))
cut_index = min(cut_index, len(years) - 1)
cut_year = years[cut_index]

train_mask = df["release_year"] < cut_year
test_mask = df["release_year"] >= cut_year

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

if X_train.empty or X_test.empty:
    raise ValueError(
        "Training or testing data is empty. Check release years."
    )

if y_train.nunique() < 2:
    raise ValueError(
        "Training data has only one class. A classifier needs HIT and FLOP examples."
    )

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Training years:", df.loc[train_mask, "release_year"].min(),
      "-", df.loc[train_mask, "release_year"].max())
print("Testing years:", df.loc[test_mask, "release_year"].min(),
      "-", df.loc[test_mask, "release_year"].max())

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# ==========================================
# 4. PREPROCESSING PIPELINE
# ==========================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])


# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
}


# ==========================================
# 6. TRAIN AND EVALUATE
# ==========================================

results = []
trained_pipelines = {}

print("\n========== MODEL TRAINING ==========")

for name, classifier in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )

    try:
        auc = roc_auc_score(y_test, probabilities)
    except ValueError:
        auc = np.nan

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc,
    })

    trained_pipelines[name] = pipeline

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}" if not np.isnan(auc)
          else "ROC-AUC: unavailable")


# ==========================================
# 7. COMPARE MODELS
# ==========================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# ==========================================
# 8. SAVE BEST MODEL
# ==========================================

best_name = results_df.iloc[0]["Model"]
best_pipeline = trained_pipelines[best_name]

joblib.dump(
    best_pipeline,
    model_dir / "movie_success_model.pkl"
)

# Save feature information for the application.
joblib.dump(
    {
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target": target,
        "best_model_name": best_name,
        "test_cut_year": cut_year,
    },
    model_dir / "model_info.pkl"
)

print("\n========== BEST MODEL ==========")
print("Selected model:", best_name)

best_predictions = best_pipeline.predict(X_test)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        best_predictions,
        labels=[0, 1],
        target_names=["FLOP", "HIT"],
        zero_division=0
    )
)

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, best_predictions, labels=[0, 1]))

print("\nSaved model to models/movie_success_model.pkl")
print("Saved comparison to models/model_comparison.csv")
print("\nTraining completed!")
