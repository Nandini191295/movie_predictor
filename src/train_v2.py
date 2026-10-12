
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "movie_success_ml_ready.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "movie_success_model_v2.pkl"
RESULTS_PATH = MODEL_DIR / "model_comparison_v2.csv"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.astype(str).str.strip()

target = "movie_success"

numeric_features = [
    "budget",
    "runtime",
    "release_year",
    "release_month",
    "release_day",
    "is_friday_release",
    "is_weekend_release"
]

categorical_features = [
    "genre",
    "original_language",
    "production_country",
    "release_season"
]

features = numeric_features + categorical_features

required = features + [target]
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df[target] = pd.to_numeric(df[target], errors="coerce")
df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)

df = df.dropna(subset=[target, "release_year"]).copy()
df[target] = df[target].astype(int)

if not set(df[target].unique()).issubset({0, 1}):
    raise ValueError("movie_success must contain only 0 and 1.")

df = df.sort_values("release_year").reset_index(drop=True)

# --------------------------------------------------
# CHRONOLOGICAL TRAIN / TEST SPLIT
# --------------------------------------------------
years = sorted(df["release_year"].unique())

if len(years) < 2:
    raise ValueError("At least two release years are required.")

cut_index = max(1, int(len(years) * 0.8))
cut_index = min(cut_index, len(years) - 1)
cut_year = years[cut_index]

train_df = df[df["release_year"] < cut_year].copy()
test_df = df[df["release_year"] >= cut_year].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Train and test sets must each contain both HIT and FLOP classes. "
        "Check the chronological split."
    )

print("Training movies:", len(train_df))
print("Testing movies:", len(test_df))
print("Test period starts:", cut_year)
print("\nTraining class distribution:")
print(y_train.value_counts(normalize=True).round(3))
print("\nTest class distribution:")
print(y_test.value_counts(normalize=True).round(3))


# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
def make_pipeline(classifier, scale_numeric=False):
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


# --------------------------------------------------
# MODELS
# --------------------------------------------------
models = {
    "Logistic Regression": make_pipeline(
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            random_state=42
        ),
        scale_numeric=True
    ),

    "Decision Tree": make_pipeline(
        DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        )
    ),

    "Random Forest": make_pipeline(
        RandomForestClassifier(
            n_estimators=250,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    ),

    "Extra Trees": make_pipeline(
        ExtraTreesClassifier(
            n_estimators=250,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    )
}


# --------------------------------------------------
# TRAIN AND EVALUATE
# --------------------------------------------------
results = []
fitted_models = {}

for name, pipeline in models.items():
    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": None,
        "PR-AUC": None
    }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        row["ROC-AUC"] = roc_auc_score(y_test, probabilities)
        row["PR-AUC"] = average_precision_score(
            y_test, probabilities
        )

    results.append(row)
    fitted_models[name] = pipeline

    print("Confusion matrix (rows actual, columns predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


# --------------------------------------------------
# SAVE COMPARISON
# --------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="F1 Score", ascending=False
)

results_df.to_csv(RESULTS_PATH, index=False)

best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]

joblib.dump(best_model, MODEL_PATH)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nBest model by test F1-score:", best_name)
print("Saved model:", MODEL_PATH)
print("Saved comparison:", RESULTS_PATH)
print("\nTraining completed.")
