
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

files = {
    "Cleaned dataset": BASE_DIR / "data/processed/movie_success_cleaned.csv",
    "ML-ready dataset": BASE_DIR / "data/processed/movie_success_ml_ready.csv",
    "Saved model": BASE_DIR / "models/movie_success_model.pkl",
    "Model comparison": BASE_DIR / "models/model_comparison.csv",
    "Model information": BASE_DIR / "models/model_info.pkl",
}

print("\n========== PROJECT FILE AUDIT ==========")

for name, path in files.items():
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"[FOUND] {name}: {path} ({size_mb:.2f} MB)")
    else:
        print(f"[MISSING] {name}: {path}")

# Inspect the ML-ready dataset
data_path = files["ML-ready dataset"]

if data_path.exists():
    df = pd.read_csv(data_path)

    print("\n========== DATASET SUMMARY ==========")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("\nColumn names:")
    print(df.columns.tolist())

    if "movie_success" in df.columns:
        print("\nHIT/FLOP distribution:")
        print(df["movie_success"].value_counts(dropna=False))

    print("\nColumns with most missing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

# Inspect the saved model comparison
results_path = files["Model comparison"]

if results_path.exists():
    results = pd.read_csv(results_path)

    print("\n========== MODEL COMPARISON ==========")
    print("Result columns:", results.columns.tolist())
    print(results.to_string(index=False))

print("\n========== AUDIT COMPLETE ==========")
