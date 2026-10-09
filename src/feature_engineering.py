
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

print("Loading cleaned dataset...")

input_path = Path("data/processed/movie_success_cleaned.csv")
output_path = Path("data/processed/movie_success_ml_ready.csv")

if not input_path.exists():
    raise FileNotFoundError(
        f"Dataset not found: {input_path}\n"
        "Run python src/preprocess.py first."
    )

df = pd.read_csv(input_path, low_memory=False)

print("Original dataset shape:", df.shape)


# ==========================================
# 2. CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "id",
    "release_date",
    "budget",
    "runtime",
    "release_year",
    "release_month",
    "release_day",
    "is_friday_release",
    "is_weekend_release",
    "original_language",
    "genre",
    "production_country",
    "lead_actor",
    "director",
    "movie_success",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================
# 3. CLEAN IMPORTANT VALUES
# ==========================================

df["release_date"] = pd.to_datetime(
    df["release_date"],
    errors="coerce"
)

for col in ["budget", "runtime", "movie_success"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep only valid binary target labels.
df = df[df["movie_success"].isin([0, 1])].copy()

# Remove rows without a valid release date.
# A chronological history cannot be calculated reliably without it.
df = df.dropna(subset=["release_date"])

# Ensure names are strings and blank names are handled safely.
for col in ["director", "lead_actor"]:
    df[col] = df[col].fillna("").astype(str).str.strip()
    df.loc[df[col].str.lower().isin(["nan", "none"]), col] = ""

# Sort chronologically before calculating historical features.
df = df.sort_values(
    ["release_date", "id"]
).reset_index(drop=True)


# ==========================================
# 4. CALCULATE DIRECTOR HIT RATE
# ==========================================

print("Creating director historical hit rate...")

director_total = {}
director_hits = {}
director_rates = []

for _, row in df.iterrows():

    director = row["director"]

    # Use only movies dated before the current movie.
    # Movies on the same release date are not counted as history.
    if director and director in director_total:
        rate = (
            director_hits.get(director, 0)
            / director_total[director]
        )
    else:
        rate = np.nan

    director_rates.append(rate)

    # Update history after calculating the feature.
    if director:
        director_total[director] = (
            director_total.get(director, 0) + 1
        )

        director_hits[director] = (
            director_hits.get(director, 0)
            + int(row["movie_success"] == 1)
        )

df["director_hit_rate"] = director_rates


# ==========================================
# 5. CALCULATE LEAD ACTOR HIT RATE
# ==========================================

print("Creating actor historical hit rate...")

actor_total = {}
actor_hits = {}
actor_rates = []

for _, row in df.iterrows():

    actor = row["lead_actor"]

    # Only use the actor's earlier movies.
    if actor and actor in actor_total:
        rate = (
            actor_hits.get(actor, 0)
            / actor_total[actor]
        )
    else:
        rate = np.nan

    actor_rates.append(rate)

    # Update history after calculating the feature.
    if actor:
        actor_total[actor] = (
            actor_total.get(actor, 0) + 1
        )

        actor_hits[actor] = (
            actor_hits.get(actor, 0)
            + int(row["movie_success"] == 1)
        )

df["lead_actor_hit_rate"] = actor_rates


# ==========================================
# 6. CREATE RELEASE SEASON
# ==========================================

def get_season(month):

    if pd.isna(month):
        return "Unknown"

    month = int(month)

    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn"

    return "Unknown"


df["release_season"] = df["release_month"].apply(get_season)


# ==========================================
# 7. CLEAN CATEGORICAL FEATURES
# ==========================================

categorical_columns = [
    "genre",
    "original_language",
    "production_country",
    "release_season",
]

for col in categorical_columns:
    df[col] = (
        df[col]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )


# ==========================================
# 8. CLEAN NUMERICAL FEATURES
# ==========================================

numeric_columns = [
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

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep only plausible positive budgets.
df = df[df["budget"] > 0].copy()

# Do not calculate imputation values from the target.
# Missing numerical values will be handled by the ML pipeline.
df["director_hit_rate"] = df["director_hit_rate"].fillna(0.5)
df["lead_actor_hit_rate"] = df["lead_actor_hit_rate"].fillna(0.5)


# ==========================================
# 9. SELECT ML FEATURES
# ==========================================

feature_columns = [
    "budget",
    "runtime",
    "release_year",
    "release_month",
    "release_day",
    "is_friday_release",
    "is_weekend_release",
    "director_hit_rate",
    "lead_actor_hit_rate",
    "genre",
    "original_language",
    "production_country",
    "release_season",
]

target_column = "movie_success"

ml_df = df[feature_columns + [target_column]].copy()


# ==========================================
# 10. REMOVE INVALID TARGETS
# ==========================================

ml_df = ml_df[
    ml_df[target_column].isin([0, 1])
].copy()

ml_df[target_column] = ml_df[target_column].astype(int)


# ==========================================
# 11. CHECK RESULTS
# ==========================================

print("\n========== FEATURE ENGINEERING SUMMARY ==========")

print("Final dataset shape:", ml_df.shape)

print("\nFeatures:")
print(feature_columns)

print("\nTarget distribution:")
print(ml_df[target_column].value_counts().sort_index())

print("\nTarget percentages:")
print(
    (ml_df[target_column].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nMissing values:")
print(ml_df.isnull().sum())

print("\nFirst five rows:")
print(ml_df.head())


# ==========================================
# 12. SAVE ML-READY DATASET
# ==========================================

output_path.parent.mkdir(parents=True, exist_ok=True)

ml_df.to_csv(output_path, index=False)

print("\nSUCCESS!")
print("ML-ready dataset saved to:", output_path)
print("Next: inspect this dataset before model training.")
