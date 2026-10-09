import pandas as pd
import ast


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading datasets...")

movies = pd.read_csv(
    "data/raw/movies_metadata.csv",
    low_memory=False
)

credits = pd.read_csv(
    "data/raw/credits.csv",
    low_memory=False
)

print("Movies:", movies.shape)
print("Credits:", credits.shape)


# ==========================================
# 2. CLEAN MOVIE IDs
# ==========================================

movies["id"] = pd.to_numeric(
    movies["id"],
    errors="coerce"
)

credits["id"] = pd.to_numeric(
    credits["id"],
    errors="coerce"
)

# Remove rows where ID is missing
movies = movies.dropna(subset=["id"])
credits = credits.dropna(subset=["id"])

# Convert ID to integer
movies["id"] = movies["id"].astype(int)
credits["id"] = credits["id"].astype(int)


# ==========================================
# 3. REMOVE DUPLICATE MOVIES
# ==========================================

movies = movies.drop_duplicates(
    subset=["id"]
)

credits = credits.drop_duplicates(
    subset=["id"]
)

print("Movies after duplicate removal:", movies.shape)
print("Credits after duplicate removal:", credits.shape)


# ==========================================
# 4. CONVERT IMPORTANT COLUMNS
# ==========================================

movies["budget"] = pd.to_numeric(
    movies["budget"],
    errors="coerce"
)

movies["revenue"] = pd.to_numeric(
    movies["revenue"],
    errors="coerce"
)

movies["runtime"] = pd.to_numeric(
    movies["runtime"],
    errors="coerce"
)

movies["release_date"] = pd.to_datetime(
    movies["release_date"],
    errors="coerce"
)


# ==========================================
# 5. MERGE MOVIES + CREDITS
# ==========================================

print("\nMerging datasets...")

df = movies.merge(
    credits,
    on="id",
    how="left"
)

print("Merged dataset shape:", df.shape)


# ==========================================
# 6. EXTRACT GENRE
# ==========================================

def extract_names(value):

    try:

        data = ast.literal_eval(value)

        if isinstance(data, list):

            names = []

            for item in data:

                if isinstance(item, dict):

                    if "name" in item:
                        names.append(item["name"])

            return ", ".join(names)

    except:

        return ""

    return ""


df["genre"] = df["genres"].apply(
    extract_names
)


# ==========================================
# 7. EXTRACT PRODUCTION COMPANY
# ==========================================

df["production_company"] = (
    df["production_companies"]
    .apply(extract_names)
)


# ==========================================
# 8. EXTRACT PRODUCTION COUNTRY
# ==========================================

df["production_country"] = (
    df["production_countries"]
    .apply(extract_names)
)


# ==========================================
# 9. EXTRACT SPOKEN LANGUAGE
# ==========================================

df["spoken_language"] = (
    df["spoken_languages"]
    .apply(extract_names)
)


# ==========================================
# 10. EXTRACT DIRECTOR
# ==========================================

def extract_director(value):

    try:

        data = ast.literal_eval(value)

        if isinstance(data, list):

            for item in data:

                if isinstance(item, dict):

                    if item.get("job") == "Director":

                        return item.get("name", "")

    except:

        return ""

    return ""


df["director"] = df["crew"].apply(
    extract_director
)


# ==========================================
# 11. EXTRACT LEAD ACTOR
# ==========================================

def extract_lead_actor(value):

    try:

        data = ast.literal_eval(value)

        if isinstance(data, list):

            if len(data) > 0:

                first_actor = data[0]

                if isinstance(first_actor, dict):

                    return first_actor.get(
                        "name",
                        ""
                    )

    except:

        return ""

    return ""


df["lead_actor"] = df["cast"].apply(
    extract_lead_actor
)


# ==========================================
# 12. CREATE RELEASE FEATURES
# ==========================================

df["release_year"] = (
    df["release_date"].dt.year
)

df["release_month"] = (
    df["release_date"].dt.month
)

df["release_day"] = (
    df["release_date"].dt.day
)

df["release_weekday"] = (
    df["release_date"].dt.day_name()
)


# Friday release

df["is_friday_release"] = (
    df["release_weekday"] == "Friday"
).astype(int)


# Weekend release

df["is_weekend_release"] = (
    df["release_weekday"].isin(
        ["Friday", "Saturday", "Sunday"]
    )
).astype(int)


# ==========================================
# 13. CREATE PROFIT
# ==========================================

df["profit"] = (
    df["revenue"] - df["budget"]
)


# ==========================================
# 14. CREATE ROI
# ==========================================

df["roi"] = 0.0

valid_budget = df["budget"] > 0

df.loc[valid_budget, "roi"] = (
    (
        df.loc[valid_budget, "revenue"]
        - df.loc[valid_budget, "budget"]
    )
    /
    df.loc[valid_budget, "budget"]
)


# ==========================================
# 15. CREATE HIT / FLOP
# ==========================================

df["movie_success"] = (
    df["revenue"] >= 2 * df["budget"]
).astype(int)


# ==========================================
# 16. KEEP USEFUL COLUMNS
# ==========================================

final_columns = [

    "id",
    "title",

    "release_date",
    "release_year",
    "release_month",
    "release_day",
    "release_weekday",

    "is_friday_release",
    "is_weekend_release",

    "original_language",
    "genre",

    "runtime",

    "production_company",
    "production_country",

    "lead_actor",
    "director",

    "budget",
    "revenue",
    "profit",
    "roi",

    "movie_success"

]


df = df[final_columns]


# ==========================================
# 17. REMOVE INVALID DATA
# ==========================================

df = df.dropna(
    subset=[
        "title",
        "budget",
        "revenue"
    ]
)


# Budget and revenue must be positive

df = df[
    (df["budget"] > 0)
    &
    (df["revenue"] > 0)
]


# Remove impossible runtimes

df = df[
    (df["runtime"].isna())
    |
    (df["runtime"] > 0)
]


# ==========================================
# 18. SAVE PROCESSED DATASET
# ==========================================

output_path = (
    "data/processed/"
    "movie_success_cleaned.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ==========================================
# 19. FINAL INFORMATION
# ==========================================

print("\n================================")
print("PREPROCESSING COMPLETED")
print("================================")

print(
    "Final dataset shape:",
    df.shape
)

print("\nColumns:")
print(df.columns.tolist())

print("\nHIT / FLOP distribution:")

print(
    df["movie_success"]
    .value_counts()
)

print("\nSaved to:")
print(output_path)