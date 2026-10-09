import pandas as pd

# ==============================
# 1. Load datasets
# ==============================

movies = pd.read_csv(
    "data/raw/movies_metadata.csv",
    low_memory=False
)

credits = pd.read_csv(
    "data/raw/credits.csv",
    low_memory=False
)


# ==============================
# 2. Basic information
# ==============================

print("\n========== MOVIES DATASET ==========")

print("Shape:")
print(movies.shape)

print("\nColumns:")
print(movies.columns.tolist())


print("\n========== CREDITS DATASET ==========")

print("Shape:")
print(credits.shape)

print("\nColumns:")
print(credits.columns.tolist())


# ==============================
# 3. First 5 rows
# ==============================

print("\n========== FIRST 5 MOVIES ==========")

print(movies.head())


print("\n========== FIRST 5 CREDITS ==========")

print(credits.head())


# ==============================
# 4. Data types
# ==============================

print("\n========== MOVIE DATA TYPES ==========")

print(movies.dtypes)


# ==============================
# 5. Missing values
# ==============================

print("\n========== MISSING VALUES: MOVIES ==========")

print(movies.isnull().sum().sort_values(ascending=False))


print("\n========== MISSING VALUES: CREDITS ==========")

print(credits.isnull().sum().sort_values(ascending=False))


# ==============================
# 6. Duplicate rows
# ==============================

print("\n========== DUPLICATES ==========")

print("Duplicate movie rows:", movies.duplicated().sum())
print("Duplicate credit rows:", credits.duplicated().sum())


# ==============================
# 7. Basic statistics
# ==============================

print("\n========== NUMERICAL SUMMARY ==========")

print(movies.describe())