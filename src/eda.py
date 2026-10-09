import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

df = pd.read_csv(
    "data/processed/movie_success_cleaned.csv"
)

print("Dataset shape:")
print(df.shape)


# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\n========== DATA TYPES ==========")

print(df.dtypes)


print("\n========== MISSING VALUES ==========")

print(
    df.isnull()
    .sum()
    .sort_values(ascending=False)
)


# ==========================================
# 3. HIT / FLOP DISTRIBUTION
# ==========================================

print("\n========== HIT / FLOP ==========")

print(
    df["movie_success"]
    .value_counts()
)

print("\nPercentage:")

print(
    df["movie_success"]
    .value_counts(normalize=True) * 100
)


# ==========================================
# 4. HIT / FLOP PIE CHART
# ==========================================

success_counts = df["movie_success"].value_counts()

labels = ["FLOP", "HIT"]

plt.figure(figsize=(7, 7))

plt.pie(
    success_counts,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Movie Success Distribution")

plt.show()


# ==========================================
# 5. BUDGET DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["budget"],
    bins=50,
    kde=True
)

plt.title("Movie Budget Distribution")

plt.xlabel("Budget")

plt.ylabel("Number of Movies")

plt.show()


# ==========================================
# 6. REVENUE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["revenue"],
    bins=50,
    kde=True
)

plt.title("Movie Revenue Distribution")

plt.xlabel("Revenue")

plt.ylabel("Number of Movies")

plt.show()


# ==========================================
# 7. BUDGET VS REVENUE
# ==========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="budget",
    y="revenue",
    hue="movie_success",
    alpha=0.5
)

plt.title("Budget vs Revenue")

plt.xlabel("Budget")

plt.ylabel("Revenue")

plt.show()


# ==========================================
# 8. HIT RATE BY GENRE
# ==========================================

genre_df = df[
    ["genre", "movie_success"]
].copy()

# A movie can have multiple genres.
# Take the first genre for this simple analysis.

genre_df["main_genre"] = (
    genre_df["genre"]
    .fillna("Unknown")
    .str.split(",")
    .str[0]
    .str.strip()
)

genre_success = (
    genre_df
    .groupby("main_genre")["movie_success"]
    .mean()
    .sort_values(ascending=False)
)

print("\n========== HIT RATE BY GENRE ==========")

print(
    (genre_success * 100).round(2)
)


# Plot top genres

top_genres = genre_success.head(15)

plt.figure(figsize=(12, 7))

sns.barplot(
    x=top_genres.values * 100,
    y=top_genres.index
)

plt.title("Movie Success Rate by Genre")

plt.xlabel("Hit Rate (%)")

plt.ylabel("Genre")

plt.show()


# ==========================================
# 9. SUCCESS BY RELEASE MONTH
# ==========================================

month_success = (
    df
    .groupby("release_month")["movie_success"]
    .mean()
)

print("\n========== SUCCESS BY RELEASE MONTH ==========")

print(
    (month_success * 100).round(2)
)


plt.figure(figsize=(10, 6))

sns.barplot(
    x=month_success.index,
    y=month_success.values * 100
)

plt.title("Movie Hit Rate by Release Month")

plt.xlabel("Release Month")

plt.ylabel("Hit Rate (%)")

plt.show()


# ==========================================
# 10. FRIDAY RELEASE VS SUCCESS
# ==========================================

friday_success = (
    df
    .groupby("is_friday_release")["movie_success"]
    .mean()
)

print("\n========== FRIDAY RELEASE ==========")

print(
    friday_success * 100
)


# ==========================================
# 11. WEEKEND RELEASE VS SUCCESS
# ==========================================

weekend_success = (
    df
    .groupby("is_weekend_release")["movie_success"]
    .mean()
)

print("\n========== WEEKEND RELEASE ==========")

print(
    weekend_success * 100
)


# ==========================================
# 12. RUNTIME VS SUCCESS
# ==========================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="movie_success",
    y="runtime"
)

plt.title("Runtime vs Movie Success")

plt.xlabel("0 = FLOP, 1 = HIT")

plt.ylabel("Runtime (minutes)")

plt.show()


# ==========================================
# 13. ROI DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["roi"],
    bins=50,
    kde=True
)

plt.title("Movie ROI Distribution")

plt.xlabel("ROI")

plt.ylabel("Number of Movies")

plt.show()


# ==========================================
# 14. NUMERICAL CORRELATION
# ==========================================

numeric_columns = [
    "budget",
    "revenue",
    "profit",
    "roi",
    "runtime",
    "release_year",
    "release_month",
    "movie_success"
]

correlation = (
    df[numeric_columns]
    .corr()
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation")

plt.show()


# ==========================================
# 15. SUMMARY
# ==========================================

print("\n======================================")
print("EDA COMPLETED")
print("======================================")

print("\nTotal movies:", len(df))

print(
    "Average budget:",
    df["budget"].mean()
)

print(
    "Average revenue:",
    df["revenue"].mean()
)

print(
    "Average runtime:",
    df["runtime"].mean()
)

print(
    "Overall hit rate:",
    df["movie_success"].mean() * 100,
    "%"
)