
import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CinePredict | Model Analytics",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" / "processed" / "movie_success_ml_ready.csv"
)

RESULTS_PATH = BASE_DIR / "models" / "model_comparison.csv"

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #171c30);
    color: #f5f7ff;
}
.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}
[data-testid="stSidebar"] {
    background-color: #111729;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(110deg, #242b48, #342044);
    border: 1px solid #414765;
    margin-bottom: 24px;
}
.hero h1 {
    font-size: 36px;
}
.hero p {
    color: #c4c9dd;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🎬 CinePredict Analytics</h1>
    <p>Movie data insights and machine-learning model evaluation</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("📊 Dashboard")
st.sidebar.write("Explore the dataset and model results.")

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------
@st.cache_data
def load_dataset(path):
    return pd.read_csv(path)

if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH}")
    st.stop()

try:
    df = load_dataset(str(DATA_PATH))
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

# Clean column names to prevent accidental whitespace issues
df.columns = df.columns.astype(str).str.strip()

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------
st.header("📁 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Movies", f"{len(df):,}")
col2.metric("Total Columns", f"{len(df.columns):,}")

if "release_year" in df.columns:
    years = pd.to_numeric(df["release_year"], errors="coerce").dropna()

    if not years.empty:
        year_range = (
            f"{int(years.min())} – {int(years.max())}"
        )
    else:
        year_range = "Unavailable"
else:
    year_range = "Unavailable"

col3.metric("Release Year Range", year_range)

st.divider()

# --------------------------------------------------
# HIT VS FLOP DISTRIBUTION
# --------------------------------------------------
st.header("🎯 HIT vs FLOP Distribution")

if "movie_success" in df.columns:
    labels = df["movie_success"].map({
        1: "HIT",
        0: "FLOP"
    }).fillna("Unknown")

    class_counts = labels.value_counts()

    left, right = st.columns([1, 1])

    with left:
        st.bar_chart(class_counts)

    with right:
        st.dataframe(
            class_counts.rename_axis("Class")
            .reset_index(name="Number of Movies"),
            use_container_width=True,
            hide_index=True
        )
else:
    st.warning(
        "The movie_success column was not found. "
        "HIT/FLOP distribution cannot be displayed."
    )

st.divider()

# --------------------------------------------------
# MOVIES BY RELEASE YEAR
# --------------------------------------------------
st.header("📅 Movies by Release Year")

if "release_year" in df.columns:
    df["release_year"] = pd.to_numeric(
        df["release_year"], errors="coerce"
    )

    year_counts = (
        df.dropna(subset=["release_year"])
          .groupby("release_year")
          .size()
          .sort_index()
    )

    if not year_counts.empty:
        st.line_chart(year_counts)
    else:
        st.info("No valid release-year values are available.")
else:
    st.info(
        "Release-year chart is unavailable because the "
        "'release_year' column does not exist in this dataset."
    )
    st.caption("Available columns:")
    st.write(df.columns.tolist())

st.divider()

# --------------------------------------------------
# MEDIAN BUDGET BY OUTCOME
# --------------------------------------------------
st.header("💰 Budget by Movie Outcome")

if "budget" in df.columns and "movie_success" in df.columns:
    budget_data = df[["budget", "movie_success"]].copy()

    budget_data["budget"] = pd.to_numeric(
        budget_data["budget"], errors="coerce"
    )

    budget_data = budget_data.dropna(
        subset=["budget", "movie_success"]
    )

    budget_summary = budget_data.groupby(
        "movie_success"
    )["budget"].median()

    budget_summary.index = [
        "HIT" if int(label) == 1 else "FLOP"
        for label in budget_summary.index
    ]

    if not budget_summary.empty:
        st.bar_chart(budget_summary)
        st.caption(
            "Comparison of median production budgets. "
            "This does not establish that budget causes success."
        )
    else:
        st.info("No valid budget data is available.")
else:
    st.info("Budget comparison requires budget and movie_success columns.")

st.divider()

# --------------------------------------------------
# MODEL PERFORMANCE COMPARISON
# --------------------------------------------------
st.header("🤖 Machine Learning Model Comparison")

if RESULTS_PATH.exists():
    try:
        results = pd.read_csv(RESULTS_PATH)
        results.columns = results.columns.astype(str).str.strip()

        if results.empty:
            st.warning("The model comparison file is empty.")
        else:
            st.subheader("Training Evaluation Results")
            st.dataframe(
                results,
                use_container_width=True,
                hide_index=True
            )

            # Recognize common metric column names
            metric_aliases = {
                "Accuracy": ["Accuracy", "accuracy"],
                "Precision": ["Precision", "precision"],
                "Recall": ["Recall", "recall"],
                "F1-score": [
                    "F1", "F1-score", "f1", "f1_score",
                    "F1 Score", "f1-score"
                ],
                "ROC-AUC": [
                    "ROC-AUC", "ROC_AUC", "roc_auc", "roc-auc"
                ]
            }

            available_metrics = {}

            for display_name, possible_names in metric_aliases.items():
                for name in possible_names:
                    if name in results.columns:
                        available_metrics[display_name] = name
                        break

            # Find model-name column
            possible_model_columns = [
                "Model", "model", "Classifier", "classifier",
                "Algorithm", "algorithm"
            ]

            model_column = next(
                (
                    name for name in possible_model_columns
                    if name in results.columns
                ),
                results.columns[0]
            )

            if available_metrics:
                chart_data = results[
                    [model_column] + list(available_metrics.values())
                ].copy()

                # Convert metrics to numeric for charting
                for column in available_metrics.values():
                    chart_data[column] = pd.to_numeric(
                        chart_data[column], errors="coerce"
                    )

                chart_data = chart_data.dropna(
                    subset=list(available_metrics.values()),
                    how="all"
                )

                chart_data = chart_data.set_index(model_column)
                chart_data.columns = list(available_metrics.keys())

                if not chart_data.empty:
                    st.subheader("📈 Compare Model Metrics")
                    st.bar_chart(chart_data)

                else:
                    st.info("No numeric model metrics are available.")
            else:
                st.warning(
                    "The CSV was found, but recognized metric columns "
                    "were not detected. Check the table above."
                )

    except Exception as error:
        st.error(f"Could not read model comparison results: {error}")

else:
    st.warning(
        "Model comparison file not found. Expected location:"
    )
    st.code(str(RESULTS_PATH))
    st.info("Run python src/train.py to generate the comparison file.")

st.divider()

# --------------------------------------------------
# DATASET PREVIEW
# --------------------------------------------------
with st.expander("🔎 View Dataset Sample"):
    st.dataframe(
        df.head(20),
        use_container_width=True,
        hide_index=True
    )

with st.expander("🧾 View Available Dataset Columns"):
    st.write(df.columns.tolist())

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.caption(
    "CinePredict is an educational machine-learning project. "
    "HIT is defined by this project as revenue at least twice the budget. "
    "Predictions and historical patterns are not guarantees of actual "
    "commercial success."
)
