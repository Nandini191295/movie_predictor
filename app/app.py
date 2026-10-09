
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="CinePredict | Movie Analytics",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "movie_success_model.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "movie_success_ml_ready.csv"

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #151b2e);
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
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #c4c9dd;
    font-size: 16px;
}
.metric-card {
    background: #171f34;
    border: 1px solid #303a55;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
}
.result-hit {
    background: #12382f;
    border: 1px solid #248568;
    border-radius: 16px;
    padding: 24px;
}
.result-flop {
    background: #3b202b;
    border: 1px solid #a34b60;
    border-radius: 16px;
    padding: 24px;
}
.small-note {
    color: #aab3cb;
    font-size: 13px;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD FILES ----------------
if not MODEL_PATH.exists() or not DATA_PATH.exists():
    st.error(
        "Required files are missing. Please check the models and "
        "data/processed folders and run src/train.py first."
    )
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

model = load_model(str(MODEL_PATH))
data = load_data(str(DATA_PATH))

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>🎬 CinePredict</h1>
    <p>Movie Success Prediction &amp; Box Office Analytics</p>
    <p>Explore how movie characteristics relate to historical box-office outcomes.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎥 Movie Details")
st.sidebar.caption("Enter the movie information below.")

budget = st.sidebar.number_input(
    "Production Budget",
    min_value=1.0,
    value=10000000.0,
    step=1000000.0
)

runtime = st.sidebar.slider(
    "Runtime (minutes)", 1, 300, 120
)

release_year = st.sidebar.number_input(
    "Release Year", 1900, 2100, 2027
)

release_month = st.sidebar.selectbox(
    "Release Month",
    list(range(1, 13)),
    format_func=lambda m: pd.Timestamp(2000, m, 1).strftime("%B")
)

release_day = st.sidebar.slider(
    "Release Day", 1, 31, 15
)

genre_options = sorted(
    data["genre"].dropna().astype(str).unique().tolist()
)
language_options = sorted(
    data["original_language"].dropna().astype(str).unique().tolist()
)
country_options = sorted(
    data["production_country"].dropna().astype(str).unique().tolist()
)

genre = st.sidebar.selectbox("Genre", genre_options)
language = st.sidebar.selectbox("Original Language", language_options)
country = st.sidebar.selectbox("Production Country", country_options)

is_friday = st.sidebar.checkbox("Friday Release")
is_weekend = st.sidebar.checkbox("Weekend Release")

season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
}
release_season = season_map[release_month]

# Historical rates are neutral placeholders for this UI step.
# We will improve these in the next step.
director_hit_rate = 0.5
lead_actor_hit_rate = 0.5

# ---------------- MAIN DASHBOARD ----------------
left, right = st.columns([1.3, 1], gap="large")

with left:
    st.subheader("📊 Movie Overview")
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.write("**Selected genre:**", genre)
    st.write("**Release year:**", int(release_year))
    st.write("**Language:**", language)
    st.write("**Production country:**", country)
    st.write("**Release season:**", release_season)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("💰 Production Details")
    c1, c2 = st.columns(2)
    c1.metric("Budget", f"${budget:,.0f}")
    c2.metric("Runtime", f"{runtime} min")

with right:
    st.subheader("🎯 Prediction")

    if st.button("Predict Movie Success", type="primary"):
        input_data = pd.DataFrame([{
            "budget": budget,
            "runtime": runtime,
            "release_year": int(release_year),
            "release_month": int(release_month),
            "release_day": int(release_day),
            "is_friday_release": int(is_friday),
            "is_weekend_release": int(is_weekend),
            "director_hit_rate": director_hit_rate,
            "lead_actor_hit_rate": lead_actor_hit_rate,
            "genre": genre,
            "original_language": language,
            "production_country": country,
            "release_season": release_season
        }])

        prediction = int(model.predict(input_data)[0])

        if prediction == 1:
            st.markdown("""
            <div class="result-hit">
                <h2>🎉 HIT Prediction</h2>
                <p>The model predicts the movie will meet the project's HIT label.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-flop">
                <h2>📉 FLOP Prediction</h2>
                <p>The model predicts the movie will not meet the project's HIT label.</p>
            </div>
            """, unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = list(model.classes_)
            if 1 in classes:
                hit_probability = float(probabilities[classes.index(1)])
                st.metric("Model-estimated HIT probability",
                          f"{hit_probability:.1%}")
                st.progress(hit_probability)

        st.caption(
            "This is an educational estimate, not a guarantee of box-office results."
        )
    else:
        st.info("Enter movie details in the sidebar, then click Predict.")

st.divider()
st.subheader("📌 About this project")
st.write(
    "CinePredict is a machine-learning project that classifies movies using "
    "historical metadata such as budget, runtime, genre, language and release timing."
)
st.markdown(
    '<p class="small-note">HIT is defined by this project as revenue at least '
    'twice the budget in the source dataset. Actual commercial success also '
    'depends on marketing, distribution, reviews and other factors.</p>',
    unsafe_allow_html=True
)
