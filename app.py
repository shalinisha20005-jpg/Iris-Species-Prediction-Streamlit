
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json

st.markdown("""
<style>
/* Sidebar button visibility */
section[data-testid="stSidebar"] .stButton > button {
    background-color: #f4f5e9 !important;
    color: #183e2a !important;
    border: 2px solid #d5e4cc !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 45px !important;
}

section[data-testid="stSidebar"] .stButton > button p,
section[data-testid="stSidebar"] .stButton > button span,
section[data-testid="stSidebar"] .stButton > button div {
    color: #183e2a !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #dce9cf !important;
    color: #183e2a !important;
    border-color: #a9c49b !important;
}
</style>
""", unsafe_allow_html=True)

# PAGE CONFIG
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BOTANICAL DESIGN
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background: #faf8ef;
    color: #26352b;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] {
    background: #faf8ef;
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#183e2a,#315c3b);
}
section[data-testid="stSidebar"] * {
    color: #fffef5 !important;
}
section[data-testid="stSidebar"] .stButton button {
    background: #f4f5e9;
    color: #183e2a !important;
    border-radius: 12px;
    border: none;
    font-weight: 700;
}
.hero {
    background: linear-gradient(110deg,#edf2df,#fffef8,#f2f0df);
    border: 1px solid #e1e5d3;
    border-radius: 24px;
    padding: 30px;
    text-align: center;
    margin-bottom: 20px;
}
.hero h1 {
    font-family: 'Playfair Display',serif;
    color: #183e2a;
    font-size: 2.8rem;
}
.hero p {
    color: #55734c;
}
.panel {
    background: #fffef9;
    border: 1px solid #e3e5d3;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 5px 15px rgba(45,65,39,.05);
}
.section-title {
    font-family: 'Playfair Display',serif;
    color: #214c2c;
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 12px;
}
.value-box {
    background: #f1f5e9;
    border: 1px solid #e5eadb;
    border-radius: 10px;
    padding: 12px;
    margin: 7px 0;
    display: flex;
    justify-content: space-between;
}
.value-number {
    color: #234b2e;
    font-weight: bold;
}
.result-name {
    font-family: 'Playfair Display',serif;
    color: #214c2c;
    font-size: 2rem;
    font-weight: bold;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#4e853f,#214f30);
    color: white;
    border: none;
    border-radius: 14px;
    min-height: 52px;
    font-size: 1.05rem;
    font-weight: bold;
}
div[data-testid="stProgress"] > div > div {
    background: #39743a;
}
div[data-testid="stImage"] img {
    border-radius: 14px;
}
.stExpander {
    background: #fffef9;
    border-radius: 12px;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)



# LOAD MODEL
def load_model(format_type="joblib"):
    try:
        if format_type == "joblib":
            model_path = "models/iris_model.joblib"
            model = joblib.load(model_path)

        elif format_type == "pickle":
            model_path = "models/iris_model.pickle"
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        else:
            raise ValueError("Unsupported model format")

        return model

    except Exception as e:
        st.error(f"Model loading error ({format_type}): {e}")
        return None

# LOAD MODEL INFORMATION
def load_model_info():
    try:
        with open("models/model_info.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Metadata error: {e}")
        return None


# LOAD FEATURE RANGES
def load_feature_ranges():
    try:
        with open("models/feature_ranges.json", "r") as f:
            return json.load(f)
    except Exception:
        return {
            "sepal length (cm)": [4.3, 7.9],
            "sepal width (cm)": [2.0, 4.4],
            "petal length (cm)": [1.0, 6.9],
            "petal width (cm)": [0.1, 2.5]
        }


# INITIALIZE MODEL AND SESSION STATE
model_info = load_model_info()
feature_ranges = load_feature_ranges()

if "active_model_format" not in st.session_state:
    st.session_state.active_model_format = "joblib"

if "active_model" not in st.session_state:
    st.session_state.active_model = load_model("joblib")

if "model_format" not in st.session_state:
    st.session_state.model_format = "joblib"


# FLOWER IMAGES
FLOWERS = {
    "setosa": {
    "name": "Iris Setosa",
    "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Iris_setosa.JPG"
    },
    "versicolor": {
        "name": "Iris Versicolor",
        "image": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"
    },
    "virginica": {
        "name": "Iris Virginica",
        "image": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }
}


def get_species_key(name):
    name = str(name).lower()
    for key in FLOWERS:
        if key in name:
            return key
    return None


# SIDEBAR
with st.sidebar:
    st.markdown("<h1 style='text-align:center'>🌸</h1>",
                unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>Iris Classifier</h2>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align:center'>ML Flower Prediction</p>",
                unsafe_allow_html=True)

    st.divider()
    st.subheader("⚙️ Settings")

    
    model_format = st.radio(
        "Model Format",
        ["joblib", "pickle"],
        key="model_format"
    )

    if st.button("↻ Reload Model", use_container_width=True):
        with st.spinner(f"Loading {model_format} model..."):
            st.session_state.active_model = load_model(model_format)
            st.session_state.active_model_format = model_format

        if st.session_state.active_model is not None:
            st.success(f"Model successfully loaded from {model_format} format!")
        else:
            st.error(f"Could not load the {model_format} model.")

    if st.session_state.active_model is not None:
        st.caption(
            f"Active model: {st.session_state.active_model_format}"
        )

# HEADER
st.markdown("""
<div class="hero">
    <h1>🌸 Iris Flower Classification</h1>
    <p style="font-size:20px">
        Machine Learning • Flower Species Prediction
    </p>
    <p>
        This app predicts the species of an Iris flower based on
        its measurements using a machine learning model.
        <br>
        Adjust the sliders and discover your flower!
    </p>
</div>
""", unsafe_allow_html=True)


# INPUT AND RESULT COLUMNS
input_col, result_col = st.columns([1, 1.08], gap="medium")

with input_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🌿 Input Features</div>',
        unsafe_allow_html=True
    )

    sepal_length = st.slider(
        "🍃 Sepal Length (cm)",
        float(feature_ranges["sepal length (cm)"][0]),
        float(feature_ranges["sepal length (cm)"][1]),
        5.8, 0.1
    )

    sepal_width = st.slider(
        "🍃 Sepal Width (cm)",
        float(feature_ranges["sepal width (cm)"][0]),
        float(feature_ranges["sepal width (cm)"][1]),
        3.0, 0.1
    )

    petal_length = st.slider(
        "🌱 Petal Length (cm)",
        float(feature_ranges["petal length (cm)"][0]),
        float(feature_ranges["petal length (cm)"][1]),
        4.0, 0.1
    )

    petal_width = st.slider(
        "🌱 Petal Width (cm)",
        float(feature_ranges["petal width (cm)"][0]),
        float(feature_ranges["petal width (cm)"][1]),
        1.2, 0.1
    )

    st.markdown('</div>', unsafe_allow_html=True)

    # CURRENT VALUES
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📊 Current Values</div>',
        unsafe_allow_html=True
    )

    values = [
        ("Sepal Length", sepal_length),
        ("Sepal Width", sepal_width),
        ("Petal Length", petal_length),
        ("Petal Width", petal_width)
    ]

    for label, value in values:
        st.markdown(
            f"""
            <div class="value-box">
                <span>{label}</span>
                <span class="value-number">{value:.2f} cm</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)


# PREDICTION RESULT PANEL
with result_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">🏆 Prediction Result</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.get("iris_result")

    if result:
        key = get_species_key(result["class"])

        if key:
            st.image(
                FLOWERS[key]["image"],
                use_container_width=True
            )

        st.caption("Predicted Species")
        st.markdown(
            f'<div class="result-name">{result["class"]}</div>',
            unsafe_allow_html=True
        )

        st.caption("🎯 Prediction Confidence")
        st.markdown(
            f'<div class="result-name">{result["confidence"]:.2f}%</div>',
            unsafe_allow_html=True
        )

        st.progress(result["confidence"] / 100)

    else:
        st.markdown(
            "<div style='font-size:100px;text-align:center'>🌷</div>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<h3 style='text-align:center;color:#214c2c'>"
            "Your flower awaits</h3>",
            unsafe_allow_html=True
        )
        st.write(
            "Adjust the measurements and click Predict Species "
            "to identify your Iris flower."
        )

    st.markdown('</div>', unsafe_allow_html=True)


# PREDICTION BUTTON
input_features = np.array([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])

predict_clicked = st.button(
    "🎯 Predict Species",
    type="primary",
    use_container_width=True
)

quick_predict = st.session_state.pop("make_prediction", False)

if predict_clicked or quick_predict:
    active_model = st.session_state.active_model

    if active_model is not None and model_info is not None:
        try:
            prediction = active_model.predict(input_features)
            prediction_proba = active_model.predict_proba(input_features)[0]

            predicted_class = model_info["target_names"][prediction[0]]
            confidence = float(np.max(prediction_proba) * 100)

            st.session_state.iris_result = {
                "class": predicted_class,
                "confidence": confidence,
                "probabilities": prediction_proba.tolist()
            }

            st.rerun()

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Model could not be loaded. Check your model files.")


# CONFIDENCE SCORES
result = st.session_state.get("iris_result")

if result:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-title">📈 Confidence by Species</div>',
        unsafe_allow_html=True
    )

    for i, probability in enumerate(result["probabilities"]):
        species = model_info["target_names"][i]
        st.markdown(
            f"**{species} — {probability * 100:.2f}%**"
        )
        st.progress(float(probability))

    st.markdown('</div>', unsafe_allow_html=True)


# THREE IRIS FLOWER EXAMPLES
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">🌿 Iris Flower Species Examples</div>',
    unsafe_allow_html=True
)

cols = st.columns(3)

for col, key in zip(cols, ["setosa", "versicolor", "virginica"]):
    with col:
        st.image(FLOWERS[key]["image"], use_container_width=True)
        st.markdown(
            f"<p style='text-align:center;color:#214c2c;"
            f"font-weight:bold'>{FLOWERS[key]['name']}</p>",
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)


# DATASET INFORMATION
with st.expander(
    "📚 About the Iris Dataset",
    expanded=st.session_state.get("show_dataset_info", False)
):
    st.session_state.show_dataset_info = False

    st.markdown("""
    The Iris dataset is a classic machine learning dataset.

    - **150 samples** — 50 samples per species
    - **4 features** — Sepal length, sepal width,
      petal length and petal width
    - **3 species** — Iris Setosa, Iris Versicolor
      and Iris Virginica

    This application uses your saved classification model
    to predict the species from the four measurements.
    """)


# FOOTER
st.markdown("""
<p style="text-align:center;color:#718071;padding:20px">
    🌿 Made with Streamlit and Scikit-learn
</p>
""", unsafe_allow_html=True)
