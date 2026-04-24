import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from openai import OpenAI

DATASET_PATH = r"d:\AML project\blood_cell_anomaly_detection.csv"

IMPORTANT_FEATURES = [
    "wbc_count_per_ul",
    "rbc_count_millions_per_ul",
    "hemoglobin_g_dl",
    "hematocrit_pct",
    "platelet_count_per_ul",
    "mcv_fl",
    "mchc_g_dl",
    "cell_diameter_um",
    "nucleus_area_pct",
    "chromatin_density",
    "cytoplasm_ratio",
    "circularity",
    "eccentricity",
    "granularity_score",
    "lobularity_score",
    "membrane_smoothness",
    "stain_intensity",
]

FEATURE_GROUPS = {
    "Blood Counts": [
        "wbc_count_per_ul",
        "rbc_count_millions_per_ul",
        "hemoglobin_g_dl",
        "hematocrit_pct",
        "platelet_count_per_ul",
    ],
    "Red Cell Indices": [
        "mcv_fl",
        "mchc_g_dl",
    ],
    "Cell Morphology": [
        "cell_diameter_um",
        "nucleus_area_pct",
        "chromatin_density",
        "cytoplasm_ratio",
        "circularity",
        "eccentricity",
    ],
    "Cell Texture and Structure": [
        "granularity_score",
        "lobularity_score",
        "membrane_smoothness",
        "stain_intensity",
    ],
}

CUSTOM_RANGES = {
    "wbc_count_per_ul": (0.0, 60000.0),
    "platelet_count_per_ul": (0.0, 90000.0),
}

st.set_page_config(
    page_title="Blood Cell Anomaly Detection",
    page_icon="🩸",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, #1e3a5f 0%, #0f172a 35%, #020617 100%);
        color: white;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .main-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1rem;
        text-align: center;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }

    .risk-ok {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background: rgba(34, 197, 94, 0.18);
        border: 1px solid rgba(34, 197, 94, 0.4);
    }

    .risk-warn {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background: rgba(249, 115, 22, 0.18);
        border: 1px solid rgba(249, 115, 22, 0.4);
    }

    .risk-high {
        padding: 0.75rem 1rem;
        border-radius: 12px;
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Expanded content panel */
    div[data-testid="stExpander"] details > div {
        background: rgba(15, 23, 42, 0.95) !important;
    }

    /* Expander header */
    div[data-testid="stExpander"] summary {
        background-color: rgba(255,255,255,0.06) !important;
        color: #f8fafc !important;
    }

    div[data-testid="stExpander"] summary:hover {
        background-color: rgba(255,255,255,0.10) !important;
    }

    label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Specifically target Streamlit widgets */
.stSlider label,
.stNumberInput label,
.stSelectbox label,
.stTextInput label {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Fix slider label text (important for your case) */
div[data-baseweb="slider"] label {
    color: #FFFFFF !important;
}

/* Fix any leftover muted text */
span, p {
    color: #E5E7EB;
}


/* Sidebar feature list (your highlighted area) */
section[data-testid="stSidebar"] .stMarkdown p {
    color: #000000 !important;
}

/* Slider labels */
div[data-testid="stSlider"] label {
    color: #000000 !important;
}

/* General labels fallback */
label, .stMarkdown, .stText {
    color: #000000 !important;
}

/* Placeholder text inside inputs/selectbox (Quick Preset) */
section[data-testid="stSidebar"] input::placeholder {
    color: #000000 !important;
    opacity: 1 !important;
}

/* Selectbox placeholder / text */
section[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #000000 !important;
}

/* Disabled buttons (Apply Preset, Reset All) */
section[data-testid="stSidebar"] button:disabled {
    color: #000000 !important;
    opacity: 1 !important;   /* remove faded look */
}

/* Button text specifically */
section[data-testid="stSidebar"] button {
    color: #000000 !important;
}

/* Sidebar buttons text -> BLACK */
section[data-testid="stSidebar"] button {
    color: #000000 !important;
}

/* Disabled buttons (important for your case) */
section[data-testid="stSidebar"] button:disabled {
    color: #000000 !important;
    opacity: 1 !important;   /* remove faded look */
}

/* Target inner span (Streamlit wraps text inside span) */
section[data-testid="stSidebar"] button span {
    color: #000000 !important;
}

section[data-testid="stSidebar"] button * {
    color: #000000 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] 
div[data-testid="stWidgetLabel"] p {
    color: #000000 !important;
    opacity: 1 !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data
def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def get_api_key() -> str | None:
    return os.getenv("OPENAI_API_KEY")

def prettify_label(col: str) -> str:
    return (
        col.replace("_", " ")
        .replace("pct", "(%)")
        .replace("ul", "/uL")
        .replace("dl", "g/dL")
        .replace("fl", "fL")
        .title()
    )

def predict_from_features(features: dict) -> dict:
    score = 0.0
    findings = []

    if features.get("wbc_count_per_ul", 0) > 11000:
        score += 1.0
        findings.append("Elevated white blood cell count")

    if features.get("hemoglobin_g_dl", 100) < 12:
        score += 1.0
        findings.append("Low hemoglobin")

    if features.get("platelet_count_per_ul", 999999) < 150000:
        score += 1.0
        findings.append("Low platelet count")

    if features.get("chromatin_density", 0) > 0.7:
        score += 1.0
        findings.append("High chromatin density")

    if features.get("eccentricity", 0) > 0.8:
        score += 0.5
        findings.append("Increased cell eccentricity")

    if features.get("granularity_score", 0) > 0.7:
        score += 0.5
        findings.append("Increased cellular granularity")

    label = "Anomaly" if score >= 2 else "Normal"
    confidence = min(0.55 + (score / 5.0), 0.99)

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "risk_score": round(score, 2),
        "key_findings": findings,
    }

def generate_report(client: OpenAI, features: dict, prediction: dict) -> str:
    prompt = f"""
You are a medical explanation assistant.

These blood and cell morphology features were entered:
{json.dumps(features, indent=2)}

This is the screening output:
{json.dumps(prediction, indent=2)}

Write a clear and concise report.

Rules:
- Use simple language.
- Do not prescribe treatment.
- Do not claim a confirmed diagnosis.
- Explain what the result may suggest.
- Mention the most relevant abnormal-looking indicators.
- Recommend professional clinical review for confirmation.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You generate safe medical explanation reports without prescriptions or definitive diagnosis."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.output_text

def build_feature_chart(features: dict, df: pd.DataFrame) -> go.Figure:
    feature_names = list(features.keys())
    input_values = [features[f] for f in feature_names]
    medians = [float(df[f].median()) if f in df.columns else 0 for f in feature_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[prettify_label(f) for f in feature_names],
        y=input_values,
        name="Input",
    ))
    fig.add_trace(go.Scatter(
        x=[prettify_label(f) for f in feature_names],
        y=medians,
        mode="lines+markers",
        name="Dataset Median",
    ))

    fig.update_layout(
        title="Input vs Dataset Median",
        xaxis_title="Features",
        yaxis_title="Value",
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig

def initialize_state(df: pd.DataFrame, feature_columns: list[str]) -> None:
    if "feature_values" not in st.session_state:
        st.session_state.feature_values = {}
        for col in feature_columns:
            series = df[col].dropna()
            initial_value = float(series.median()) if not series.empty else 0.0

            if col in CUSTOM_RANGES:
                min_allowed, max_allowed = CUSTOM_RANGES[col]
                initial_value = max(min_allowed, min(initial_value, max_allowed))

            st.session_state.feature_values[col] = initial_value

def clamp_custom_range_values(feature_columns: list[str]) -> None:
    for col in feature_columns:
        if col in CUSTOM_RANGES and col in st.session_state.feature_values:
            min_allowed, max_allowed = CUSTOM_RANGES[col]
            current_value = float(st.session_state.feature_values[col])
            st.session_state.feature_values[col] = max(min_allowed, min(current_value, max_allowed))

def reset_features(df: pd.DataFrame, feature_columns: list[str]) -> None:
    for col in feature_columns:
        series = df[col].dropna()
        reset_value = float(series.median()) if not series.empty else 0.0

        if col in CUSTOM_RANGES:
            min_allowed, max_allowed = CUSTOM_RANGES[col]
            reset_value = max(min_allowed, min(reset_value, max_allowed))

        st.session_state.feature_values[col] = reset_value

def load_preset(name: str, df: pd.DataFrame, feature_columns: list[str]) -> None:
    reset_features(df, feature_columns)

    if name == "Typical Normal":
        overrides = {
            "wbc_count_per_ul": 7500,
            "hemoglobin_g_dl": 13.8,
            "platelet_count_per_ul": 250000,
            "chromatin_density": 0.45,
            "eccentricity": 0.55,
            "granularity_score": 0.45,
        }
    elif name == "Possible Anomaly":
        overrides = {
            "wbc_count_per_ul": 14500,
            "hemoglobin_g_dl": 10.2,
            "platelet_count_per_ul": 120000,
            "chromatin_density": 0.82,
            "eccentricity": 0.86,
            "granularity_score": 0.79,
        }
    else:
        overrides = {}

    for k, v in overrides.items():
        if k in st.session_state.feature_values:
            if k in CUSTOM_RANGES:
                min_allowed, max_allowed = CUSTOM_RANGES[k]
                v = max(min_allowed, min(float(v), max_allowed))
            st.session_state.feature_values[k] = v

# Load data
try:
    df = load_dataset(DATASET_PATH)
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

df.columns = [str(col).strip() for col in df.columns]
feature_columns = [col for col in IMPORTANT_FEATURES if col in df.columns]

if not feature_columns:
    st.error("None of the important features were found in the dataset.")
    st.write("Dataset columns:", list(df.columns))
    st.stop()

initialize_state(df, feature_columns)
clamp_custom_range_values(feature_columns)

api_key = get_api_key()
client = OpenAI(api_key=api_key) if api_key else None

# Header
st.title("🩸 Blood Cell Anomaly Detection")
st.caption("Interactive screening dashboard for blood and cell morphology features")

# Sidebar
with st.sidebar:
    st.header("Controls")

    preset = st.selectbox(
        "Quick Preset",
        ["Custom", "Typical Normal", "Possible Anomaly"],
        index=0
    )

    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("Apply Preset", use_container_width=True):
            load_preset(preset, df, feature_columns)
            st.rerun()

    with col_sb2:
        if st.button("Reset All", use_container_width=True):
            reset_features(df, feature_columns)
            st.rerun()

    st.divider()
    st.subheader("Available Features")
    for col in feature_columns:
        st.write(f"• {prettify_label(col)}")

    st.divider()
    if api_key:
        st.success("OPENAI_API_KEY loaded")
    else:
        st.warning("OPENAI_API_KEY not found. Prediction works, report generation is disabled.")

# Build feature input dictionary
features = {}

# Live preview score
for col in feature_columns:
    features[col] = st.session_state.feature_values[col]

live_prediction = predict_from_features(features)
risk_ratio = min(live_prediction["risk_score"] / 4.0, 1.0)

# Top row summary
top1, top2, top3 = st.columns(3)
with top1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>Live Label</h4>
            <h2>{live_prediction["label"]}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>Risk Score</h4>
            <h2>{live_prediction["risk_score"]}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top3:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4>Confidence</h4>
            <h2>{int(live_prediction["confidence"] * 100)}%</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.progress(risk_ratio)

if live_prediction["risk_score"] < 1.5:
    st.markdown(
        '<div class="risk-ok">Current inputs look lower-risk based on the simple screening rules.</div>',
        unsafe_allow_html=True
    )
elif live_prediction["risk_score"] < 2.5:
    st.markdown(
        '<div class="risk-warn">Current inputs show some warning signals. Review before analyzing.</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="risk-high">Current inputs suggest a higher anomaly likelihood in this screening view.</div>',
        unsafe_allow_html=True
    )

# Input tabs
tab1, tab2, tab3 = st.tabs(["Feature Input", "Preview", "Visualization"])

with tab1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Enter Clinical and Morphology Features")

    for group_name, group_features in FEATURE_GROUPS.items():
        available_group_features = [f for f in group_features if f in feature_columns]
        if not available_group_features:
            continue

        with st.expander(group_name, expanded=True):
            cols = st.columns(2)

            for i, col in enumerate(available_group_features):
                series = df[col].dropna()

                dataset_min = float(series.min()) if not series.empty else 0.0
                dataset_max = float(series.max()) if not series.empty else 100.0

                min_value = dataset_min
                max_value = dataset_max

                if col in CUSTOM_RANGES:
                    min_value, max_value = CUSTOM_RANGES[col]

                current_value = float(st.session_state.feature_values.get(col, 0.0))
                current_value = max(min_value, min(current_value, max_value))

                step_value = (max_value - min_value) / 100 if max_value > min_value else 0.1
                step_value = max(step_value, 0.0001)

                with cols[i % 2]:
                    value = st.slider(
                        label=prettify_label(col),
                        min_value=float(min_value),
                        max_value=float(max_value),
                        value=float(current_value),
                        step=float(step_value),
                        help=f"Allowed range: {min_value:.0f} to {max_value:.0f}",
                    )

                    st.session_state.feature_values[col] = value
                    features[col] = value

    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Input Preview")
    preview_df = pd.DataFrame([st.session_state.feature_values])
    st.dataframe(preview_df, use_container_width=True)

    current_preview_prediction = predict_from_features(st.session_state.feature_values)

    if current_preview_prediction["key_findings"]:
        st.subheader("Live Rule-Based Findings")
        for finding in current_preview_prediction["key_findings"]:
            st.write(f"• {finding}")
    else:
        st.info("No strong rule-based findings from the current inputs.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Feature Comparison")
    fig = build_feature_chart(st.session_state.feature_values, df)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Analyze section
st.markdown('<div class="main-card">', unsafe_allow_html=True)
left, right = st.columns([1, 1])

with left:
    analyze = st.button("Analyze Sample", use_container_width=True, type="primary")

with right:
    generate_ai = st.toggle("Generate AI explanation report", value=True)

st.markdown('</div>', unsafe_allow_html=True)

if analyze:
    prediction = predict_from_features(st.session_state.feature_values)

    report = None
    if generate_ai and client is not None:
        try:
            with st.spinner("Generating explanation report..."):
                report = generate_report(client, st.session_state.feature_values, prediction)
        except Exception as e:
            st.warning(f"Prediction completed, but report generation failed: {e}")

    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("Prediction Result")

    r1, r2, r3 = st.columns(3)
    r1.metric("Classification", prediction["label"])
    r2.metric("Confidence", f'{int(prediction["confidence"] * 100)}%')
    r3.metric("Risk Score", prediction["risk_score"])

    st.write("### Key Findings")
    if prediction["key_findings"]:
        for finding in prediction["key_findings"]:
            st.write(f"• {finding}")
    else:
        st.write("No major rule-based abnormalities detected from the current inputs.")

    st.write("### Structured Output")
    st.json(prediction)

    if report:
        st.write("### AI Explanation Report")
        st.write(report)
    elif generate_ai and client is None:
        st.info("Set OPENAI_API_KEY to enable AI explanation reports.")

    st.caption("This tool is a screening interface and does not provide a confirmed diagnosis.")
    st.markdown('</div>', unsafe_allow_html=True)