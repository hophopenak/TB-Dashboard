import os
import pickle
import joblib
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="TB Risk Prediction & Explainable AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "Age",
    "Gender",
    "Distance_to_Clinic",
    "Ambient_Temperature",
    "Ventilation",
    "Symptom_Duration",
    "Productive_Cough",
    "Dyspnea",
    "Chest_Pain",
    "Fever",
    "History_of_Contact",
]

# Research results from notebook TB fix(1).html
METRICS = {
    "Accuracy": 0.9822,
    "Precision": 0.9194,
    "Recall": 0.9500,
    "F1-Score": 0.9344,
    "ROC-AUC": 0.9985,
    "Best CV F1": 0.9568,
}

CONFUSION_MATRIX = np.array([
    [385, 5],
    [3, 57],
])

SHAP_GLOBAL = pd.DataFrame({
    "Feature": [
        "Symptom_Duration",
        "Chest_Pain",
        "History_of_Contact",
        "Ventilation",
        "Distance_to_Clinic",
        "Age",
        "Dyspnea",
        "Fever",
        "Gender",
        "Productive_Cough",
        "Ambient_Temperature",
    ],
    "Mean_Abs_SHAP": [
        0.286042,
        0.130508,
        0.064230,
        0.044894,
        0.013174,
        0.011518,
        0.008033,
        0.006066,
        0.002775,
        0.000960,
        0.000344,
    ],
})

RF_IMPORTANCE = pd.DataFrame({
    "Feature": [
        "Symptom_Duration",
        "Chest_Pain",
        "History_of_Contact",
        "Ventilation",
        "Age",
        "Distance_to_Clinic",
        "Dyspnea",
        "Fever",
        "Gender",
        "Productive_Cough",
        "Ambient_Temperature",
    ],
    "RF_Importance": [
        0.471097,
        0.268506,
        0.116722,
        0.051413,
        0.031082,
        0.030428,
        0.013127,
        0.009160,
        0.005724,
        0.001760,
        0.000980,
    ],
})

BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "class_weight": "balanced_subsample",
    "random_state": 42,
}

# ============================================================
# STYLE
# ============================================================
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(37, 99, 235, .10), transparent 28%),
            radial-gradient(circle at 90% 10%, rgba(16, 185, 129, .08), transparent 28%),
            #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, .90);
    }

    [data-testid="stSidebar"] {
        background: #0f172a;
        border-right: 1px solid #1e293b;
    }

    [data-testid="stSidebar"] * {
        color: #dbeafe;
    }

    .hero {
        padding: 26px 30px;
        border: 1px solid #24324a;
        border-radius: 22px;
        background: linear-gradient(135deg, #111c33 0%, #0f172a 55%, #102238 100%);
        box-shadow: 0 16px 45px rgba(0,0,0,.22);
        margin-bottom: 20px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.15rem;
        color: #f8fafc;
    }

    .hero p {
        margin: 8px 0 0 0;
        color: #94a3b8;
        font-size: 1rem;
    }

    .card {
        background: #111827;
        border: 1px solid #263449;
        border-radius: 18px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 10px 30px rgba(0,0,0,.16);
    }

    .metric-title {
        color: #94a3b8;
        font-size: .84rem;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 1.65rem;
        font-weight: 700;
    }

    .risk-high {
        background: linear-gradient(135deg, #3b1117, #241018);
        border: 1px solid #7f1d1d;
        border-radius: 18px;
        padding: 22px;
    }

    .risk-low {
        background: linear-gradient(135deg, #06261d, #0c1f1a);
        border: 1px solid #065f46;
        border-radius: 18px;
        padding: 22px;
    }

    .risk-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: #f8fafc;
    }

    .risk-sub {
        color: #cbd5e1;
        margin-top: 4px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 14px 0 8px 0;
    }

    .small-note {
        color: #94a3b8;
        font-size: .82rem;
    }

    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #263449;
        padding: 14px 16px;
        border-radius: 16px;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #111827;
        border-color: #334155;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid #334155;
        background: #172033;
        color: #e2e8f0;
    }

    .stButton > button:hover {
        border-color: #60a5fa;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Folder structure:
# TB_Dashboard/
# ├── dashboard.py
# └── model/
#     ├── Best_RandomForest_TB.pkl
#     └── TB_Imputer.pkl
#
# ============================================================
# MODEL LOADER
# ============================================================
# Research model file
# Use the dashboard.py location so the path does not depend
# on the folder from which the "streamlit run" command is executed.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_PATH = os.path.join(MODEL_DIR, "Best_RandomForest_TB.pkl")
IMPUTER_PATH = os.path.join(MODEL_DIR, "TB_Imputer.pkl")


@st.cache_resource
def load_model():
    """Load the Random Forest model using joblib, with pickle as a fallback."""
    if not os.path.isfile(MODEL_PATH):
        return None, MODEL_PATH

    errors = []

    # Prefer joblib because the research model is usually saved
    # using joblib.dump().
    try:
        loaded = joblib.load(MODEL_PATH)
        estimator = getattr(loaded, "best_estimator_", loaded)
        return estimator, MODEL_PATH
    except Exception as exc:
        errors.append(f"joblib: {exc}")

    # Fallback if the file was created using pickle.dump().
    try:
        with open(MODEL_PATH, "rb") as f:
            loaded = pickle.load(f)
        estimator = getattr(loaded, "best_estimator_", loaded)
        return estimator, MODEL_PATH
    except Exception as exc:
        errors.append(f"pickle: {exc}")

    st.error(
        "Failed to load the model. The file was found, but it could not be "
        "deserialized.\n\n"
        + "\n".join(errors)
    )
    st.caption(
        "If 'STACK_GLOBAL requires str' appears, the PKL file may have been "
        "created with a different environment/library or the model file may be "
        "corrupted. The model needs to be re-saved using joblib.dump()."
    )
    return None, MODEL_PATH


@st.cache_resource
def load_imputer():
    """Load the research imputer using joblib, with pickle as a fallback."""
    if not os.path.isfile(IMPUTER_PATH):
        return None

    errors = []

    try:
        return joblib.load(IMPUTER_PATH)
    except Exception as exc:
        errors.append(f"joblib: {exc}")

    try:
        with open(IMPUTER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        errors.append(f"pickle: {exc}")

    st.error(
        "Failed to load the imputer. The file was found, but it could not be "
        "deserialized.\n\n"
        + "\n".join(errors)
    )
    return None


model, model_path = load_model()
imputer = load_imputer()

# Diagnostic information to verify the files used by the dashboard.
if model is not None:
    try:
        MODEL_TYPE = type(model).__name__
    except Exception:
        MODEL_TYPE = "Unknown"
else:
    MODEL_TYPE = None

# ============================================================
# HELPERS
# ============================================================
def clean_feature_name(name):
    return str(name).strip()


def prepare_input(values):
    """Create a DataFrame using the research feature order."""
    df = pd.DataFrame([values])
    df = df[FEATURES].copy()

    # If the imputer is available, use the research imputer.
    # Dashboard inputs are normally complete, so there should be no missing values.
    if imputer is not None:
        try:
            df = pd.DataFrame(
                imputer.transform(df),
                columns=FEATURES,
                index=df.index,
            )
        except Exception:
            # If the imputer expects different columns/names, the complete input
            # can still be passed to the model.
            pass

    # Match column names with the model's feature_names_in_ if available.
    if hasattr(model, "feature_names_in_"):
        model_names = list(model.feature_names_in_)
        stripped = [clean_feature_name(x) for x in model_names]
        if set(stripped) == set(FEATURES):
            rename_map = dict(zip(FEATURES, model_names))
            df = df.rename(columns=rename_map)
            df = df[model_names]

    return df


def predict_probability(input_df):
    if model is None:
        return None, None

    pred = int(model.predict(input_df)[0])

    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(input_df)[0, 1])
    else:
        proba = None

    return pred, proba


def get_tree_estimator():
    """Get the Random Forest estimator for TreeExplainer."""
    if model is None:
        return None

    if hasattr(model, "best_estimator_"):
        return model.best_estimator_

    # Pipeline sederhana
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "estimators_"):
                return step

    if hasattr(model, "estimators_"):
        return model

    return None


def calculate_individual_shap(input_df):
    """
    Menghitung SHAP untuk satu observasi.
    Compatible with several SHAP output formats for classifiers.
    """
    try:
        import shap
    except ImportError:
        return None, None

    estimator = get_tree_estimator()
    if estimator is None:
        return None, None

    try:
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(input_df)

        # SHAP lama: list [class0, class1]
        if isinstance(shap_values, list):
            values = np.asarray(shap_values[1])[0]
        else:
            arr = np.asarray(shap_values)

            # Format baru bisa (n, features, classes)
            if arr.ndim == 3:
                values = arr[0, :, 1]
            elif arr.ndim == 2:
                values = arr[0]
            else:
                return None, None

        base_values = getattr(explainer, "expected_value", None)

        if isinstance(base_values, (list, np.ndarray)):
            base_arr = np.asarray(base_values).reshape(-1)
            base_value = float(base_arr[-1])
        elif base_values is not None:
            base_value = float(base_values)
        else:
            base_value = None

        feature_names = [clean_feature_name(x) for x in input_df.columns]

        result = pd.DataFrame({
            "Feature": feature_names,
            "SHAP": values.astype(float),
        }).sort_values("SHAP")

        return result, base_value

    except Exception as exc:
        st.warning(f"Individual SHAP values could not be calculated: {exc}")
        return None, None


def feature_label(feature):
    labels = {
        "Age": "Age",
        "Gender": "Gender",
        "Distance_to_Clinic": "Distance to Clinic",
        "Ambient_Temperature": "Ambient Temperature",
        "Ventilation": "Ventilation",
        "Symptom_Duration": "Symptom Duration",
        "Productive_Cough": "Productive Cough",
        "Dyspnea": "Dyspnea",
        "Chest_Pain": "Chest Pain",
        "Fever": "Fever",
        "History_of_Contact": "History of Contact",
    }
    return labels.get(feature, feature)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>🫁 TB Risk Prediction & Explainable AI</h1>
        <p>
            Tuberculosis risk prediction using Random Forest
            with an Explainable AI (SHAP) approach.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## ⚙️ Model Status")

    if model is not None:
        st.success("Model loaded successfully")
        st.caption(f"File: `{model_path}`")
        st.caption(f"Model type: `{MODEL_TYPE}`")
    else:
        st.error("Model could not be loaded")
        if os.path.isfile(model_path):
            st.caption(
                "The model file was found, but could not be read. "
                "See the error message above."
            )
        else:
            st.caption(
                f"File not found: `{model_path}`"
            )

    if imputer is not None:
        st.success("Imputer loaded successfully")
        st.caption(f"File: `{IMPUTER_PATH}`")
    else:
        st.warning("Imputer could not be loaded")
        if os.path.isfile(IMPUTER_PATH):
            st.caption("The imputer file was found, but could not be read.")
        else:
            st.caption(f"File not found: `{IMPUTER_PATH}`")

    st.markdown("---")
    st.markdown("### Research Model")
    st.write("**Random Forest + SHAP**")
    st.write("Research data: **1,500 observations**")
    st.write("Predictor features: **11 variables**")
    st.write("Target: **Status_TB**")

    st.markdown("---")
    st.caption(
        "This dashboard is a model-based analytical tool "
        "and is not a substitute for medical diagnosis."
    )

# ============================================================
# TABS
# ============================================================
tab_pred, tab_model, tab_shap = st.tabs([
    "🔮 TB Risk Prediction",
    "📊 Model Analysis",
    "🧠 Dominant Factors",
])

# ============================================================
# TAB 1 - PREDICTION
# ============================================================
with tab_pred:
    st.markdown("### Enter Individual Characteristics")
    st.caption(
        "Change the input values. The probability and individual SHAP results "
        "will be updated automatically."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=35,
            step=1,
        )

        gender_label = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0,
        )
        gender = 1 if gender_label == "Laki-laki" else 0

        distance = st.number_input(
            "Distance to Clinic",
            min_value=0.0,
            max_value=200.0,
            value=5.0,
            step=0.5,
        )

        temperature = st.number_input(
            "Ambient Temperature",
            min_value=0.0,
            max_value=60.0,
            value=24.0,
            step=0.5,
        )

    with c2:
        ventilation_label = st.selectbox(
            "Ventilation",
            ["Inadequate (0)", "Adequate (1)"],
            index=1,
        )
        ventilation = 1 if ventilation_label.startswith("Adequate") else 0

        symptom_duration = st.number_input(
            "Symptom Duration",
            min_value=0.0,
            max_value=365.0,
            value=7.0,
            step=1.0,
        )

        productive_cough_label = st.selectbox(
            "Productive Cough",
            ["No (0)", "Yes (1)"],
            index=0,
        )
        productive_cough = 1 if productive_cough_label.startswith("Yes") else 0

        dyspnea_label = st.selectbox(
            "Shortness of Breath",
            ["No (0)", "Yes (1)"],
            index=0,
        )
        dyspnea = 1 if dyspnea_label.startswith("Yes") else 0

    with c3:
        chest_pain_label = st.selectbox(
            "Chest Pain",
            ["No (0)", "Yes (1)"],
            index=0,
        )
        chest_pain = 1 if chest_pain_label.startswith("Yes") else 0

        fever_label = st.selectbox(
            "Fever",
            ["No (0)", "Yes (1)"],
            index=0,
        )
        fever = 1 if fever_label.startswith("Yes") else 0

        contact_label = st.selectbox(
            "History of Contact",
            ["No (0)", "Yes (1)"],
            index=0,
        )
        history_contact = 1 if contact_label.startswith("Yes") else 0

    values = {
        "Age": age,
        "Gender": gender,
        "Distance_to_Clinic": distance,
        "Ambient_Temperature": temperature,
        "Ventilation": ventilation,
        "Symptom_Duration": symptom_duration,
        "Productive_Cough": productive_cough,
        "Dyspnea": dyspnea,
        "Chest_Pain": chest_pain,
        "Fever": fever,
        "History_of_Contact": history_contact,
    }

    input_df = prepare_input(values)

    st.markdown("---")
    st.markdown("### Prediction Result")

    if model is None:
        st.warning(
            "The model cannot be used. Make sure the file "
            "`Best_RandomForest_TB.pkl` berada di folder `model/` "
            "and can be read by the current Python environment."
        )
    else:
        pred, probability = predict_probability(input_df)

        if probability is not None:
            risk_percent = probability * 100
        else:
            risk_percent = None

        result_col, prob_col = st.columns([1.15, 1])

        with result_col:
            if pred == 1:
                st.markdown(
                    """
                    <div class="risk-high">
                        <div class="risk-title">⚠️ TB INDICATED</div>
                        <div class="risk-sub">
                            The model predicts the TB class (Status_TB = 1).
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="risk-low">
                        <div class="risk-title">✅ NON-TB</div>
                        <div class="risk-sub">
                            The model predicts the Non-TB class (Status_TB = 0).
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with prob_col:
            if risk_percent is not None:
                st.metric(
                    "TB Probability",
                    f"{risk_percent:.2f}%",
                )
                st.progress(min(max(probability, 0.0), 1.0))

                if probability >= 0.5:
                    st.caption(
                        "The probability falls on the TB class side "
                        "based on the model prediction threshold."
                    )
                else:
                    st.caption(
                        "The probability falls on the Non-TB class side "
                        "based on the model prediction threshold."
                    )
            else:
                st.info("The model does not provide `predict_proba`.")

        # SHAP individual
        st.markdown("---")
        st.markdown("### 🔎 Individual Explanation Using SHAP")

        shap_result, base_value = calculate_individual_shap(input_df)

        if shap_result is not None:
            plot_df = shap_result.copy()
            plot_df["Label"] = plot_df["Feature"].map(feature_label)

            fig = px.bar(
                plot_df,
                x="SHAP",
                y="Label",
                orientation="h",
                title="Contribution of Each Feature to the Individual Prediction",
                labels={
                    "SHAP": "SHAP Value",
                    "Label": "Feature",
                },
            )
            fig.update_layout(
                height=520,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            pos = shap_result[shap_result["SHAP"] > 0].sort_values(
                "SHAP", ascending=False
            )
            neg = shap_result[shap_result["SHAP"] < 0].sort_values(
                "SHAP"
            )

            a, b = st.columns(2)

            with a:
                st.markdown("#### ⬆️ Pushes the Prediction Toward TB")
                if len(pos):
                    for _, row in pos.head(5).iterrows():
                        st.write(
                            f"**{feature_label(row['Feature'])}**: "
                            f"+{row['SHAP']:.4f}"
                        )
                else:
                    st.caption("There is no positive contribution for this observation.")

            with b:
                st.markdown("#### ⬇️ Pushes the Prediction Away from TB")
                if len(neg):
                    for _, row in neg.head(5).iterrows():
                        st.write(
                            f"**{feature_label(row['Feature'])}**: "
                            f"{row['SHAP']:.4f}"
                        )
                else:
                    st.caption("There is no negative contribution for this observation.")

            if base_value is not None:
                st.caption(
                    f"Model SHAP base value: {base_value:.4f}. "
                    "SHAP values represent the relative contribution of features "
                    "to the model output."
                )

        else:
            st.info(
                "Individual SHAP requires the `shap` package and a Random Forest "
                "model that can be processed by TreeExplainer."
            )

# ============================================================
# TAB 2 - MODEL ANALYSIS
# ============================================================
with tab_model:
    st.markdown("### Random Forest Performance")

    cols = st.columns(6)
    metric_items = list(METRICS.items())

    for col, (name, value) in zip(cols, metric_items):
        with col:
            st.metric(name, f"{value:.4f}")

    st.markdown("---")

    left, right = st.columns([1, 1])

    with left:
        st.markdown("### Confusion Matrix")

        cm_fig = go.Figure(
            data=go.Heatmap(
                z=CONFUSION_MATRIX,
                x=["Predicted Non-TB", "Predicted TB"],
                y=["Actual Non-TB", "Actual TB"],
                text=CONFUSION_MATRIX,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=False,
            )
        )
        cm_fig.update_layout(
            template="plotly_dark",
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=30, r=20, t=30, b=30),
        )
        st.plotly_chart(cm_fig, use_container_width=True)

        st.caption(
            "TN = 385, FP = 5, FN = 3, TP = 57. "
            "The matrix comes from evaluation results on 450 test observations."
        )

    with right:
        st.markdown("### Classification Report")

        report = pd.DataFrame({
            "Class": ["Non-TB", "TB"],
            "Precision": [0.99, 0.92],
            "Recall": [0.99, 0.95],
            "F1-Score": [0.99, 0.93],
            "Support": [390, 60],
        })

        st.dataframe(
            report,
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("#### Best Model Configuration")
        params_df = pd.DataFrame(
            list(BEST_PARAMS.items()),
            columns=["Parameter", "Value"],
        )
        st.dataframe(
            params_df,
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("### Performance Interpretation")

    st.info(
        "The model achieved an accuracy of 0.9822 and ROC-AUC of 0.9985. "
        "A TB-class recall of 0.9500 indicates that the model can "
        "identify most TB observations in the test data. "
        "The dashboard displays the same evaluation metrics as the research results."
    )

# ============================================================
# TAB 3 - DOMINANT FACTORS
# ============================================================
with tab_shap:
    st.markdown("### Dominant Factors Based on SHAP")
    st.caption(
        "Mean Absolute SHAP values indicate the magnitude of the average contribution "
        "of each feature to the model predictions in the research."
    )

    dominant = SHAP_GLOBAL.iloc[0]

    st.markdown(
        f"""
        <div class="card">
            <div class="metric-title">DOMINANT FACTOR</div>
            <div class="metric-value">{feature_label(dominant["Feature"])}</div>
            <div class="small-note">
                Mean Absolute SHAP = {dominant["Mean_Abs_SHAP"]:.6f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    shap_plot = SHAP_GLOBAL.sort_values("Mean_Abs_SHAP", ascending=True)
    shap_plot["Label"] = shap_plot["Feature"].map(feature_label)

    fig = px.bar(
        shap_plot,
        x="Mean_Abs_SHAP",
        y="Label",
        orientation="h",
        title="Mean Absolute SHAP",
        labels={
            "Mean_Abs_SHAP": "Mean Absolute SHAP",
            "Label": "Feature",
        },
    )
    fig.update_layout(
        height=560,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    comparison = pd.merge(
        RF_IMPORTANCE,
        SHAP_GLOBAL,
        on="Feature",
        how="inner",
    )

    comparison = comparison.sort_values(
        "Mean_Abs_SHAP",
        ascending=False,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Random Forest Feature Importance")
        rf_plot = RF_IMPORTANCE.sort_values(
            "RF_Importance",
            ascending=True,
        )
        rf_plot["Label"] = rf_plot["Feature"].map(feature_label)

        fig_rf = px.bar(
            rf_plot,
            x="RF_Importance",
            y="Label",
            orientation="h",
            title="RF Feature Importance",
            labels={
                "RF_Importance": "Importance",
                "Label": "Feature",
            },
        )
        fig_rf.update_layout(
            height=500,
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_rf, use_container_width=True)

    with right:
        st.markdown("### RF vs SHAP Comparison")

        table = comparison.copy()
        table["Feature"] = table["Feature"].map(feature_label)
        table.columns = [
            "Feature",
            "RF Importance",
            "Mean Absolute SHAP",
        ]

        st.dataframe(
            table.round(6),
            hide_index=True,
            use_container_width=True,
            height=500,
        )

    st.markdown("---")
    st.markdown("### Interpretation Summary")

    st.success(
        "Symptom_Duration is the most dominant factor according to SHAP "
        "with a Mean Absolute SHAP value of 0.286042. The next factors "
        "are Chest_Pain, History_of_Contact, and Ventilation."
    )

    st.caption(
        "Note: SHAP explains model behavior and does not establish "
        "clinical causal relationships."
    )
