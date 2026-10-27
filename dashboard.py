
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.parent / "data"
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "reports"

st.set_page_config(
    page_title="Student Success Prediction",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_DIR / "raw" / "student_dropout.csv"
    )

@st.cache_resource
def load_model():
    return joblib.load(
        MODEL_DIR / "best_random_forest.pkl"
    )

df = load_data()
model = load_model()

feature_columns = list(model.feature_names_in_)

categorical_columns = [
    "Marital Status", "Application mode",
    "Course", "Daytime/evening attendance",
    "Previous qualification", "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced", "Educational special needs",
    "Debtor", "Tuition fees up to date",
    "Gender", "Scholarship holder",
    "International"
]

st.title("🎓 Student Dropout & Academic Success")
st.caption(
    "CST8504 — Machine Learning Prediction Dashboard"
)

tabs = st.tabs([
    "Data Overview",
    "Interactive Exploration",
    "Student Prediction",
    "Model Performance"
])

# PAGE 1: DATA OVERVIEW
with tabs[0]:
    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Students", len(df))
    c2.metric("Input Features", len(feature_columns))
    c3.metric("Missing Values", int(df.isna().sum().sum()))

    st.subheader("Student Outcomes")
    counts = df["Target"].value_counts().reset_index()
    counts.columns = ["Outcome", "Students"]

    fig = px.bar(
        counts,
        x="Outcome",
        y="Students",
        color="Outcome",
        title="Distribution of Student Outcomes"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Student Records")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

# PAGE 2: INTERACTIVE EXPLORATION
with tabs[1]:
    st.header("Interactive Data Exploration")

    outcomes = st.multiselect(
        "Select student outcomes",
        options=sorted(df["Target"].unique()),
        default=sorted(df["Target"].unique())
    )

    courses = st.multiselect(
        "Filter by course codes",
        options=sorted(df["Course"].unique()),
        default=sorted(df["Course"].unique())
    )

    filtered = df[
        df["Target"].isin(outcomes)
        & df["Course"].isin(courses)
    ]

    if filtered.empty:
        st.warning("No records match the selected filters.")
    else:
        st.metric("Filtered students", len(filtered))

        # Chart 1
        fig1 = px.histogram(
            filtered,
            x="Age at enrollment",
            color="Target",
            nbins=30,
            title="Age Distribution by Student Outcome"
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2
        fig2 = px.box(
            filtered,
            x="Target",
            y="Admission grade",
            color="Target",
            title="Admission Grades by Outcome"
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Chart 3
        semester = st.selectbox(
            "Choose academic semester",
            ["1st", "2nd"]
        )

        grade_column = (
            f"Curricular units {semester} sem (grade)"
        )

        fig3 = px.box(
            filtered,
            x="Target",
            y=grade_column,
            color="Target",
            title=f"{semester} Semester Grades by Outcome"
        )
        st.plotly_chart(fig3, use_container_width=True)

# PAGE 3: PREDICTION
with tabs[2]:
    st.header("Student Outcome Prediction")

    st.info(
        "Predictions use information available through "
        "the second semester. They are illustrative "
        "model outputs, not individual academic decisions."
    )

    example_index = st.selectbox(
        "Select an example record",
        options=list(range(len(df))),
        format_func=lambda i: f"Example student {i + 1}"
    )

    example = df.iloc[example_index]
    input_values = {}

    with st.form("prediction_form"):
        st.subheader("Student Characteristics")

        col1, col2 = st.columns(2)

        for i, feature in enumerate(feature_columns):
            container = col1 if i % 2 == 0 else col2

            with container:
                if feature in categorical_columns:
                    options = sorted(
                        df[feature].dropna().unique().tolist()
                    )
                    default = example[feature]

                    input_values[feature] = st.selectbox(
                        feature,
                        options=options,
                        index=options.index(default)
                    )
                else:
                    default = example[feature]

                    if pd.api.types.is_integer_dtype(
                        df[feature]
                    ):
                        input_values[feature] = st.number_input(
                            feature,
                            value=int(default),
                            step=1
                        )
                    else:
                        input_values[feature] = st.number_input(
                            feature,
                            value=float(default),
                            format="%.4f"
                        )

        submitted = st.form_submit_button(
            "Predict Academic Outcome",
            type="primary"
        )

    if submitted:
        student_input = pd.DataFrame(
            [input_values],
            columns=feature_columns
        )

        prediction = model.predict(student_input)[0]

        probabilities = model.predict_proba(
            student_input
        )[0]

        classes = model.named_steps[
            "classifier"
        ].classes_

        probability_df = pd.DataFrame({
            "Outcome": classes,
            "Probability": probabilities
        })

        st.success(f"Predicted Outcome: {prediction}")

        st.subheader("Predicted Class Probabilities")

        fig = px.bar(
            probability_df,
            x="Outcome",
            y="Probability",
            color="Outcome",
            range_y=[0, 1],
            title="Estimated Outcome Probabilities"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )

# PAGE 4: MODEL PERFORMANCE
with tabs[3]:
    st.header("Final Model Evaluation")

    results_file = (
        REPORT_DIR / "final_model_comparison.csv"
    )

    if results_file.exists():
        results = pd.read_csv(results_file)

        st.subheader("Model Comparison")
        st.dataframe(results, use_container_width=True)

        fig = px.bar(
            results,
            x="Model",
            y="Accuracy",
            color="Model",
            title="Final Test Accuracy"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Model comparison file not found.")

    cm_file = (
        REPORT_DIR /
        "tuned_random_forest_confusion_matrix.csv"
    )

    if cm_file.exists():
        cm = pd.read_csv(cm_file, index_col=0)

        st.subheader("Random Forest Confusion Matrix")

        fig = px.imshow(
            cm.values,
            x=cm.columns.tolist(),
            y=cm.index.tolist(),
            text_auto=True,
            labels={
                "x": "Predicted Outcome",
                "y": "Actual Outcome",
                "color": "Students"
            },
            title="Final Test Confusion Matrix",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Confusion matrix file not found.")

    st.caption(
        "Models were evaluated on a held-out test "
        "dataset of 664 students."
    )
