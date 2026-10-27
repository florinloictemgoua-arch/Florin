
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Define project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# STEP 2: Load training and validation datasets
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train.drop(columns=["Target"])
y_train = train["Target"]

X_val = val.drop(columns=["Target"])
y_val = val["Target"]

# STEP 3: Define categorical features
categorical_columns = [
    "Marital Status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International"
]

numerical_columns = [
    col for col in X_train.columns
    if col not in categorical_columns
]

# STEP 4: Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            "passthrough",
            numerical_columns
        )
    ]
)

# STEP 5: Create Random Forest pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

# STEP 6: Train model
print("\n===== TRAINING RANDOM FOREST =====")
model.fit(X_train, y_train)

# STEP 7: Predict validation outcomes
predictions = model.predict(X_val)

# STEP 8: Evaluate performance
accuracy = accuracy_score(y_val, predictions)

macro_precision = precision_score(
    y_val, predictions, average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_val, predictions, average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_val, predictions, average="macro",
    zero_division=0
)

print("\n===== VALIDATION RESULTS =====")
print("Accuracy:", round(accuracy, 4))
print("Macro Precision:", round(macro_precision, 4))
print("Macro Recall:", round(macro_recall, 4))
print("Macro F1-score:", round(macro_f1, 4))

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_val,
        predictions,
        zero_division=0
    )
)

labels = ["Dropout", "Enrolled", "Graduate"]

print("\n===== CONFUSION MATRIX =====")
print(
    confusion_matrix(
        y_val, predictions, labels=labels
    )
)
print("Class order:", labels)

# STEP 9: Save trained pipeline
model_path = MODEL_DIR / "random_forest.pkl"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)

# STEP 10: Save validation results
results = pd.DataFrame([{
    "Model": "Random Forest",
    "Accuracy": accuracy,
    "Macro Precision": macro_precision,
    "Macro Recall": macro_recall,
    "Macro F1": macro_f1
}])

results.to_csv(
    REPORT_DIR / "random_forest_results.csv",
    index=False
)

print("\nRandom Forest training completed!")
