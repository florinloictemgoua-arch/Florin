
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Locate project files
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

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

# Remaining columns are treated as numerical
numerical_columns = [
    col for col in X_train.columns
    if col not in categorical_columns
]

# STEP 4: Build preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_columns
        )
    ]
)

# STEP 5: Create Decision Tree pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=5,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)

# STEP 6: Train the model
print("===== TRAINING DECISION TREE =====")
model.fit(X_train, y_train)

# STEP 7: Evaluate on validation set
predictions = model.predict(X_val)

print("\n===== VALIDATION RESULTS =====")
print(
    "Accuracy:",
    round(accuracy_score(y_val, predictions), 4)
)

print("\n===== CLASSIFICATION REPORT =====")
print(
    classification_report(
        y_val,
        predictions,
        zero_division=0
    )
)

print("\n===== CONFUSION MATRIX =====")
labels = ["Dropout", "Enrolled", "Graduate"]
print(confusion_matrix(
    y_val, predictions, labels=labels
))
print("Class order:", labels)

# STEP 8: Save the complete pipeline
model_path = MODEL_DIR / "decision_tree.pkl"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)
print("Decision Tree training completed!")
