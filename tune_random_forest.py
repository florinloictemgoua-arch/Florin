
import pandas as pd
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score
)

# STEP 1: File locations
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# STEP 2: Load training and validation data
train = pd.read_csv(DATA_DIR / "train.csv")
val = pd.read_csv(DATA_DIR / "validation.csv")

X_train = train.drop(columns=["Target"])
y_train = train["Target"]

X_val = val.drop(columns=["Target"])
y_val = val["Target"]

# STEP 3: Define categorical columns
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
    c for c in X_train.columns
    if c not in categorical_columns
]

# STEP 4: Preprocessing
preprocessor = ColumnTransformer([
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
])

# STEP 5: Define configurations
configurations = [
    {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 1,
        "class_weight": "balanced"
    },
    {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 1,
        "class_weight": "balanced"
    },
    {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_leaf": 2,
        "class_weight": "balanced"
    },
    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 2,
        "class_weight": "balanced"
    },
    {
        "n_estimators": 300,
        "max_depth": 15,
        "min_samples_leaf": 3,
        "class_weight": None
    },
    {
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "class_weight": None
    }
]

# STEP 6: Train and evaluate each configuration
results = []
best_score = -1
best_model = None
best_config = None

for i, config in enumerate(configurations, start=1):

    print(f"\n===== EXPERIMENT {i} =====")
    print(config)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            **config,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )
    dropout_recall = recall_score(
        y_val, predictions,
        labels=["Dropout"],
        average=None
    )[0]

    results.append({
        "Experiment": i,
        **config,
        "Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Dropout Recall": dropout_recall
    })

    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Dropout Recall:", round(dropout_recall, 4))

    # Primary selection criterion: Macro F1
    # Tie-breaker: Dropout recall
    current_score = (macro_f1, dropout_recall)

    if current_score > (
        (-1, -1) if best_config is None else best_score
    ):
        best_score = current_score
        best_model = model
        best_config = config

# STEP 7: Save experiment results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["Macro F1", "Dropout Recall"],
    ascending=False
)

results_path = REPORT_DIR / "tuning_results.csv"
results_df.to_csv(results_path, index=False)

# STEP 8: Save the best validated model
best_model_path = MODEL_DIR / "best_random_forest.pkl"
joblib.dump(best_model, best_model_path)

print("\n===== FINAL TUNING SUMMARY =====")
print(results_df.to_string(index=False))

print("\nBest configuration:", best_config)
print("Best validation Macro F1:", best_score[0])
print("Best validation Dropout Recall:", best_score[1])
print("Results saved to:", results_path)
print("Model saved to:", best_model_path)
print("\nTuning completed successfully!")
