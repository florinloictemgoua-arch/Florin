
import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# STEP 1: Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# STEP 2: Load untouched test dataset
test = pd.read_csv(DATA_DIR / "test.csv")

X_test = test.drop(columns=["Target"])
y_test = test["Target"]

print("===== FINAL TEST DATA =====")
print("Test students:", len(test))

assert len(test) == 664, "Unexpected test set size!"

# STEP 3: Load previously trained models
models = {
    "Decision Tree": joblib.load(
        MODEL_DIR / "decision_tree.pkl"
    ),
    "Tuned Random Forest": joblib.load(
        MODEL_DIR / "best_random_forest.pkl"
    )
}

# STEP 4: Define consistent class order
labels = ["Dropout", "Enrolled", "Graduate"]

results = []

# STEP 5: Evaluate both models
for model_name, model in models.items():

    print(f"\n===== {model_name.upper()} =====")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    macro_precision = precision_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test, predictions,
        average="macro",
        zero_division=0
    )

    dropout_recall = recall_score(
        y_test, predictions,
        labels=["Dropout"],
        average=None,
        zero_division=0
    )[0]

    print("Accuracy:", round(accuracy, 4))
    print("Macro Precision:", round(macro_precision, 4))
    print("Macro Recall:", round(macro_recall, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Dropout Recall:", round(dropout_recall, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=labels,
            zero_division=0
        )
    )

    # Compute confusion matrix
    cm = confusion_matrix(
        y_test, predictions, labels=labels
    )

    print("\nConfusion Matrix:")
    print(cm)
    print("Class order:", labels)

    # Save confusion matrix
    safe_name = model_name.lower().replace(" ", "_")

    cm_df = pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )

    cm_df.to_csv(
        REPORT_DIR / f"{safe_name}_confusion_matrix.csv"
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Dropout Recall": dropout_recall
    })

# STEP 6: Save final comparison
results_df = pd.DataFrame(results)

results_df.to_csv(
    REPORT_DIR / "final_model_comparison.csv",
    index=False
)

print("\n===== FINAL MODEL COMPARISON =====")
print(results_df.to_string(index=False))

print("\nFinal evaluation completed successfully!")
