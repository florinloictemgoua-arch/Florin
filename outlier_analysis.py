
import pandas as pd
from pathlib import Path

# Locate our existing data
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "raw" / "student_dropout.csv"

# Load original dataset
df = pd.read_csv(DATA_PATH)

# Measurable numerical features for IQR screening
numerical_columns = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP"
]

results = []

for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    results.append({
        "Feature": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": int(outliers.sum()),
        "Outlier Percentage": round(
            outliers.mean() * 100, 2
        )
    })

# Convert results into a DataFrame
outlier_report = pd.DataFrame(results)

# Sort from highest to lowest outlier count
outlier_report = outlier_report.sort_values(
    by="Outlier Count",
    ascending=False
)

# Save report
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

output_file = REPORT_DIR / "outlier_report.csv"
outlier_report.to_csv(output_file, index=False)

print("\n===== OUTLIER ANALYSIS =====")
print(outlier_report.to_string(index=False))

print("\n===== SUMMARY =====")
print("Total student records:", len(df))
print("Features analyzed:", len(numerical_columns))
print("Original records removed: 0")
print("Report saved to:", output_file)
