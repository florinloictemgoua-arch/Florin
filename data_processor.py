
from pathlib import Path
import pandas as pd

# Locate the folder containing the dataset
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "student_dropout.csv"

# Check whether the file exists
print("===== FILE VERIFICATION =====")
print("Dataset location:", DATA_PATH)
print("File exists:", DATA_PATH.is_file())

# Load the dataset only if the file exists
if DATA_PATH.is_file():
    df = pd.read_csv(DATA_PATH)

    print("\n===== DATASET INFORMATION =====")
    print("Dataset shape:", df.shape)
    print("Total missing values:", df.isnull().sum().sum())
    print("Number of duplicates:", df.duplicated().sum())
else:
    print("ERROR: The dataset was not found.")

# STEP 8: Inspect feature types and values

print("\n===== COLUMN NAMES AND DATA TYPES =====")

for column in df.columns:
    print(f"{column}: {df[column].dtype}")

print("\n===== UNIQUE VALUES PER COLUMN =====")

for column in df.columns:
    unique_count = df[column].nunique()
    print(f"{column}: {unique_count} unique values")

print("\n===== NUMERICAL SUMMARY =====")
print(df.describe().to_string())

print("\n===== TARGET CLASS DISTRIBUTION =====")
print(df["Target"].value_counts())

print("\n===== TARGET CLASS PERCENTAGES =====")
print(
    df["Target"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)



# STEP 10: Generate a readable statistical report

from pathlib import Path

# Create the output folder
REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Create a statistical summary
summary = df.describe(include="all").transpose()

# Add data type and number of unique values
summary["Data Type"] = df.dtypes.astype(str)
summary["Unique Values"] = df.nunique()
summary["Missing Values"] = df.isnull().sum()

# Save the results
output_file = REPORT_DIR / "data_summary.csv"
summary.to_csv(output_file)

print("\n===== STATISTICAL REPORT =====")
print("Report saved to:", output_file)

# Print the first ten variables
print("\n===== FIRST 10 VARIABLES =====")
print(summary.head(10).to_string())




# STEP 13: Correct feature classification

# Categorical features and binary indicators
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

# Separate features and target
X = df.drop(columns=["Target"])
y = df["Target"]

# Identify numerical features automatically
numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

# Verify every categorical column exists
unknown_columns = [
    col for col in categorical_columns
    if col not in X.columns
]

if unknown_columns:
    raise ValueError(
        f"Unknown categorical columns: {unknown_columns}"
    )

# Print categorical features
print("\n===== CATEGORICAL FEATURES =====")
for col in categorical_columns:
    print(col)

# Print numerical features
print("\n===== NUMERICAL FEATURES =====")
for col in numerical_columns:
    print(col)

# Print totals
print("\n===== FEATURE CLASSIFICATION SUMMARY =====")
print("Categorical features:", len(categorical_columns))
print("Numerical features:", len(numerical_columns))
print("Total classified:", len(categorical_columns) +
      len(numerical_columns))
print("Total input features:", X.shape[1])

# Check that there are no unclassified features
assert set(categorical_columns).isdisjoint(
    numerical_columns
), "Feature groups overlap!"

assert set(categorical_columns + numerical_columns) == set(
    X.columns
), "Some features are missing or unknown!"

print("\nSUCCESS: All 36 features classified!")

