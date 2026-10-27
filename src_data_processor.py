
import pandas as pd
from pathlib import Path

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "student_dropout.csv"

# Load the dataset
df = pd.read_csv(RAW_FILE)

print("===== DATASET INFORMATION =====")
print("Shape:", df.shape)

print("\n===== DATA TYPES =====")
print(df.dtypes.to_string())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum().to_string())

print("\nTotal missing values:", df.isnull().sum().sum())

print("\n===== DUPLICATE RECORDS =====")
print("Duplicate rows:", df.duplicated().sum())

print("\n===== DESCRIPTIVE STATISTICS =====")
print(df.describe().to_string())

print("\n===== TARGET DISTRIBUTION =====")
print(df["Target"].value_counts())

print("\n===== TARGET PERCENTAGES =====")
print(
    df["Target"].value_counts(normalize=True).mul(100).round(2)
)

python src/data_processor.py
