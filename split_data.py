
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project paths
BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = (
    BASE_DIR.parent / "data" / "raw" /
    "student_dropout.csv"
)

OUTPUT_DIR = BASE_DIR.parent / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original dataset
df = pd.read_csv(RAW_FILE)

# Separate input features and target
X = df.drop(columns=["Target"])
y = df["Target"]

# First split: 70% training, 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: 15% validation, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Save all three subsets
for name, features, target in [
    ("train", X_train, y_train),
    ("validation", X_val, y_val),
    ("test", X_test, y_test)
]:
    result = features.copy()
    result["Target"] = target

    result.to_csv(
        OUTPUT_DIR / f"{name}.csv",
        index=False
    )

    print(f"\n===== {name.upper()} SET =====")
    print("Records:", len(result))
    print("Class distribution:")
    print(target.value_counts())
    print("Class percentages:")
    print(
        target.value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

# Confirm all records are accounted for
total = len(X_train) + len(X_val) + len(X_test)

assert total == len(df), "Record count mismatch!"

# Confirm no overlap between original row indices
assert set(X_train.index).isdisjoint(X_val.index)
assert set(X_train.index).isdisjoint(X_test.index)
assert set(X_val.index).isdisjoint(X_test.index)

print("\n===== FINAL VERIFICATION =====")
print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))
print("Total records:", total)
print("Data splitting completed successfully!")
