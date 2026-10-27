
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Download the approved UCI dataset
dataset = fetch_ucirepo(id=697)

# Extract features and target
X = dataset.data.features
y = dataset.data.targets

# Combine into one DataFrame
df = pd.concat([X, y], axis=1)

# Create the destination folder
raw_dir = BASE_DIR / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

# Save a local copy
file_path = raw_dir / "student_dropout.csv"
df.to_csv(file_path, index=False)

# Display dataset information
print("Dataset downloaded successfully!")
print("Dataset shape:", df.shape)
print("Features:", X.shape[1])
print("Target columns:", list(y.columns))
print("\nFirst five records:")
print(df.head())
print("\nClass distribution:")
print(y.value_counts())
print("\nSaved to:", file_path)
