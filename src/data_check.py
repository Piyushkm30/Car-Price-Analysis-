
import pandas as pd
from pathlib import Path

# Find the dataset
data_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "car data.csv"
)

# Load the dataset
if not data_path.exists():
    raise FileNotFoundError(f"Dataset not found: {data_path}")

df = pd.read_csv(data_path)

print("\n===== DATASET OVERVIEW =====")
print("Dataset shape:", df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\nDataset check completed successfully!")
