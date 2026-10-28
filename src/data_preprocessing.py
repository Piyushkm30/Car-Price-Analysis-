
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "car data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Load original data without modifying it
df = pd.read_csv(RAW_DATA_PATH)

print("Original rows:", len(df))

# Standardize column names and text spacing
df.columns = df.columns.str.strip()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

# Remove exact duplicate rows
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# Validate essential numeric columns
numeric_columns = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Kms_Driven",
    "Owner"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("Exact duplicate rows removed:", duplicates_removed)
print("Rows after duplicate removal:", len(df))
print("Missing values after conversion:")
print(df.isnull().sum())

# Save cleaned dataset
output_path = PROCESSED_DIR / "cleaned_car_data.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved to:", output_path)
print("Preprocessing completed successfully!")
