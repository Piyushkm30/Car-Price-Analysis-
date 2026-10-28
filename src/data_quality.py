
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "car data.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("\n===== DATA QUALITY REPORT =====")
print("Dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("\nMissing values:")
print(df.isnull().sum())

print("\nUnique values in categorical columns:")
for col in ["Fuel_Type", "Seller_Type", "Transmission"]:
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False))

print("\n===== NUMERICAL RANGE CHECK =====")
numeric_cols = [
    "Year", "Selling_Price", "Present_Price",
    "Kms_Driven", "Owner"
]
print(df[numeric_cols].describe().T)

print("\n===== EXTREME MILEAGE RECORDS =====")
print(
    df.nlargest(5, "Kms_Driven")[
        ["Car_Name", "Year", "Selling_Price", "Kms_Driven"]
    ].to_string(index=False)
)

print("\n===== POTENTIAL INVALID VALUES =====")
checks = {
    "Selling_Price <= 0": df["Selling_Price"] <= 0,
    "Present_Price <= 0": df["Present_Price"] <= 0,
    "Kms_Driven < 0": df["Kms_Driven"] < 0,
    "Year outside 2000-2026": ~df["Year"].between(2000, 2026),
}

for label, mask in checks.items():
    print(f"{label}: {mask.sum()} records")

df.describe(include="all").T.to_csv(
    REPORT_DIR / "data_quality_summary.csv"
)

print("\nData quality report saved successfully.")
