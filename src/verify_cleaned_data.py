
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "cleaned_car_data.csv"

df = pd.read_csv(DATA_PATH)

print("\n===== CLEANED DATA VERIFICATION =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("Missing values:", df.isnull().sum().sum())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== PRICE SUMMARY =====")
print(df[["Selling_Price", "Present_Price"]].describe())

print("\n===== CATEGORY COUNTS =====")
for col in ["Fuel_Type", "Seller_Type", "Transmission"]:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("\nVerification completed!")
