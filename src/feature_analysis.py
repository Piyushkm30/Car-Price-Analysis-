
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "car data.csv"
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

# 1. Price statistics by fuel type
fuel_summary = df.groupby("Fuel_Type")["Selling_Price"].agg(
    Count="count",
    Mean="mean",
    Median="median",
    Minimum="min",
    Maximum="max"
).round(2)

print("\n===== PRICE BY FUEL TYPE =====")
print(fuel_summary)
fuel_summary.to_csv(REPORT_DIR / "price_by_fuel_type.csv")

# 2. Price statistics by seller type
seller_summary = df.groupby("Seller_Type")["Selling_Price"].agg(
    Count="count",
    Mean="mean",
    Median="median"
).round(2)

print("\n===== PRICE BY SELLER TYPE =====")
print(seller_summary)
seller_summary.to_csv(REPORT_DIR / "price_by_seller_type.csv")

# 3. Price statistics by transmission
transmission_summary = df.groupby("Transmission")["Selling_Price"].agg(
    Count="count",
    Mean="mean",
    Median="median"
).round(2)

print("\n===== PRICE BY TRANSMISSION =====")
print(transmission_summary)
transmission_summary.to_csv(
    REPORT_DIR / "price_by_transmission.csv"
)

# 4. Most common car names
print("\n===== TOP 10 MOST COMMON CAR NAMES =====")
print(df["Car_Name"].value_counts().head(10))

# 5. Price relationship
print("\n===== PRESENT PRICE VS SELLING PRICE =====")
print(
    df[["Present_Price", "Selling_Price"]]
    .corr()
    .round(3)
)

print("\nFeature analysis completed successfully.")
