
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "car data.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Basic preparation for analysis
df.columns = df.columns.str.strip()
df["Selling_Price"] = pd.to_numeric(
    df["Selling_Price"], errors="coerce"
)

df = df.dropna(subset=["Selling_Price"])

sns.set_theme(style="whitegrid")

print("\n===== EDA REPORT =====")
print(f"Total records: {len(df)}")
print(f"Unique car names: {df['Car_Name'].nunique()}")
print(f"Average selling price: {df['Selling_Price'].mean():.2f}")
print(f"Median selling price: {df['Selling_Price'].median():.2f}")
print(f"Minimum selling price: {df['Selling_Price'].min():.2f}")
print(f"Maximum selling price: {df['Selling_Price'].max():.2f}")

# 1. Selling price distribution
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Selling_Price", kde=True, bins=25)
plt.title("Distribution of Used Car Selling Prices")
plt.xlabel("Selling Price (dataset units)")
plt.ylabel("Number of Cars")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "selling_price_distribution.png", dpi=150)
plt.close()

# 2. Selling price by fuel type
plt.figure(figsize=(9, 5))
sns.boxplot(data=df, x="Fuel_Type", y="Selling_Price")
plt.title("Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price (dataset units)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "price_by_fuel_type.png", dpi=150)
plt.close()

# 3. Selling price by transmission
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Transmission", y="Selling_Price")
plt.title("Selling Price by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Selling Price (dataset units)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "price_by_transmission.png", dpi=150)
plt.close()

# 4. Kilometres driven vs selling price
plt.figure(figsize=(9, 5))
sns.scatterplot(
    data=df,
    x="Kms_Driven",
    y="Selling_Price",
    hue="Fuel_Type",
    alpha=0.8
)
plt.title("Kilometres Driven vs Selling Price")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price (dataset units)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "kms_vs_selling_price.png", dpi=150)
plt.close()

# 5. Correlation heatmap for numerical columns
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

plt.figure(figsize=(10, 7))
sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title("Correlation Between Numerical Features")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
plt.close()

# Save descriptive statistics
df.describe(include="all").to_csv(
    BASE_DIR / "reports" / "descriptive_statistics.csv"
)

# Print useful group-level summaries
print("\n===== AVERAGE PRICE BY FUEL TYPE =====")
print(df.groupby("Fuel_Type")["Selling_Price"].agg(
    ["count", "mean", "median"]
).round(2))

print("\n===== AVERAGE PRICE BY TRANSMISSION =====")
print(df.groupby("Transmission")["Selling_Price"].agg(
    ["count", "mean", "median"]
).round(2))

print("\n===== CORRELATION WITH SELLING PRICE =====")
print(correlation["Selling_Price"].sort_values(ascending=False).round(3))

print("\nEDA completed successfully!")
print(f"Charts saved in: {FIGURES_DIR}")
