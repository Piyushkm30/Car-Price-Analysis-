
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "car data.csv"

df = pd.read_csv(DATA_PATH)

duplicates = df[df.duplicated(keep=False)]

print("\n===== DUPLICATE RECORDS =====")
if duplicates.empty:
    print("No duplicate records found.")
else:
    print(duplicates.sort_values(by=list(df.columns)).to_string(index=False))

    print("\n===== DUPLICATE ROW COUNT =====")
    print("All rows involved:", len(duplicates))
    print("Extra copies:", df.duplicated().sum())

print("\nOriginal dataset rows:", len(df))
