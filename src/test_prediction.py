
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "car_price_model.pkl"

model = joblib.load(MODEL_PATH)

# Example car details
car = pd.DataFrame([{
    "Year": 2018,
    "Present_Price": 8.5,
    "Kms_Driven": 30000,
    "Fuel_Type": "Petrol",
    "Seller_Type": "Dealer",
    "Transmission": "Manual",
    "Owner": 0
}])

predicted_price = model.predict(car)[0]

print("===== USED CAR PRICE PREDICTION =====")
print(f"Predicted selling price: {predicted_price:.2f}")
print("Prediction completed successfully.")
