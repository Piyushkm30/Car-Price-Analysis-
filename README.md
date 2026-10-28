
# 🚘 AutoPulse — Used Car Price Analysis & Prediction

An end-to-end Data Science and Machine Learning project for analyzing
used-car market trends and estimating vehicle selling prices through
an interactive Streamlit dashboard.

## 📌 Project Overview

AutoPulse explores used-car data using data cleaning, exploratory data
analysis (EDA), statistical summaries, interactive visualizations, and
regression models. It also provides a web-based interface for exploring
the dataset and generating price estimates.

## ✨ Features

- Data quality checks and duplicate removal
- Exploratory Data Analysis (EDA)
- Interactive Plotly charts and correlation heatmap
- Fuel type, seller type, and transmission analysis
- Comparison of Linear Regression, Random Forest, and Gradient Boosting
- Machine Learning-based selling-price estimation
- Interactive Streamlit dashboard
- Dataset filtering and CSV export
- Model evaluation using MAE, RMSE, and R²

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas, NumPy | Data processing and numerical analysis |
| Matplotlib, Seaborn | Exploratory visualizations |
| Plotly | Interactive charts |
| Scikit-learn | Machine Learning and model evaluation |
| Streamlit | Interactive web dashboard |
| Joblib | Saving and loading the trained model |
| Jupyter Notebook | Interactive data analysis |

## 📂 Project Structure

```text
Car Data Analysis/
├── data/
│   ├── car data.csv
│   └── processed/
│       └── cleaned_car_data.csv
├── notebooks/
│   └── car_analysis.ipynb
├── src/
│   ├── app.py
│   ├── data_check.py
│   ├── data_quality.py
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── feature_analysis.py
│   ├── inspect_duplicates.py
│   ├── train_model.py
│   ├── test_prediction.py
│   └── verify_cleaned_data.py
├── models/
│   └── car_price_model.pkl
├── reports/
│   ├── descriptive_statistics.csv
│   ├── model_comparison.csv
│   └── figures/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md
```

## 📊 Dataset

The project uses a used-car dataset containing vehicle attributes such as:

- Car name
- Manufacturing year
- Selling price
- Present price
- Kilometres driven
- Fuel type
- Seller type
- Transmission
- Previous owners

The original dataset contains 301 records. After removing two exact
duplicate rows, the cleaned dataset contains 299 records.

**Note:** Verify the dataset source and its price units before interpreting
predictions as Indian rupees or lakhs.

## 🤖 Machine Learning

Three regression models were trained and evaluated using a held-out
test set.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | 1.122 | 2.464 | 0.765 |
| Linear Regression | 1.473 | 2.524 | 0.753 |
| Random Forest | 1.358 | 3.182 | 0.607 |

Gradient Boosting was selected based on the lowest test Mean Absolute Error.

These results describe the current test split; they do not guarantee
the same performance on new vehicles. MAE and RMSE use the dataset's
original price unit. R² is not the same as prediction accuracy.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Car Data Analysis"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual repository URL.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the dashboard

```bash
streamlit run src/app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 📈 Model Evaluation

- **MAE:** Average absolute difference between actual and predicted prices.
- **RMSE:** Penalizes larger prediction errors more heavily.
- **R²:** Measures how much of the target variation is explained by the model.

## ⚠️ Limitations

- The dataset is relatively small.
- Test-set results may vary with different data splits.
- Predictions may be less reliable for unusual vehicles or values outside
  the training data.
- Actual resale prices depend on vehicle condition, location, service
  history, demand, and other factors.
- The price unit and original dataset licence should be verified.

## 🔮 Future Improvements

- Add automated model testing and cross-validation
- Improve feature engineering and hyperparameter tuning
- Add model explainability
- Expand the dataset with more recent vehicle listings
- Deploy the dashboard online

## 👨‍💻 Author

**Piyush Kumar Mishra**

Data Science | Machine Learning | Python

---

*Built as a Data Science and Machine Learning project.*
