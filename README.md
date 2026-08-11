# 🏍️ Bike Resale Price Predictor

A machine learning project that predicts the resale price of used bikes in India based on brand, age, mileage, engine power, city, and ownership history. Built end-to-end — from raw scraped data to a deployed interactive web app.

## 🎯 Problem Statement

Used bike buyers and sellers in India often lack a reliable way to estimate a fair resale price. This project builds a regression model trained on real listing data to predict prices, helping both parties make informed decisions.

## 📊 Dataset

- Source: [Used Bikes Prices in India](https://www.kaggle.com/datasets/saisaathvik/used-bikes-prices-in-india) (Kaggle, scraped from droom.in)
- ~32,600 raw listings, 8 features (brand, price, city, kms driven, owner, age, power, bike name)

## 🧹 Data Cleaning & EDA

- **Discovered and removed ~78% duplicate rows** (32,648 → 7,318 unique listings) — a major data quality issue in the raw scraped data
- Removed unrealistic outliers using a km-per-year ratio check (e.g., listings implying 100,000+ km/year of usage)
- Verified extreme values (like 40+ year old bikes) against domain knowledge (Royal Enfield Bullets, Jawas) rather than blindly filtering
- Applied log-transformation to price to correct right-skewed distribution
- Grouped 443 cities into top 15 + "Other" to avoid sparse one-hot encoding

## 🔍 Key Findings

- **Engine power was the strongest price driver** (0.81 correlation with price, ~70% feature importance in the final model)
- Age and mileage had moderate, roughly equal negative influence on price
- Brand showed clear tiering (premium imports like Ducati/Harley-Davidson vs. budget brands like Hero/TVS), though its individual model importance was partly absorbed by power (multicollinearity)

## 🤖 Models Compared

| Model | R² Score | MAE (₹) | MAPE |
|---|---|---|---|
| Linear Regression | 0.81 | ₹32,588 | 24.1% |
| Random Forest | 0.92 | ₹14,066 | 17.2% |
| **XGBoost (final)** | **0.93** | **₹13,387** | **16.0%** |

## 🖥️ Demo

Built an interactive Streamlit app where users input bike details (brand, city, owner, age, kms driven, power) and get an instant price estimate.

## 🛠️ Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit

## 🚀 How to Run

\`\`\`bash
git clone https://github.com/aryanagrawal552-coder/bike-price-prediction.git
cd bike-price-prediction
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
\`\`\`

(Note: dataset not included in repo — download from the Kaggle link above and place in a `data/` folder as `bikes_raw.csv` before running the notebooks.)

## 📈 Future Improvements

- Scrape fresh listings from OLX for a self-collected, current dataset
- Extract bike model/variant from `bike_name` as an additional feature
- Extend to used cars
- Hyperparameter tuning for further model improvement