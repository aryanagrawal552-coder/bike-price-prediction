import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🏍️ Bike Resale Price Predictor")
st.write("Enter your bike's details to get an estimated resale price.")

# Load model and column structure
model = joblib.load('models/bike_price_model.pkl')
model_columns = joblib.load('models/model_columns.pkl')
model_lower = joblib.load('models/bike_price_model_lower.pkl')
model_upper = joblib.load('models/bike_price_model_upper.pkl')

st.sidebar.header("Enter Bike Details")

brand = st.sidebar.selectbox("Brand", [
    'Bajaj', 'Benelli', 'Ducati', 'Harley-Davidson', 'Hero', 'Honda',
    'Hyosung', 'Ideal', 'Indian', 'Jawa', 'KTM', 'Kawasaki', 'LML', 'MV',
    'Mahindra', 'Rajdoot', 'Royal Enfield', 'Suzuki', 'TVS', 'Triumph',
    'Yamaha', 'Yezdi', 'BMW'
])

city = st.sidebar.selectbox("City", [
    'Bangalore', 'Chennai', 'Delhi', 'Faridabad', 'Ghaziabad', 'Gurgaon',
    'Hyderabad', 'Jaipur', 'Kolkata', 'Ludhiana', 'Mumbai', 'Noida',
    'Pune', 'Thane', 'Other'
])

owner = st.sidebar.selectbox("Owner", [
    'First Owner', 'Second Owner', 'Third Owner', 'Fourth Owner Or More'
])

age = st.sidebar.slider("Age (years)", 1, 30, 5)
kms_driven = st.sidebar.number_input("Kms Driven", min_value=0, value=15000)
power = st.sidebar.number_input("Power (cc)", min_value=100, value=150)

owner_map = {
    'First Owner': 0,
    'Second Owner': 1,
    'Third Owner': 2,
    'Fourth Owner Or More': 3
}

# Create a dataframe with all zeros, matching the model's expected columns
input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)

# Fill in the actual values
input_df['kms_driven'] = kms_driven
input_df['age'] = age
input_df['power'] = power
input_df['owner_encoded'] = owner_map[owner]

brand_col = f'brand_{brand}'
if brand_col in input_df.columns:
    input_df[brand_col] = 1

city_col = f'city_{city}'
if city_col in input_df.columns:
    input_df[city_col] = 1

# Calculate predictions once, used by both tabs
log_prediction = model.predict(input_df)[0]
predicted_price = np.expm1(log_prediction)

log_lower = model_lower.predict(input_df)[0]
log_upper = model_upper.predict(input_df)[0]
lower_price = np.expm1(log_lower)
upper_price = np.expm1(log_upper)

tab1, tab2 = st.tabs(["💰 Price Predictor", "🔍 Deal Checker"])

with tab1:
    if st.button("Predict Price"):
        st.subheader("Estimated Resale Price")
        st.markdown(f"## ₹{predicted_price:,.0f}")
        st.write(f"**Fair price range:** ₹{lower_price:,.0f} – ₹{upper_price:,.0f}")
        st.write(f"Based on a **{brand}** bike, **{age} years** old, with **{kms_driven:,} km** driven, **{power}cc** power, in **{city}**, owned by **{owner}**.")

with tab2:
    st.subheader("Is this listing a good deal?")
    st.write("Using the bike details from the sidebar, enter a listed price to check if it's fairly priced.")

    actual_listing_price = st.number_input("Listed Price (₹)", min_value=0, value=50000, key="deal_checker_price")

    if st.button("Check Deal"):
        st.write(f"**Fair price range for this bike:** ₹{lower_price:,.0f} – ₹{upper_price:,.0f}")

        if actual_listing_price < lower_price:
            st.success("🟢 This looks **Underpriced** — potentially a good deal!")
        elif actual_listing_price > upper_price:
            st.error("🔴 This looks **Overpriced** compared to similar bikes.")
        else:
            st.info("🟡 This looks like a **Fair Price** for this bike.")