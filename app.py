import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🏍️ Bike Resale Price Predictor")
st.write("Enter your bike's details to get an estimated resale price.")

# Load model and column structure
model = joblib.load('models/bike_price_model.pkl')
model_columns = joblib.load('models/model_columns.pkl')

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


if st.sidebar.button("Predict Price"):
    log_prediction = model.predict(input_df)[0]
    predicted_price = np.expm1(log_prediction)

    st.subheader("Estimated Resale Price")
    st.markdown(f"## ₹{predicted_price:,.0f}")

    st.write(f"Based on a **{brand}** bike, **{age} years** old, with **{kms_driven:,} km** driven, **{power}cc** power, in **{city}**, owned by **{owner}**.")