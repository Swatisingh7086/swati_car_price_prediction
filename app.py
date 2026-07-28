import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

model = pickle.load(open("LinearRegressionModel.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.title("🚗 Car Price Prediction")

present_price = st.number_input("Present Price (Lakh)", min_value=0.0, value=5.0)
km_driven = st.number_input("KM Driven", min_value=0, value=30000)
car_age = st.number_input("Car Age", min_value=0, value=5)
owner = st.selectbox("Owner", [0, 1, 2, 3])

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
)

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer", "Individual"]
)

transmission = st.selectbox(
    "Transmission",
    ["Manual", "Automatic"]
)

brand = st.text_input("Brand", "Maruti")

input_df = pd.DataFrame(columns=model_columns)
input_df.loc[0] = 0

if "present_price" in input_df.columns:
    input_df.loc[0, "present_price"] = present_price

if "km_driven" in input_df.columns:
    input_df.loc[0, "km_driven"] = km_driven

if "car_age" in input_df.columns:
    input_df.loc[0, "car_age"] = car_age

if "owner" in input_df.columns:
    input_df.loc[0, "owner"] = owner

fuel_column = f"fuel_type_{fuel_type}"
if fuel_column in input_df.columns:
    input_df.loc[0, fuel_column] = 1

seller_column = f"seller_type_{seller_type}"
if seller_column in input_df.columns:
    input_df.loc[0, seller_column] = 1

transmission_column = f"transmission_{transmission}"
if transmission_column in input_df.columns:
    input_df.loc[0, transmission_column] = 1

brand_column = f"brand_{brand}"
if brand_column in input_df.columns:
    input_df.loc[0, brand_column] = 1

if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Selling Price: ₹ {prediction:,.2f}")