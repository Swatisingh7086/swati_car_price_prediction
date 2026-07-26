import streamlit as st
import pickle
import pandas as pd
import numpy as np
from model_utils import cast_to_float  # required so pickle.load can find this function

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")

@st.cache_resource
def load_model():
    return pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@st.cache_data
def load_data():
    return pd.read_csv('Cleaned_Car_data.csv')

model = load_model()
car = load_data()

st.title("🚗 Car Price Predictor")
st.write("This app predicts the price of a car you want to sell. Fill in the details below:")

companies = sorted(car['company'].unique())
car_models_all = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = sorted(car['fuel_type'].unique())

company = st.selectbox("Select the company", companies)

# Mirror the original form's JS behavior: only show models belonging to the selected company
models_for_company = sorted(car[car['company'] == company]['name'].unique())
car_model = st.selectbox("Select the model", models_for_company if models_for_company else car_models_all)

year = st.selectbox("Select Year of Purchase", years)
fuel_type = st.selectbox("Select the Fuel Type", fuel_types)
driven = st.number_input("Enter the Number of Kilometres that the car has travelled", min_value=0, step=1000, value=10000)

if st.button("Predict Price"):
    input_df = pd.DataFrame(
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
        data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
    )
    prediction = model.predict(input_df)
    st.success(f"Estimated Price: ₹ {np.round(prediction[0], 2):,}")
