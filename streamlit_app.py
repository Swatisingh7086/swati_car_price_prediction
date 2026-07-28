
import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

model=pickle.load(open("LinearRegressionModel.pkl","rb"))
model_columns=pickle.load(open("model_columns.pkl","rb"))
df=pd.read_csv("Cleaned_Car_data.csv")

st.title("🚗 Car Price Prediction")

brand=st.selectbox("Brand",sorted(df["brand"].unique()))
model_name=st.selectbox("Model",sorted(df[df["brand"]==brand]["model"].unique()))

km_driven=st.number_input("KM Driven",0,1000000,50000)
car_age=st.number_input("Car Age",0,50,5)
owner=st.selectbox("Owner",sorted(df["owner"].unique()))

fuel_col="fuel" if "fuel" in df.columns else "fuel_type"
fuel=st.selectbox("Fuel Type",sorted(df[fuel_col].unique()))
seller=st.selectbox("Seller Type",sorted(df["seller_type"].unique()))
trans=st.selectbox("Transmission",sorted(df["transmission"].unique()))

if st.button("Predict Price"):
    X=pd.DataFrame(0,index=[0],columns=model_columns)
    for c,v in [("car_age",car_age),("km_driven",km_driven)]:
        if c in X.columns: X.at[0,c]=v
    for pref,val in [(fuel_col,fuel),("seller_type",seller),("transmission",trans),("owner",str(owner)),("brand",brand),("model",model_name)]:
        col=f"{pref}_{val}"
        if col in X.columns:
            X.at[0,col]=1
        elif pref=="owner" and "owner" in X.columns:
            X.at[0,"owner"]=owner
    pred=model.predict(X)[0]
    st.success(f"Estimated Selling Price: ₹ {pred:,.2f}")
