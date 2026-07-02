import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("random_forest_model.pkl")   # Apni model file ka naam yahan likhein

# Page Title
st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction")
st.write("Please enter house details below.")

# User Inputs
area = st.number_input("Area (Square Feet)", min_value=0)
bedrooms = st.number_input("Bedrooms",
min_value=1)
bathrooms = st.number_input("Bathrooms", min_value=1)
stories = st.number_input("Stories", min_value=1)
parking = st.number_input("Parking Spaces", min_value=0)

# Predict Button
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        "Area": [area],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Stories": [stories],
        "Parking": [parking]
        })

    prediction = model.predict(input_data)

    st.success(f"Predicted House Price: Rs. {prediction[0]:,.2f}") 