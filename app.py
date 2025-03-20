# Import required libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and preprocessing tools
model = joblib.load('trained_models/random_forest_model.pkl')
scaler = joblib.load('trained_models/scaler.pkl')
label_encoder_target = joblib.load('trained_models/label_encoder_target.pkl')
label_encoder_season = joblib.load('trained_models/label_encoder_season.pkl')
footer_text = "Made with 💜🤍 for Kavuuuu"

# Define the Streamlit app
def main():
    st.title("Crop Recommendation System")
    st.write("Enter the parameters below to predict the best crop to plant.")

    # Input fields for user
    N = st.slider("Nitrogen (N)", min_value=0, max_value=150, value=50)
    P = st.slider("Phosphorus (P)", min_value=0, max_value=150, value=50)
    K = st.slider("Potassium (K)", min_value=0, max_value=150, value=50)
    temperature = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
    ph = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
    rainfall = st.slider("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0, step=0.1)
    season = st.selectbox("Season", ["Kharif", "Rabi", "Summer"])
    planting_month = st.slider("Planting Month", min_value=1, max_value=12, value=6)
    harvesting_month = st.slider("Harvesting Month", min_value=1, max_value=12, value=10)

    # Encode categorical features
    season_encoded = label_encoder_season.transform([season])[0]

    # Create input array for prediction
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall, planting_month, harvesting_month, season_encoded]])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the crop
    if st.button("Predict Crop"):
        prediction = model.predict(input_data_scaled)[0]
        predicted_crop = label_encoder_target.inverse_transform([prediction])[0]
        st.success(f"The recommended crop is: **{predicted_crop}**")

# Run the app
if __name__ == "__main__":
    main()
