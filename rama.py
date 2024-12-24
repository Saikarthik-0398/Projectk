import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# Load pre-trained models and scalers
clr = load("random_forest_model.joblib")
scaler = load("scaler.joblib")
labelencoder = load("label_encoder.joblib")

# Define feature names
feature_names = [
    'Alcoholuse', 'GeneticRisk', 'chronicLungDisease',
    'Smoking', 'ChestPain', 'CoughingofBlood', 'Fatigue',
    'WeightLoss', 'ShortnessofBreath', 'SwallowingDifficulty',
    'FrequentCold', 'DryCough'
]

# Streamlit App
st.title("Health Risk Prediction App")

st.sidebar.header("Input Features")
# Collect inputs from the user for each feature
inputs = {}
for feature in feature_names:
    inputs[feature] = st.sidebar.slider(feature, 1, 10, 5)

# Convert inputs to a DataFrame
input_data = pd.DataFrame([inputs])

# Predict button
if st.button("Predict"):
    # Scale inputs
    x_test = scaler.transform(input_data)

    # Make prediction
    y_pred = clr.predict(x_test)
    prediction = labelencoder.inverse_transform(y_pred)[0]

    # Show prediction
    st.success(f"Predicted Outcome: {prediction}")

    # Feature Importance (if applicable)
    if hasattr(clr, "feature_importances_"):
        feature_importances = clr.feature_importances_ * 100
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance (%)": feature_importances
        }).sort_values(by="Importance (%)", ascending=False)

        st.subheader("Feature Importances")
        st.table(importance_df)
