# app.py
import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and preprocessors
model = joblib.load("career_prediction_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoder = joblib.load("label_encoder.joblib")

# App title
st.title("🎓 Career Prediction App")

# Collect input features from user
st.header("Enter Your Details")

# Assuming you know the order and meaning of the features
feature_names = ['Logical_Reasoning', 'Leadership', 'Coding', 'Communication_Skills', 
                 'Problem_Solving', 'Memory', 'Decision_making', 'Creativity', 
                 'Technical_Skills', 'Presentation_Skills']

user_input = []
for feature in feature_names:
    value = st.slider(f"{feature.replace('_', ' ')}", 1, 10, 5)
    user_input.append(value)

# Prediction
if st.button("Predict Career Path"):
    # Preprocess input
    input_scaled = scaler.transform([user_input])
    prediction = model.predict(input_scaled)
    career = label_encoder.inverse_transform(prediction)[0]
    
    st.success(f"🎯 Predicted Career Path: **{career}**")
