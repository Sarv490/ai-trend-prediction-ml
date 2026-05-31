import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# =========================
# Title
# =========================
st.title("📈 AI Tools 12-Week Popularity Prediction")

# =========================
# Load Model
# =========================
model = joblib.load("naive_model.pkl")

# =========================
# Load Dataset
# =========================
df = pd.read_csv("ai_adoption_full.csv")

# =========================
# Show Dataset
# =========================
st.subheader("Dataset Preview")
st.write(df.head())

# =========================
# Last Known Value
# =========================
last_value = df["ChatGPT"].iloc[-1]

# =========================
# Generate 12 Week Prediction
# =========================
weeks = np.arange(1, 13)

# Simple prediction logic
predictions = []

current_value = last_value

for i in weeks:
    predicted = current_value + np.random.randint(-2, 5)
    
    # Keep values positive
    if predicted < 0:
        predicted = 0
        
    predictions.append(predicted)
    current_value = predicted

# =========================
# Prediction DataFrame
# =========================
future_df = pd.DataFrame({
    "Week": weeks,
    "Predicted Popularity": predictions
})

# =========================
# Display Predictions
# =========================
st.subheader("12-Week Popularity Forecast")
st.write(future_df)

# =========================
# Plot Graph
# =========================
fig, ax = plt.subplots(figsize=(10,5))

ax.plot(
    future_df["Week"],
    future_df["Predicted Popularity"],
    marker='o'
)

ax.set_title("12-Week AI Popularity Prediction")
ax.set_xlabel("Week")
ax.set_ylabel("Popularity Score")

st.pyplot(fig)

# =========================
# Download CSV
# =========================
csv = future_df.to_csv(index=False)

st.download_button(
    label="Download Prediction CSV",
    data=csv,
    file_name="12_week_prediction.csv",
    mime="text/csv"
)