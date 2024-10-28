import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
import json

# Firebase configuration
firebase_url = st.secrets["firebase"]["url"]
firebase_auth_token = st.secrets["firebase"]["auth_token"]

# Function to fetch data from Firebase using REST API
def fetch_firebase_data():
    try:
        response = requests.get(f'{firebase_url}/parameters.json?auth={firebase_auth_token}')
        if response.ok:
            entries = response.json()
            return entries
        else:
            st.error("Error fetching data from Firebase.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None

# Fetch Firebase data
entries = fetch_firebase_data()

# Load the dataset, model, and initialize the scaler
df = pd.read_csv('DATASET.csv')
model = load_model('SOH_forecast.h5')

features = ['Time' , 'Voltage' , 'Current'] 
target = 'SOH'

scaler = MinMaxScaler()
scaler.fit(df[features + [target]])

# Parse entries if available
if entries:
    # Initialize variables
    current_input = entries.get('current')
    voltage_input = entries.get('voltage')

# Streamlit UI
st.title('SOH Prediction with LSTM')

# Display sensor inputs
st.write(f"Voltage from the sensor: {voltage_input}")
st.write(f"Current from the sensor: {current_input}")

# Define the prediction function
def predict_temperature(time, current, voltage):
    input_data = [time, current, voltage, 0]  # Append a dummy target value
    input_scaled = scaler.transform([input_data])
    reshaped_input = input_scaled[:, :-1].reshape(1, 1, len(features))
    prediction = model.predict(reshaped_input)
    rescaled_prediction = scaler.inverse_transform(
        np.concatenate((reshaped_input[:, 0], prediction), axis=1)
    )[:, -1]
    return rescaled_prediction[0]

# Forecast for the next 5 steps
def forecast_next_5_steps(initial_time):
    initial_sequence = df[df['Time'] <= initial_time][['Time', 'Current', 'Voltage', 'SOH']].values
    initial_scaled = scaler.transform(initial_sequence)
    current_input = initial_scaled[:, :-1].reshape(1, initial_scaled.shape[0], -1)
    
    future_predictions = []
    for _ in range(5):
        predicted_temp = model.predict(current_input)[0, 0]

        new_row = np.zeros((1, len(features) + 1))
        new_row[0, :-1] = current_input[:, -1, :]
        new_row[0, -1] = predicted_temp

        rescaled_temp = scaler.inverse_transform(new_row)[0, -1]
        future_predictions.append(rescaled_temp)

        # Update input sequence for the next step
        next_input = np.concatenate(
            (current_input[:, 1:, :], new_row[:, :-1].reshape(1, 1, -1)), axis=1
        )
        current_input = next_input

    return future_predictions

# Use fixed time input for prediction (for demonstration)
time_input = 10

# Predict temperature for current input
predicted_temp = predict_temperature(time_input, current_input, voltage_input)
st.write(f"### Predicted SOH: {predicted_temp:.2f} ")

# Forecast future temperatures
future_temps = forecast_next_5_steps(time_input)
st.write("### Forecasted SOH for the Next 5 Time Steps:")
for i, temp in enumerate(future_temps, start=1):
    st.write(f"Time = {time_input + i}: {temp:.2f} ")

# Plotting the forecasted temperatures
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(int(time_input) + 1, int(time_input) + 6), future_temps, marker='o', label='Predicted SOH')
ax.set_title("Predicted SOH for the Next 5 Time Steps")
ax.set_xlabel("Time")
ax.set_ylabel("SOH")
ax.legend()
st.pyplot(fig)
