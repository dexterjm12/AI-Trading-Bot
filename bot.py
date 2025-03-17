import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf  # Fetch market data

# Load market data
def load_data(symbol="AAPL", period="1y", interval="1d"):
    data = yf.download(symbol, period=period, interval=interval)
    return data["Close"].values.reshape(-1, 1)  # Using closing prices

# Preprocess data
def preprocess_data(data, window_size=30):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - window_size):
        X.append(data_scaled[i:i+window_size])
        y.append(data_scaled[i+window_size])

    return np.array(X), np.array(y), scaler

# Create AI model
def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        keras.layers.LSTM(50, return_sequences=False),
        keras.layers.Dense(25, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Train the model
def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return model

# Predict market trends
def predict(model, latest_data, scaler):
    latest_data_scaled = scaler.transform(latest_data)
    prediction = model.predict(np.array([latest_data_scaled]))
    return scaler.inverse_transform(prediction)[0][0]

# Execute trades based on AI signals
def execute_trade(current_price, predicted_price):
    if predicted_price > current_price:
        print("Buying stocks...")
    else:
        print("Selling stocks...")

# Main execution
data = load_data()
X_train, y_train, scaler = preprocess_data(data)

# Reshaping data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

model = build_model((X_train.shape[1], 1))
model = train_model(model, X_train, y_train)

# Predict the next price movement
latest_data = data[-30:]  # Use last 30 days as input
predicted_price = predict(model, latest_data, scaler)

# Execute trade decision
current_price = data[-1][0]
execute_trade(current_price, predicted_price)