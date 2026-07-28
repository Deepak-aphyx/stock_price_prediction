# Import necessary libraries
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# Fetch stock data
df = pdr.get_data_tiingo(
    'AAPL',
    api_key='YOUR_TIINGO_API_KEY'
)

# Save and read (optional)
df.to_csv("2) Stock Prices Data Set.csv")

# Closing prices
df1 = df.reset_index()['close']

# Plot original data
plt.figure(figsize=(10,5))
plt.plot(df1)
plt.title("Apple Closing Price")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))

# Split data
training_size = int(len(df1) * 0.65)

train_data = df1[:training_size]
test_data = df1[training_size:]

# Create dataset
def create_dataset(dataset, time_step=1):
    X, Y = [], []

    for i in range(len(dataset)-time_step-1):
        X.append(dataset[i:(i+time_step),0])
        Y.append(dataset[i+time_step,0])

    return np.array(X), np.array(Y)

time_step = 100

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM Model
model = Sequential([
    Input(shape=(100,1)),
    LSTM(50, return_sequences=True),
    LSTM(50, return_sequences=True),
    LSTM(50),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Train model
model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=64,
    verbose=1
)

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Convert back to original scale
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

train_actual = scaler.inverse_transform(y_train.reshape(-1,1))
test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

# RMSE
train_rmse = math.sqrt(mean_squared_error(train_actual, train_predict))
test_rmse = math.sqrt(mean_squared_error(test_actual, test_predict))

print("Train RMSE:", train_rmse)
print("Test RMSE :", test_rmse)

# Plot predictions
look_back = 100

train_predict_plot = np.empty_like(df1)
train_predict_plot[:] = np.nan
train_predict_plot[
    look_back:len(train_predict)+look_back
] = train_predict

test_predict_plot = np.empty_like(df1)
test_predict_plot[:] = np.nan

start = len(train_predict) + (look_back * 2) + 1
end = start + len(test_predict)

test_predict_plot[start:end] = test_predict

plt.figure(figsize=(12,6))
plt.plot(
    scaler.inverse_transform(df1),
    label="Actual Price"
)

plt.plot(
    train_predict_plot,
    label="Training Prediction"
)

plt.plot(
    test_predict_plot,
    label="Testing Prediction"
)

plt.legend()
plt.show()

# Future prediction

n_steps = 100

if len(test_data) < n_steps:
    raise ValueError("Not enough test data for prediction.")

temp_input = test_data[-n_steps:].flatten().tolist()

lst_output = []

future_days = 10

for i in range(future_days):

    x_input = np.array(temp_input[-n_steps:])
    x_input = x_input.reshape(1, n_steps, 1)

    yhat = model.predict(x_input, verbose=0)

    temp_input.append(yhat[0][0])
    lst_output.append(yhat[0][0])

lst_output = np.array(lst_output).reshape(-1,1)

future_prediction = scaler.inverse_transform(lst_output)

print("Next 10 Days Prediction")
print(future_prediction)

# Plot future prediction
day_new = np.arange(1,101)
day_pred = np.arange(101,111)

plt.figure(figsize=(10,5))

plt.plot(
    day_new,
    scaler.inverse_transform(df1[-100:]),
    label="Last 100 Days"
)

plt.plot(
    day_pred,
    future_prediction,
    label="Next 10 Days Prediction"
)

plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.title("Future Stock Price Prediction")
plt.legend()
plt.grid(True)
plt.show()
