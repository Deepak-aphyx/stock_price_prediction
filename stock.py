#importing necessary libraries
import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

#function to fetch stock data
df = pdr.get_data_tiingo('AAPL', api_key='8684df2b4150fe0651c2d0fe481c60478d345e15')
df.to_csv('2) Stock Prices Data Set.csv')
df=pd.read_csv('2) Stock Prices Data Set.csv')
df.head()
df2=df.reset_index()['close']
df2[1228:]
df1=df.reset_index()['close']
df1.shape
plt.plot(df1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1,1))
print(df1.shape)
training_size = int(len(df1) * 0.65)
train_data = df1[0:training_size,:]

# Splitting the data into training and testing sets
training_size = int(len(df1) * 0.65)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size,:], df1[training_size:len(df1),:1]

#converting an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

#reshape into X=t and Y=t+1
time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
print(X_train)
print(X_test.shape)
print(y_test.shape)

#reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

#create the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model=Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=100, batch_size=64, verbose=1)

#making predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

#transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

#calculate RMSE
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train, train_predict))
math.sqrt(mean_squared_error(y_test, test_predict)) #Test Data RMSE

#plotting
#shift train predictions for plotting
look_bacl = 100
train_predict_plot = np.empty_like(df1)
train_predict_plot[:, :] = np.nan
train_predict_plot[look_bacl:len(train_predict) + look_bacl, :] = train_predict

#shift test predictions for plotting
test_predict_plot = np.empty_like(df1)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (look_bacl * 2) + 1:len(df1) - 1, :] = test_predict

#plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(train_predict_plot)
plt.plot(test_predict_plot)
plt.show()

len(test_data)
x_input= test_data[341:].reshape(1,-1)
x_input.shape
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# Keep only this correct block:
n_steps = 100
if len(test_data) < n_steps:
    raise ValueError("Not enough test data for prediction window.")

temp_input = test_data[-n_steps:].flatten().tolist()

lst_output = []
for i in range(10):
    x_input = np.array(temp_input[-n_steps:]).reshape(1, n_steps, 1)
    yhat = model.predict(x_input, verbose=0)
    next_pred = yhat[0][0]
    temp_input.append(next_pred)
    lst_output.append(next_pred)

# Convert lst_output to 2D array for inverse_transform
lst_output = np.array(lst_output).reshape(-1, 1)

# Prepare days for plotting
day_new = np.arange(1, 101)
day_pred = np.arange(101, 111)

# Plot last 100 days and next 10 predicted days
plt.plot(day_new, scaler.inverse_transform(df1[-100:]), label='Last 100 Days')
plt.plot(day_pred, scaler.inverse_transform(lst_output), label='Next 10 Days Prediction')
plt.xlabel('Day')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
