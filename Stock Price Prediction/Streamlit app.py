# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import math
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.title("Stock Price Prediction and Comparison")

tab1, tab2 = st.tabs(["Stock 1", "Stock 2"])

# Scrapping data
start = "2010-1-1"
end = "2019-12-31"

with tab1:

    st.markdown("Stock 1")

    # Taking Input from the user
    user_input = st.text_input("Enter Stock Ticker", "AAPL")
    df = yf.download(user_input, start=start, end=end)

    # Describing the data
    st.header("Data from 2010-2019")
    st.write(df.describe())
    st.divider()

    # Resetting the index
    df = df.reset_index()
    df = df[["Close"]]

    # Plotting the Closing price over the years
    st.header("Graph of Closing Price of the stock")
    fig = plt.figure(figsize=(10,6))
    plt.plot(df)
    st.pyplot(fig)

    st.divider()

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0,1))
    df = scaler.fit_transform(np.array(df).reshape(-1,1))

    # Splitting the data into train and test
    training_size = int(len(df) * 0.65)
    testing_size = len(df) - training_size
    train_data, test_data = df[0:training_size, : ] , df[training_size : len(df), :1]

    # Preprocessing the data to convert it into a dataset of dependant and independant variables based on the timesteps

    # Creating a function to convert an array into a dataset matrix based on the time step value
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i+time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    # Reshaping the X_train and X_test data into 3 dimensions for the LSTM model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Loading the model
    model = load_model("C:\Investment Compass Assignment\Stock Price Prediction\stock price prediction model.h5")

    # Making predictions for the next 3 days based on the previous 100 days
    x_input = test_data[(len(test_data) - 100) : ].reshape(1, -1)
    temp_input = list(x_input[0])

    # Making predictions for the next 3 days and adding it to the past 100 days data stored in temp_input list.
    lst_output = []
    n_steps = 100
    i = 0
    while (i < 3):

        if (len(temp_input) > 100):
            x_input = np.array(temp_input[1 : ])  # shifting the 100 values by excluding the 0th idx value and including the new predicted value
            # print(f"{i} day input {x_input}")
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            # print(f"{i} day output {yhat}")
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1

        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i += 1

    # Plotting the next 3 days predictions
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 104)

    st.header("Graph of the last 100 days Closing price and next 3 days predictions")
    fig2 = plt.figure(figsize=(10,6))
    plt.plot(day_new, scaler.inverse_transform(df[(len(df) - 100): ]), label="Last 100 days")
    plt.plot(day_pred, scaler.inverse_transform(lst_output), label="Next 3 days Prediction")
    plt.xlabel("Number of Days")
    plt.ylabel("Closing Price")
    plt.title("Stock Price Graph")
    plt.legend()
    st.pyplot(fig2)


with tab2:
    st.markdown("Stock 2")

    # Taking Input from the user
    user_input = st.text_input("Enter Stock Ticker", "TSLA")
    df = yf.download(user_input, start=start, end=end)

    # Describing the data
    st.header("Data from 2010-2019")
    st.write(df.describe())
    st.divider()

    # Resetting the index
    df = df.reset_index()
    df = df[["Close"]]

    # Plotting the Closing price over the years
    st.header("Graph of Closing Price of the stock")
    fig = plt.figure(figsize=(10,6))
    plt.plot(df)
    st.pyplot(fig)

    st.divider()

    # Scaling the data
    scaler = MinMaxScaler(feature_range=(0,1))
    df = scaler.fit_transform(np.array(df).reshape(-1,1))

    # Splitting the data into train and test
    training_size = int(len(df) * 0.65)
    testing_size = len(df) - training_size
    train_data, test_data = df[0:training_size, : ] , df[training_size : len(df), :1]

    # Preprocessing the data to convert it into a dataset of dependant and independant variables based on the timesteps

    # Creating a function to convert an array into a dataset matrix based on the time step value
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i+time_step, 0])
        return np.array(dataX), np.array(dataY)

    time_step = 100
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    # Reshaping the X_train and X_test data into 3 dimensions for the LSTM model
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Loading the model
    model = load_model("C:\Investment Compass Assignment\Stock Price Prediction\stock price prediction model.h5")

    # Making predictions for the next 3 days based on the previous 100 days
    x_input = test_data[(len(test_data) - 100) : ].reshape(1, -1)
    temp_input = list(x_input[0])

    # Making predictions for the next 3 days and adding it to the past 100 days data stored in temp_input list.
    lst_output = []
    n_steps = 100
    i = 0
    while (i < 3):

        if (len(temp_input) > 100):
            x_input = np.array(temp_input[1 : ])  # shifting the 100 values by excluding the 0th idx value and including the new predicted value
            # print(f"{i} day input {x_input}")
            x_input = x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            # print(f"{i} day output {yhat}")
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i += 1

        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            # print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            # print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i += 1

    # Plotting the next 3 days predictions
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 104)

    st.header("Graph of the last 100 days Closing price and next 3 days predictions")
    fig2 = plt.figure(figsize=(10,6))
    plt.plot(day_new, scaler.inverse_transform(df[(len(df) - 100): ]), label="Last 100 days")
    plt.plot(day_pred, scaler.inverse_transform(lst_output), label="Next 3 days Prediction")
    plt.xlabel("Number of Days")
    plt.ylabel("Closing Price")
    plt.title("Stock Price Graph")
    plt.legend()
    st.pyplot(fig2)