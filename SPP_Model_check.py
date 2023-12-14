import sys
import pandas as pd
import datetime as dt
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt


# data preparation
def data_prep(string):
    stock = string
    start = dt.datetime(2013, 1, 1)
    df = yf.download(stock, start)
    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    def clean_values(col):
        for i in range(len(col)):
            if type(col[i]) == str:
                col[i] = float(''.join(char for char in col[i]
                               if char.isalnum() or char == "."))

        return col
    df.set_index('Date', inplace=True)
    arr = list(df.columns)
    for i in range(5):
        cv = list(df[arr[i]])
        cv = clean_values(cv)
        df.loc[:, (arr[i])] = cv
    return df

# Train Test Split and RandomForest


def RandFor(Stock):
    Stock["Tomorrow"] = Stock["Close"].shift(-1)
    Stock = Stock[Stock['Tomorrow'] > 0]
    Stock["Target"] = (Stock["Tomorrow"] > Stock["Close"]).astype(int)
    # splitting the data
    x = Stock[['Open', 'Close', 'High', 'Low', 'Volume', 'Tomorrow']]
    y = Stock[['Target']]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=101)

    model = RandomForestClassifier(
        n_estimators=15, min_samples_split=10, random_state=42)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    prediction = pd.Series(prediction, index=x_test.index, name="Predictions")
    combined = pd.concat([y_test["Target"], prediction], axis=1)
    pred_matches = (combined['Target'] == combined['Predictions']).sum()
    total_values = len(combined)
    percentage = pred_matches / total_values * 100

    # Predictions for the last 5 entries
    last_entries = Stock.iloc[-5:]
    last_entries_x = last_entries[[
        'Open', 'Close', 'High', 'Low', 'Volume', 'Tomorrow']]
    last_entries_predictions = model.predict(last_entries_x)
    last_entries["Predictions"] = last_entries_predictions
    return last_entries["Predictions"],  percentage


def LSTM_stock_prices(csv_path):
    # Read data from CSV file
    df = pd.read_csv(csv_path)

    # Extract relevant columns and format date column
    df = df[['Date', 'Open', 'Close']]
    df['Date'] = pd.to_datetime(df['Date'].apply(lambda x: x.split()[0]))
    df.set_index('Date', drop=True, inplace=True)

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df)
    training_size = round(len(df) * 0.80)
    train_data = df[:training_size]
    test_data = df[training_size:]

    # Define function to create sequences and labels for training
    def create_sequence(dataset, window_size):
        sequences = []
        labels = []
        start_idx = 0
        for stop_idx in range(window_size, len(dataset)):
            sequences.append(dataset.iloc[start_idx:stop_idx])
            labels.append(dataset.iloc[stop_idx])
            start_idx += 1
        return (np.array(sequences), np.array(labels))

    # Create training and testing sequences
    window_size = 50
    train_seq, train_label = create_sequence(train_data, window_size)
    test_seq, test_label = create_sequence(test_data, window_size)

    # Define and train the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(
        train_seq.shape[1], train_seq.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['mean_absolute_error'])
    model.fit(train_seq, train_label, epochs=80,
              validation_data=(test_seq, test_label), verbose=1)

    # Make predictions on test data
    test_predicted = model.predict(test_seq)
    test_inverse_predicted = scaler.inverse_transform(test_predicted)

    # Merge predicted and actual data into a single DataFrame
    merged_data = pd.concat([df.iloc[-(len(test_predicted)):].copy(),
                             pd.DataFrame(test_inverse_predicted, columns=['Open_predicted', 'Close_predicted'],
                                          index=df.iloc[-(len(test_predicted)):].index)], axis=1)
    merged_data[['Open', 'Close']] = scaler.inverse_transform(
        merged_data[['Open', 'Close']])

    # Plot the predicted vs actual data for the 'Close' column
    merged_data[['Close', 'Close_predicted']].plot(figsize=(10, 6))
    plt.xticks(rotation=45)
    plt.xlabel('Date', size=15)
    plt.ylabel('Stock Price', size=15)
    plt.title('Actual vs Predicted for Close Price', size=15)
    plt.show()

    # Return the merged data DataFrame
    return merged_data[['Close', 'Close_predicted']].tail(5)


stock_code = input('Yahoo finance Stock code of Stock you would like: ')
stock = data_prep(stock_code)
stock.to_csv('Stock.csv')
Stock = pd.read_csv("Stock.csv", index_col=0)
predictions = RandFor(stock)
Value_predictions = LSTM_stock_prices(Stock)
print(predictions)
print(Value_predictions)
