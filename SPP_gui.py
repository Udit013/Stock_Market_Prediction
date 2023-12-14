import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget
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



# Function to be called when the "Predict" button is clicked
def on_predict():
    stock_code = stock_code_entry.text()
    if stock_code:
        stock = data_prep(stock_code)
        stock.to_csv('Stock.csv')
        Stock = pd.read_csv("Stock.csv", index_col=0)
        predictions = RandFor(stock)
        output_text.clear()
        output_text.insertPlainText(f"Predictions:\n{predictions}")
    else:
        output_text.clear()
        output_text.insertPlainText("Please enter a valid stock code.")



# Creating the main window
app = QApplication(sys.argv)
app.setStyleSheet("""
    QMainWindow {
        background-color: #1E1F26;
    }

    QLabel {
        color: #F7F7F7;
        font: bold 14px;
    }

    QLineEdit {
        background-color: #2C2E3B;
        color: #F7F7F7;
        font: bold 14px;
        border: 1px solid #3A3D4A;
        border-radius: 5px;
        padding: 5px;
    }

    QPushButton {
        background-color: #2D98DA;
        color: #FFFFFF;
        font: bold 14px;
        border: none;
        border-radius: 5px;
        padding: 5px;
        min-width: 100px;
    }

    QPushButton:hover {
        background-color: #268AB4;
    }

    QTextEdit {
        background-color: #2C2E3B;
        color: #F7F7F7;
        font: bold 14px;
        border: 1px solid #3A3D4A;
        border-radius: 5px;
        padding: 5px;
    }
""")

window = QMainWindow()
window.setWindowTitle("Stock Predictor")

# Creating the central widget and layout
central_widget = QWidget()
layout = QVBoxLayout()

# Adding widgets
stock_code_label = QLabel("Enter Yahoo Finance Stock Code:")
layout.addWidget(stock_code_label)

stock_code_entry = QLineEdit()
layout.addWidget(stock_code_entry)

predict_button = QPushButton("Predict")
predict_button.clicked.connect(on_predict)
layout.addWidget(predict_button)

output_label = QLabel("Output:")
layout.addWidget(output_label)

output_text = QTextEdit()
output_text.setReadOnly(True)
layout.addWidget(output_text)

# Setting the layout to the central widget and adding it to the main window
central_widget.setLayout(layout)
window.setCentralWidget(central_widget)

# Showing the main window and running the event loop
window.show()
sys.exit(app.exec_())