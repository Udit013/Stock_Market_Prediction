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
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os

# data preparation
def data_prep(string):
   stock = string
   start = dt.datetime(2013,1,1)
   df = yf.download(stock,start)
   df.reset_index(inplace = True)
   df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
   def clean_values(col) :
      for i in range(len(col)):
         if type(col[i]) == str :
            col[i] = float(''.join(char for char in col[i] if char.isalnum() or char=="."))

      return col
   df.set_index('Date',inplace = True) 
   arr = list(df.columns)
   for i in range(5):
      cv = list(df[arr[i]])
      cv = clean_values(cv)
      df.loc[:,(arr[i])]= cv
   return df

# Train Test Split and RandomForest
def RandFor(Stock):  
   Stock["Tomorrow"] = Stock["Close"].shift(-1)
   Stock = Stock[Stock['Tomorrow']>0]
   Stock["Target"] = (Stock["Tomorrow"] > Stock["Close"]).astype(int)
   # splitting the data
   x = Stock[['Open', 'Close', 'High', 'Low', 'Volume','Tomorrow']]
   y = Stock[['Target']]
   x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.1,random_state=101)

   model = RandomForestClassifier(n_estimators=15, min_samples_split=10, random_state=42)
   model.fit(x_train, y_train)
   prediction = model.predict(x_test)
   prediction = pd.Series(prediction, index=x_test.index, name="Predictions")
   combined = pd.concat([y_test["Target"], prediction], axis=1)
   pred_matches = (combined['Target'] == combined['Predictions']).sum()
   total_values = len(combined)
   percentage = pred_matches / total_values * 100
   
   
   # Predictions for the last 5 entries
   last_entries = Stock.iloc[-5:]
   last_entries_x = last_entries[['Open', 'Close', 'High', 'Low', 'Volume', 'Tomorrow']]
   last_entries_predictions = model.predict(last_entries_x)
   last_entries["Predictions"] = last_entries_predictions
   return last_entries["Predictions"] ,  percentage

def LSTM_MODEL(STOCK): 
   df = STOCK
   df = df [['Open','Close']] 

   # Normalize data using MinMaxScaler
   Ms = MinMaxScaler()
   df [df .columns] = Ms.fit_transform(df )
   training_size = round(len(df ) * 0.80)
   train_data = df [:training_size]
   test_data  = df [training_size:]
   
   # Define function to create sequences and labels for training
   def create_sequence(dataset,window_size):
      sequences=[]
      labels=[]
      start_idx=0
      for stop_idx in range(window_size,len(dataset)):
         sequences.append (dataset.iloc[ start_idx:stop_idx])
         labels.append(dataset.iloc[stop_idx])
         start_idx += 1
      return (np.array(sequences),np.array(labels))  

   # Create training and testing sequences
   window_size = 50
   train_seq, train_label = create_sequence(train_data,window_size)
   test_seq, test_label = create_sequence(test_data,window_size)
   
   # Define and train the LSTM model
   model = Sequential()
   model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

   model.add(Dropout(0.1)) 
   model.add(LSTM(units=50))
   model.add(Dense(2))
   model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
   model.fit(train_seq, train_label, epochs=80,validation_data=(test_seq, test_label), verbose=1)

   # Make predictions on test data
   test_predicted = model.predict(test_seq)
   test_inverse_predicted = Ms.inverse_transform(test_predicted)

   # Merge predicted and actual data into a single DataFrame
   merged_data = pd.concat([df .iloc[-(len(test_predicted)):].copy(),pd.DataFrame(test_inverse_predicted,columns=['Open_predicted','Close_predicted'],index=df .iloc[-(len(test_predicted)):].index)], axis=1)

   merged_data[['Open','Close']] = Ms.inverse_transform(merged_data[['Open','Close']])

   final_ans =  merged_data[['Close','Close_predicted']].tail()
   # print(sqrt(mean_squared_error(merged_data[['Close']],merged_data[['Close_predicted']])))
   # Plot the predicted vs actual data for the 'Close' column
   merged_data[['Close','Close_predicted']].plot(figsize=(10,6))
   plt.xticks(rotation=45)
   plt.xlabel('Date',size=15)
   plt.ylabel('Stock Price',size=15)
   plt.title('Actual vs Predicted for Close Price',size=15)
   plt.show()
   # return final_ans
   return final_ans


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Stock Prediction'
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Input label and textbox
        self.label_stock_code = QLabel("Yahoo Finance Stock Code:")
        self.textbox_stock_code = QLineEdit(self)
        layout.addWidget(self.label_stock_code)
        layout.addWidget(self.textbox_stock_code)

        # Run button
        self.button = QPushButton('Run Analysis', self)
        self.button.clicked.connect(self.run_analysis)
        layout.addWidget(self.button)

        # Output
        self.label_output = QLabel("Output:")
        self.output = QTextEdit(self)
        self.output.setReadOnly(True)
        layout.addWidget(self.label_output)
        layout.addWidget(self.output)

        # Graph
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def run_analysis(self):
        stock_code = self.textbox_stock_code.text()
        stock = data_prep(stock_code)
        stock.to_csv('Stock.csv')
        Stock = pd.read_csv("Stock.csv", index_col=0)
        predictions, randFor_percentage = RandFor(stock)
        value_predictions = LSTM_MODEL(stock)

        # Display results in the output textbox
        self.output.clear()
        self.output.append(f"Random Forest Predictions:\n{predictions}")
        self.output.append(f"Random Forest Accuracy: {randFor_percentage:.2f}%")
        self.output.append(f"\nLSTM Predictions:\n{value_predictions}")

        # Display the graph in the GUI
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        graph_data = value_predictions
        graph_data.plot(ax=ax, figsize=(10, 6))
        ax.set_xticklabels(graph_data.index, rotation=45)
        ax.set_xlabel('Date', size=15)
        ax.set_ylabel('Stock Price', size=15)
        ax.set_title('Actual vs Predicted for Close Price', size=15)
        self.canvas.draw()
  

if __name__ == '__main__':
   app = QApplication(sys.argv)
   ex = App()
   ex.show()
   sys.exit(app.exec_())

   stock_code = input('Yahoo finance Stock code of Stock you would like: ')
   stock = data_prep(stock_code)
   stock.to_csv('Stock.csv')
   Stock = pd.read_csv("Stock.csv",index_col=0)
   predictions = RandFor(stock)
   Value_predictions = LSTM_MODEL(stock)
   print(predictions)
   print(Value_predictions)
  
