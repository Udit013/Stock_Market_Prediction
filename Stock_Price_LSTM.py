


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt


df = pd.read_csv('Tesla.csv')



df = df [['Date','Open','Close']] 
df ['Date'] = pd.to_datetime(df ['Date'].apply(lambda x: x.split()[0])) 
df.set_index('Date',drop=True,inplace=True) 



Ms = MinMaxScaler()
df [df .columns] = Ms.fit_transform(df )
training_size = round(len(df ) * 0.80)
train_data = df [:training_size]
test_data  = df [training_size:]



def create_sequence(dataset,window_size):
    sequences=[]
    labels=[]
    start_idx=0
    for stop_idx in range(window_size,len(dataset)):
      sequences.append (dataset.iloc[ start_idx:stop_idx])
      labels.append(dataset.iloc[stop_idx])
      start_idx += 1
    return (np.array(sequences),np.array(labels))  
   

window_size = 50
train_seq, train_label = create_sequence(train_data,window_size)
test_seq, test_label = create_sequence(test_data,window_size)




model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape = (train_seq.shape[1], train_seq.shape[2])))

model.add(Dropout(0.1)) 
model.add(LSTM(units=50))

model.add(Dense(2))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])



model.fit(train_seq, train_label, epochs=80,validation_data=(test_seq, test_label), verbose=1)


test_predicted = model.predict(test_seq)
test_inverse_predicted = Ms.inverse_transform(test_predicted)


merged_data = pd.concat([df .iloc[-(len(test_predicted)):].copy(),pd.DataFrame(test_inverse_predicted,columns=['Open_predicted','Close_predicted'],index=df .iloc[-(len(test_predicted)):].index)], axis=1)


merged_data[['Open','Close']] = Ms.inverse_transform(merged_data[['Open','Close']])




merged_data[['Close','Close_predicted']].tail(5)


print(sqrt(mean_squared_error(merged_data[['Close']],merged_data[['Close_predicted']])))


# merged_data[['Open','Open_predicted']].plot(figsize=(10,6))
# plt.xticks(rotation=45)
# plt.xlabel('Date',size=15)
# plt.ylabel('Stock Price',size=15)
# plt.title('Actual vs Predicted for Open Price',size=15)
# plt.show()


merged_data[['Close','Close_predicted']].plot(figsize=(10,6))
plt.xticks(rotation=45)
plt.xlabel('Date',size=15)
plt.ylabel('Stock Price',size=15)
plt.title('Actual vs Predicted for Close Price',size=15)
plt.show()




