#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:22:22 2019

@author: mathsys2
"""

import numpy as np
from numpy import array
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#from keras.layers import Dropout  
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
#import matplotlib.pyplot as plt
from scipy import stats
#from scipy.special import inv_boxcox
#from sklearn.model_selection import train_test_split

def fuzzify(x):
  # Add some "measurement error"" to each data point
    zero_idx = x==0
    x[zero_idx]+=0.005*np.random.uniform(0,1,1)[0]
    x[~zero_idx]+=0.005*np.random.uniform(-1,1,1)[0]
#return(y)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def train_test_split_ts(X, y, test_size=0.2):
    r = X.shape[0]
    split_index = int((1-test_size)*r)
    return(X[:split_index,:,:],X[split_index:,:,:],y[:split_index],y[split_index:])

data = pd.read_csv("precipitationBWh.csv")
data = data["0"].values

name = []
MSE = []
R2 = []

x = data
fuzzify(x)
x_transforme, lambda_ = stats.boxcox(x)

# define input sequence
raw_seq = x_transforme
# choose a number of time steps
n_steps = 30
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

X_train, X_test, y_train, y_test = train_test_split_ts(X, y, test_size=0.2)


#Stacked LSTM
model = keras.models.Sequential()
model.add(layers.LSTM(75, activation='relu',return_sequences=True, input_shape=(X_train.shape[1], n_features)))
model.add(layers.LSTM(75, activation='relu'))
model.add(layers.Dense(1))


model.compile(optimizer='adam',loss='mse')
model.fit(X_train, y_train, epochs=200, verbose=0)
y_hat = model.predict(X_test, verbose=0)

name.append("stacked")
MSE.append(mean_squared_error(y_test,y_hat))
R2.append(r2_score(y_test,y_hat))


#Bidirectional LSTM
model = keras.models.Sequential()
model.add(layers.Bidirectional(layers.LSTM(75, activation='relu',return_sequences=True, input_shape=(X_train.shape[1], n_features))))
model.add(layers.Dense(1))

    
model.compile(optimizer='adam',loss='mse')
model.fit(X_train, y_train, epochs=200, verbose=0)
y_hat = model.predict(X_test, verbose=0)

name.append("Bidirectional")
MSE.append(mean_squared_error(y_test,y_hat))
R2.append(r2_score(y_test,y_hat))




# choose a number of time steps
n_steps = 30
# split into samples
X, y = split_sequence(raw_seq, n_steps)
# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 2
n_steps = 2
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))

X_train, X_test, y_train, y_test = train_test_split_ts(X, y, test_size=0.2)


#CNN LSTM
model = keras.models.Sequential()
model.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
model.add(layers.TimeDistributed(layers.Flatten()))
model.add(layers.LSTM(75, activation='relu'))
model.add(layers.Dense(1))


model.compile(optimizer='adam',loss='mse')
model.fit(X_train, y_train, epochs=100, verbose=0)
y_hat = model.predict(X_test, verbose=0)

name.append("CNN")
MSE.append(mean_squared_error(y_test,y_hat))
R2.append(r2_score(y_test,y_hat))



#ConvLSTM
model = keras.models.Sequential()
model.add(layers.ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu', input_shape=(n_seq, 1, n_steps, n_features)))
model.add(layers.Flatten())
model.add(layers.Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(X_train, y_train, epochs=100, verbose=0)
y_hat = model.predict(X_test, verbose=0)

name.append("ConvLSTM")
MSE.append(mean_squared_error(y_test,y_hat))
R2.append(r2_score(y_test,y_hat))


df = pd.DataFrame({'names': name,'MSE': MSE, 'R':R2})
df.to_csv('models.csv')   

#model = keras.models.Sequential([
#  layers.LSTM(75, activation='relu',return_sequences=True, input_shape=(X_train.shape[1], n_features)),
#  layers.LSTM(75, activation='relu'),
#  layers.Dense(1)
#])
