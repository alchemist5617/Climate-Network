#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:29:39 2019

@author: Mohammad
"""

import numpy as np
from numpy import array
import tensorflow as tf
#from keras.layers import Dropout  
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
#import matplotlib.pyplot as plt
from scipy import stats
#from scipy.special import inv_boxcox
from sklearn.model_selection import train_test_split

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

data = pd.read_csv("precipitationBWh.csv")
data = data["0"].values


#start = np.arange(5,255,5)
n_steps_lst = np.arange(5,255,5)
lstm_out_lst = np.arange(25,300,50)

x = data
fuzzify(x)
x_transforme, lambda_ = stats.boxcox(x)

steps = []
lstm = []
MSE  = []
R2 = []


for lstm_out in lstm_out_lst:
    for n_steps in n_steps_lst: 
        # define input sequence
        raw_seq = x_transforme
        # choose a number of time steps
        #n_steps = 100
        # split into samples
        X, y = split_sequence(raw_seq, n_steps)
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # define model
        model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], n_features)),
                tf.keras.layers.Dense(1)
                ])
        model.compile(optimizer='adam', loss='mse')
        
        # fit model
        model.fit(X_train, y_train, epochs=100, verbose=0)
        
        y_hat = model.predict(X_test, verbose=0)
        
        steps.append(n_steps)
        lstm.append(lstm_out)
        try:
            MSE.append(mean_squared_error(y_test,y_hat))
            R2.append(r2_score(y_test,y_hat))
        except ValueError:
            print("ValueError happend at step of %d and lstm_out of %d"%(n_steps,lstm_out))
            df = pd.DataFrame({'n_steps': steps[:-1],'lstm_out': lstm[:-1],'MSE': MSE, 'R':R2})
            df.to_csv('tunning.csv')
        
df = pd.DataFrame({'n_steps': steps,'lstm_out': lstm,'MSE': MSE, 'R':R2})
df.to_csv('tunning')     