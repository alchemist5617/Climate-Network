#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:27:27 2019

@author: M Noorbakhsh
"""

import numpy as np
from scipy import eye, asarray, dot, sum, diag
from scipy.linalg import svd
import random

def varimax(Phi, gamma = 1.0, q = 1000, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return dot(Phi, R)

def orthomax(U, rtol = np.finfo(np.float32).eps ** 0.5, gamma = 1.0, maxiter = 1000):
    """
    Rotate the matrix U using a varimax scheme.  Maximum no of rotation is 1000 by default.
    The rotation is in place (the matrix U is overwritten with the result).  For optimal performance,
    the matrix U should be contiguous in memory.  The implementation is based on MATLAB docs & code,
    algorithm is due to DN Lawley and AE Maxwell.
    """
    n,m = U.shape
    Ur = U.copy(order = 'C')
    ColNorms = np.zeros((1, m))
    
    dsum = 0.0
    for indx in range(maxiter):
        old_dsum = dsum
        np.sum(Ur**2, axis = 0, out = ColNorms[0,:])
        C = n * Ur**3
        if gamma > 0.0:
            C -= gamma * Ur * ColNorms  # numpy will broadcast on rows
        L, d, Mt = svd(np.dot(Ur.T, C), False, True, True)
        R = np.dot(L, Mt)
        dsum = np.sum(d)
        np.dot(U, R, out = Ur)
        if abs(dsum - old_dsum) / dsum < rtol:
            break
        
    # flip signs of components, where max-abs in col is negative
    for i in range(m):
        if np.amax(Ur[:,i]) < -np.amin(Ur[:,i]):
            Ur[:,i] *= -1.0
            R[i,:] *= -1.0
            
    return Ur, R, indx

def uni_deseasonalize(ts,freq=12):
    ts = np.array(ts)
    N = len(ts)
    #averages = np.zeros((freq,n))
    temp = ts
    result = np.zeros((N))
    for j in range(freq):
        Idx = np.arange(j,N,freq)
        result[Idx] = temp[Idx] - temp[Idx].mean()
    return(result) 

def deseasonalize(data,freq=12):
    n  = data.shape[1]
    N  = data.shape[0]
    averages = np.zeros((freq,n))
    data_deseasonal = np.zeros(data.shape)
    for i in range(n):
        temp = data[:,i]
        result = np.zeros((N))
        for j in range(freq):
            Idx = np.arange(j,N,freq)
            averages[j,i] = temp[Idx].mean()
            result[Idx] = (temp[Idx] - temp[Idx].mean())/temp[Idx].std()
        data_deseasonal[:,i] = result
    return(data_deseasonal) 
    
def deseasonalize_NoStd(data,freq=12):
    n  = data.shape[1]
    N  = data.shape[0]
    averages = np.zeros((freq,n))
    data_deseasonal = np.zeros(data.shape)
    for i in range(n):
        temp = data[:,i]
        result = np.zeros((N))
        for j in range(freq):
            Idx = np.arange(j,N,freq)
            averages[j,i] = temp[Idx].mean()
            result[Idx] = temp[Idx] - temp[Idx].mean()
        data_deseasonal[:,i] = result
    return(data_deseasonal)

def index_finder(loading,col,percent = 0.98):
    values = loading[col].sort_values(ascending = False)
    s = 0
    i = 0
    while s < percent:
        s+=values[i]
        i = i+1
    Idx = values[:i].index
    
    return(values[:i],Idx)

def index_finder_percentile(loading,col,percentile = 0.9):
    threshold = loading[col].quantile(percentile)
    values = loading[col].values   
    Idx = np.where(values>=threshold)[0]
    return(values[values>=threshold],Idx)

def random_color(n):
    result = []
    for i in range(n):
        r = lambda: random.randint(0,255)
        result.append('#%02X%02X%02X' % (r(),r(),r()))
    return(result)
