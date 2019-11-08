#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:55:16 2019

@author: mathsys2
"""

import numpy as np
import math

np.random.seed(1360)

def tau_computer(t1_index, i):
    if (len(t1_index) != 1):  
        if(i == 0):
            tau_t1 = t1_index[i+1] - t1_index[i]
        elif(i == (len(t1_index)-1)):
            tau_t1 = t1_index[i] - t1_index[i-1]
        else:
            tau_t1 = min(t1_index[i+1] - t1_index[i],t1_index[i] - t1_index[i-1])
    else:
        tau_t1 = float('NaN')
    return(tau_t1)
    
def ES(t1, t2, t_max):
    t1_index = np.where((t1 == 0) & (np.roll(t1,-1) == 1))[0] + 1
    t2_index = np.where((t2 == 0) & (np.roll(t2,-1) == 1))[0] + 1
    s = 0
    if((len(t1_index)!=0) and (len(t2_index)!=0)): 
        for i in range(len(t1_index)):
            for j in range(len(t2_index)):
                if(t2_index[j] < t1_index[i]): 
                    continue
                d = t2_index[j] - t1_index[i]
                tau_t1 = tau_computer(t1_index, i)   
                tau_t2 = tau_computer(t2_index, j)
                if(np.isnan(tau_t1) and not np.isnan(tau_t2)):
                    tau = tau_t2/2
                elif(not np.isnan(tau_t1) and np.isnan(tau_t2)):
                    tau = tau_t1/2
                elif(np.isnan(tau_t1) and np.isnan(tau_t2)):
                    tau = float('NaN')
                else:
                    tau = min(tau_t1, tau_t2)/2
                
                if(np.isnan(tau)):
                    if(d<=t_max):
                        if(t1_index[i] == t2_index[j]): 
                            s = s + 0.5
                        else:
                            s = s + 1                   
                else:
                    if ((d<=t_max) and (d<=tau)):
                        if(t1_index[i] == t2_index[j]): 
                            s = s + 0.5
                        else:
                            s = s + 1
    if(len(t1_index)>0): s = s/len(t1_index)                        
    return(s)

def ES_matrix(data, t_max):
    lat_number = data.shape[1]
    lon_number = data.shape[2]
    PR = np.zeros((lat_number*lon_number,lat_number*lon_number))
    
    for i in range(lat_number*lon_number):
        lat_index = math.floor(i/lon_number)
        lon_index = i % lon_number
        if np.isnan(data[-1,lat_index ,lon_index]): 
            PR[i,:] = np.nan
            continue
        for j in range(lat_number*lon_number):
            lat_index_sec = math.floor(j/lon_number)
            lon_index_sec = j % lon_number
            if np.isnan(data[-1,lat_index ,lon_index]): 
                PR[i,j] = np.nan
                continue
            else:
                PR[i,j] = ES(data[:,lat_index,lon_index], data[:,lat_index_sec,lon_index_sec], t_max)
    return(PR)


def unif_generator(l):
    n = 0
    lst = []
    lst.append(np.random.randint(2))
    if (lst[-1] == 1): 
        n = 1
    while (l > n):
        r = np.random.randint(2)
        if(r == 1):
            if(lst[-1]==0):
                lst.append(r)
                n = n + 1
        else:
            lst.append(r)
    return(np.array(lst))
    
def event_count(t):
    t_index = np.where((t == 0) & (np.roll(t,-1) == 1))[0] + 1
    return(len(t_index))
    
def event_count_matrix(data):
    lat_number = data.shape[1]
    lon_number = data.shape[2]
    result = np.zeros((lat_number,lon_number))
    
    for i in range(lat_number):
        for j in range(lon_number):
            result[i,j] = event_count(data[:,i,j])
    return(result)


def test(t1_count, t2_count, weight, n, t_max, significance):
    result = []
    for i in range(n):
        sample1 = unif_generator(t1_count)
        sample2 = unif_generator(t2_count)
        sample_weight = ES(sample1, sample2, t_max)
        result.append(sample_weight)
    percentile = np.percentile(np.array(result), significance) 
    if(weight>percentile):
        return(1)
    else:
        return(0)
        
def adjacency_matrix(weighted_matrix, count_matrix, n ,t_max, significance):
    N = weighted_matrix.shape[0]
    lat_number = count_matrix.shape[0]
    lon_number = count_matrix.shape[1]
    result = np.zeros((N,N))
    
    for i in range(N):        
        lat_index = math.floor(i/lon_number)
        lon_index = i % lon_number
        for j in range(N):
            if(weighted_matrix[i,j] == 0):
                continue;
            lat_index_sec = math.floor(j/lon_number)
            lon_index_sec = j % lon_number
            result[i,j] = test(count_matrix[lat_index, lon_index], count_matrix[lat_index_sec, lon_index_sec],weighted_matrix[i,j], n, t_max, significance)
    return(result)
