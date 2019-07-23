#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:14:40 2019

@author: Mohammad Noorabakhsh
"""

from .Data import Data
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import networkx as nx

class ClimateNetwork:

    """
    Encapsulates climate network.

    """
    def __init__(self, Data, correlation_type = "pearson",
                 data_type = "Phase Averaging",threshold, 
                 winter_only = False):
        
        
        data = Data.get_data()
        date = Date.get_date()
        
        first_item = 0
        while date[i].month != 12:
            first_item+=1
        
        if data_type == "Phase Averaging":
            data_deseasonal, data_seasonal = _phase_averaging(data)
        elif data_type == "STL Decomposition":
            data_deseasonal, data_seasonal = _STL_decomposition(data)
        else:
            data_deseasonal = data
            data_seasonal = np.zeros(data.shape[1])

        weight_matrix = _weighted_matrix(data_deseasonal,correlation_type,
                                         WINTER_ONLY, first_item)
    
        G,adjacency_matrix, c = _graph_builder(weight_matrix, threshold)
        
        self._graph = G
        self._adjacency_matrix = adjacency_matrix
        
    def _graph_builder (weighted_matrix,threshold):
        
        weighted_matrix = np.nan_to_num(weighted_matrix)
        weighted_matrix = np.absolute(weighted_matrix)
        np.fill_diagonal(weighted_matrix, 0)
        componenets_number = 0
        adjacency_matrix = np.zeros(weighted_matrix.shape)
        adjacency_matrix[weighted_matrix > threshold] = 1
        G = nx.from_numpy_matrix(adjacency_matrix)
        G = G.to_undirected()
        G.remove_edges_from(G.selfloop_edges())
        componenets_number = nx.number_connected_components(G)
        return(G,adjacency_matrix, componenets_number)    
    
    def _STL_decomposition(data,freq=12):
        n = data.shape[1]
        data_deseasonal = np.zeros(data.shape)
        data_seasonal = np.zeros((freq,n))
        for i in range(n):
            decomp = seasonal_decompose(data[:,i], model='additive',freq=freq,extrapolate_trend="freq")
            data_deseasonal[:,i] = decomp.trend + decomp.resid
            data_seasonal[:,i] = decomp.seasonal[:freq]
        return(data_deseasonal, data_seasonal)

    def _phase_averaging(data,freq=12):
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
        return(data_deseasonal, averages)     
        
    def _winter_only(data,first_item):
        n = data.shape[0]
        N = data.shape[1]
        start = first_item + 2
        Idx = np.arange(start,n,12)
        n_new = len(Idx)
        result = np.zeros((n_new*3,N))
        for i in range(N):
            temp = []
            for j in Idx:
                temp.append(data[j -2,i])
                temp.append(data[j -1,i])
                temp.append(data[j,i])
            result[:,i] = np.array(temp)
        return(result)
    
    def _weighted_matrix(data, correlation_type = "pearson", WINTER_ONLY = False, first_item = 0):
        if WINTER_ONLY:
            input_data= _winter_only(data, first_item)
        else:
            input_data = data
        N = input_data.shape[1]
        result = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if correlation_type == "spearmanr":
                    result[i,j] = st.spearmanr(input_data[:,i],input_data[:,j])[0] 
                else:
                    result[i,j] = st.pearsonr(input_data[:,i],input_data[:,j])[0]
        return(result)