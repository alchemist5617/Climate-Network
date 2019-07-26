#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:14:40 2019

@author: Mohammad Noorabakhsh
"""

from .climate_network import ClimateNetwork
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import numpy as np
import networkx as nx
from sklearn.preprocessing import Normalizer


class network_optimizer():
    
    @staticmethod        
    def optimizer(self,data, correlation_type = "pearson",model_type = "RF",
                 data_type = "Phase Averaging"):
        
        MSE = []
        r = np.arange(0.05,1,0.05)
        
        if model_type == "NN":
            reg = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,25,10, ), random_state=1)
        elif model_type == "RF":
            reg = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
        else:
            reg = linear_model.LinearRegression() 
        
        if data_type == "PA":
            data_deseasonal, data_seasonal = ClimateNetwork.phase_averaging(data[:-1,:])
            data_seasonal = data_seasonal[(-1) % 12,:]
        elif data_type == "STL":
            data_deseasonal, data_seasonal = ClimateNetwork.STL_decomposition(data[:-1,:])
            data_seasonal = data_seasonal[(-1) % 12,:]
        else:
            data_deseasonal = data[-1,:]
            data_seasonal = np.zeros(data.shape[1])
            
        m = ClimateNetwork.weight_calculator(data_deseasonal,correlation_type)    
    
        for i in range(len(r)):
            
            G, A, C = ClimateNetwork.graph_builder_limit(m, r[i])
            X = self.feature_extractor(G) 
            if model_type == "LR":
                transformer = Normalizer().fit(X)
                X = transformer.transform(X)
    
            Y = data[-1,:]- data_seasonal
            Y_test = data[-1,:]
            reg.fit(X, Y)
            y = reg.predict(X)
            y = y + data_seasonal
            MSE.append(mean_squared_error(Y_test, y))    
            
        return(r[np.argmin(MSE)])
    
    @staticmethod          
    def feature_extractor(G):
        degree = dict(G.degree())
        closseness = nx.closeness_centrality(G)
        kcore = dict(nx.core_number(G))
        betweeness= dict(nx.betweenness_centrality(G))
        pagerank = dict(nx.pagerank(G, alpha=0.85))
        eigenvector = dict(nx.eigenvector_centrality(G, max_iter = 1000))
        knn = nx.average_neighbor_degree(G)
        
        n = len(G)
        X = []
        for j in range(n):
            if (j in degree.keys()):
                x = []
                x.append(degree[j])
                x.append(closseness[j])
                x.append(kcore[j])
                x.append(betweeness[j])
                x.append(pagerank[j])
                x.append(eigenvector[j])
                x.append(knn[j])            
                x.append(nx.clustering(G, j))
                X.append(x)
        return(X)