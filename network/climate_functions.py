#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 17:49:14 2019

@author: Mohammad Noorbakhsh
"""
import numpy as np
import network_builder as nb
import networkx as nx
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from stldecompose import decompose
import pandas as pd
import math
import scipy.stats as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.x13 import x13_arima_analysis
from sklearn.preprocessing import Normalizer
from netCDF4 import Dataset

def deseasonal_X13(data,start = '1946-01-01',freq = 'M'):
    XPATH = "./x13as"
    n = data.shape[1]
    data_deseasonal = np.zeros(data.shape)
    data_seasonal = np.zeros(data.shape)
    for i in range(n):
        result = x13_arima_analysis(endog = data[:,i],print_stdout=True,start = start,freq = freq, x12path=XPATH,forecast_years=1)
        data_deseasonal[:,i] = result.seasadj.values
       # Idx = result.results.find("Date   Forecast      Error\n   ------------------------------\n   ")
       # data_seasonal[i] = float(result.results[Idx+73:Idx+85].strip())
        data_seasonal[:,i] = data[:,i] - data_deseasonal[:,i]
    return(data_deseasonal, data_seasonal)

def deseasonal_STL(data,freq=12):
    n = data.shape[1]
    data_deseasonal = np.zeros(data.shape)
    data_seasonal = np.zeros((freq,n))
    for i in range(n):
        decomp = seasonal_decompose(data[:,i], model='additive',freq=freq,extrapolate_trend="freq")
        data_deseasonal[:,i] = decomp.trend + decomp.resid
        data_seasonal[:,i] = decomp.seasonal[:freq]
    return(data_deseasonal, data_seasonal)

def deseasonal_STL1(data,freq=12):
    n = data.shape[1]
    data_deseasonal = np.zeros(data.shape)
    data_seasonal = np.zeros(data.shape)
    for i in range(n):
        decomp = decompose(data[:,i], period=freq)
        data_deseasonal[:,i] = decomp.trend + decomp.resid
        data_seasonal[:,i] = decomp.seasonal
    return(data_deseasonal, data_seasonal)

def deseasonal_monthly_anomaly(data,freq=12):
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
    
def moving_average(data, n=10) :
    result = np.cumsum(data, dtype=float)
    result[n:] = result[n:] - result[:-n]
    result = result[n - 1:] / n
    return(np.concatenate([np.repeat(result[0],n-1),result]))    

def deseasonal_monthly_anomaly_MA(data,freq=12,MA=10):
    n  = data.shape[1]
    N  = data.shape[0]
    m = int(N/freq)
    data_deseasonal = np.zeros(data.shape)
    data_seasonal = np.zeros([m,freq,n])
    for i in range(n):
        temp = data[:,i]
        result = np.zeros((N))
        for j in range(freq):
            Idx = np.arange(j,N,freq)
            ma = moving_average(temp[Idx],MA)
            data_seasonal[:,j,i] = ma
            result[Idx] = temp[Idx] - ma
        data_deseasonal[:,i] = result
   #     ma = moving_average(data[:,i],MA)
   #     ma = np.concatenate([np.repeat(ma[0],MA-1),ma])
   #     data_deseasonal[:,i] = data[:,i] - ma
   #     data_seasonal[:,i] = ma
    return(data_deseasonal, data_seasonal) 
   
def deseasonal_curve_fitting(data,degree=4):
    n  = data.shape[1]
    data_deseasonal = np.zeros(data.shape)
    data_seasonal = np.zeros(data.shape)
    for i in range(n):
        series = data[:,i]
        X = [i%12 for i in range(0, len(series))]
        coef = np.polyfit(X, series, degree)
        curve = list()
        for j in range(len(X)):
            value = coef[-1]
            for d in range(degree):
                value += X[i]**(degree-d) * coef[d]
            curve.append(value)
        data_deseasonal[:,i] = series - curve
        data_seasonal[:,i] = curve
    return(data_deseasonal, data_seasonal)

def deseasonal_annual_anomaly(data,freq=12):
    n  = data.shape[1]
    N  = data.shape[0]    
    data_deseasonal = np.zeros(data.shape)
    for i in range(n):
        temp = data[:,i]
        temp_list = []
        for j in range(0,N,freq):   
            temp_list.append(list(temp[j:j+freq] - temp[j:j+freq].mean()))
            flat_list = [item for sublist in temp_list for item in sublist]
        data_deseasonal[:,i] = np.array(flat_list)
    return(data_deseasonal)


def unflatten(data):
    """
    Converts 3D spatio-temporal data inot 2D data
    
    Parameters
    ----------
    data : 3D matrix, shape (longtitude, latitude, time)
    
    Returns
    -------
    result : 2D matrix, shape (time, locaion)
        
    """
    result = np.zeros((data.shape[0]*data.shape[1],data.shape[2]))
    r = data.shape[0]
    c = data.shape[1]
    for i in range(r):
        for j in range(c):
            result[i * c + j,:] = data[i,j,:]
    return(result)
    
def flatten2d(data):
    result = []
    for i in range(0,data.shape[0]):
        for j in range(i+1,data.shape[1]):
            result.append(data[i,j])
    return(result)

def exponential_smoothing(data, alpha):
    """
    Computes expotential smoothing for each time series at each location
    
    Parameters
    ----------
    data : 2D matrix, shape (time, locaion)
    
    alpha: smoothing parameter
    
    Returns
    -------
    result : 1D matrix, shape (locaion)
    
    """
    n = data.shape[0]
    M = data.shape[1]
    result = np.zeros(M)
    for j in range(M):
        y = data[:,j]
        s = np.zeros(n)
        s[0] = y[0]
        for i in range(1,n):
            s[i] = alpha * y[i-1] + (1-alpha)*s[i-1] 
        result[j] = s[-1]
    return(result)

def feature_extractorNew(G):
    degree = dict(G.degree())
    knn = nx.average_neighbor_degree(G)
    pagerank = dict(nx.pagerank(G, alpha=0.85))
    
    n = len(G) 
    X = []
    for j in range(n):
        if (j in degree.keys()):
            x = []
            x.append(degree[j])
            x.append(knn[j])
            x.append(pagerank[j])
            x.append(nx.clustering(G, j))
            X.append(x)
    return(X)

def feature_extractor(G):
    degree = dict(G.degree())
    closseness = nx.closeness_centrality(G)
    kcore = dict(nx.core_number(G))
    betweeness= dict(nx.betweenness_centrality(G))
    pagerank = dict(nx.pagerank(G, alpha=0.85))
    eigenvector = dict(nx.eigenvector_centrality(G, max_iter = 1000))
    knn = nx.average_neighbor_degree(G)
    #eigenvector = dict(nx.eigenvector_centrality_numpy(G))
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

def graph_average_calculator(G):
    average_degree = []
    second_moment = []
    variance = []
    shannon_entropy = []
    transitivity_pr = []
    average_cluster= []
    average_shortest_path_length = []
    average_closeness = []
    average_betweennes = []
    average_eigenvector = []
    average_pagerank =  []
    
    degree = dict(G.degree())
    closseness = nx.closeness_centrality(G)
    kcore = dict(nx.core_number(G))
    betweeness= dict(nx.betweenness_centrality(G))
    pagerank = dict(nx.pagerank(G, alpha=0.85))
    #eigenvector = dict(nx.eigenvector_centrality(G, max_iter = 1000))
    eigenvector = dict(nx.eigenvector_centrality_numpy(G))
    
    average_degree.append(nb.momment_of_degree_distribution(G,1))                  #First Moment
    second_moment.append(nb.momment_of_degree_distribution(G,2))                   #Second Moment
    variance.append(nb.momment_of_degree_distribution(G,2) - nb.momment_of_degree_distribution(G,1)**2)     #Variance
    shannon_entropy.append(nb.shannon_entropy(G))                                  #Shanon Entropy
    transitivity_pr.append(nx.transitivity(G))                                     #Transitivity 
    average_cluster.append(nx.average_clustering(G))
    if nx.is_connected(G) == True:
        average_shortest_path_length.append(nx.average_shortest_path_length(G))     #Average Shortest Path
    else:
        average_shortest_path_length.append(0)
  
    average_closeness.append(np.mean(list(closseness.values())))                       #Average closeness centrality
    average_betweennes.append(np.mean(list(betweeness.values())))                        #Average betweenness centrality
    average_eigenvector.append(np.mean(list(eigenvector.values())))                      #Average eigenvector centrality
    average_pagerank.append(np.mean(list(pagerank.values())))
    
    return(average_degree,second_moment,variance,shannon_entropy,transitivity_pr,average_cluster,
           average_shortest_path_length, average_closeness, average_betweennes, average_eigenvector,
           average_pagerank)

def graph_builder_limit (weighted_matrix,limit):
    weighted_matrix = np.absolute(weighted_matrix)
    componenets_number = 0
    adjacency_matrix = np.zeros(weighted_matrix.shape)
    adjacency_matrix[weighted_matrix >= limit] = 1
    G = nx.from_numpy_matrix(adjacency_matrix)
    G = G.to_undirected()
    G.remove_edges_from(G.selfloop_edges())
    componenets_number = nx.number_connected_components(G)
    return(G, componenets_number)
    
    
    
def model(data, feature = "normal", data_type = "normal", start = 0, end = 841,WINTER_ONLY=False, first_dec=0, correlation_type="pearsonr"): 
    
    r = np.arange(0.05,1,0.05)
    RMSE_LR = []
    RMSE_NN = []
    RMSE_RF = []
    density = []
    average_shortest_path_length = []
    components = []
            
    if data_type == "PA":
        data_deseasonal, data_seasonal = deseasonal_monthly_anomaly(data[start:end-1,:])
        data_seasonal = data_seasonal[(end-1) % 12,:]
    elif data_type == "STL":
        data_deseasonal, data_seasonal = deseasonal_STL(data[start:end-1,:])
        data_seasonal = data_seasonal[(end-1) % 12,:]
    else:
        data_deseasonal = data[start:end-1,:]
        data_seasonal = np.zeros(data.shape[1])
   
    m = nb.weighted_matrix(data_deseasonal,correlation_type,WINTER_ONLY, first_dec)    
    
    for i in range(len(r)):

        G, c = nb.graph_builder_limit(m, r[i])

        density.append(2 * G.number_of_edges()/(len(G)*(len(G)-1)))
        components.append(c)
        
        if nx.is_connected(G) == True:
            average_shortest_path_length.append(nx.average_shortest_path_length(G))     #Average Shortest Path
        else:
            average_shortest_path_length.append(0)
        
        if feature == "normal":
            X = feature_extractor(G) 
        else:
            X = feature_extractorNew(G)   
            
        for model_type in ["NN","RF","LR"]:
            if model_type == "NN":
                reg = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,25,10, ), random_state=1)
                Y = data[end-1,:]- data_seasonal
                Y_test = data[end-1,:]
                reg.fit(X, Y)
                y = reg.predict(X)
                y = y + data_seasonal
                RMSE_NN.append(math.sqrt(mean_squared_error(Y_test, y)))
            elif model_type == "RF":
                reg = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
                Y = data[end-1,:]- data_seasonal
                Y_test = data[end-1,:]
                reg.fit(X, Y)
                y = reg.predict(X)
                y = y + data_seasonal
                RMSE_RF.append(math.sqrt(mean_squared_error(Y_test, y)))
            else:
                reg = linear_model.LinearRegression() 
                transformer = Normalizer().fit(X)
                X_normalized = transformer.transform(X)
                Y = data[end-1,:]- data_seasonal
                Y_test = data[end-1,:]
                reg.fit(X_normalized, Y)
                y = reg.predict(X_normalized)
                y = y + data_seasonal
                RMSE_LR.append(math.sqrt(mean_squared_error(Y_test, y)))
      
    return(RMSE_LR, RMSE_NN, RMSE_RF, average_shortest_path_length, components, density)
    
def model0(data, model_type = "LR", feature = "normal", data_type = "normal", start = 0, end = 841,WINTER_ONLY=False, first_dec=0, correlation_type="pearsonr"): 
    
    r = np.arange(0.05,1,0.05)
    MSE = []
    density = []
    average_shortest_path_length = []
    components = []
    
    if model_type == "NN":
        reg = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,25,10, ), random_state=1)
    elif model_type == "RF":
        reg = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
    else:
        reg = linear_model.LinearRegression() 
        
    if data_type == "PA":
        data_deseasonal, data_seasonal = deseasonal_monthly_anomaly(data[start:end-1,:])
        data_seasonal = data_seasonal[(end-1) % 12,:]
    elif data_type == "STL":
        data_deseasonal, data_seasonal = deseasonal_STL(data[start:end-1,:])
        data_seasonal = data_seasonal[(end-1) % 12,:]
    else:
        data_deseasonal = data[start:end-1,:]
        data_seasonal = np.zeros(data.shape[1])
   
    m = nb.weighted_matrix(data_deseasonal,correlation_type,WINTER_ONLY, first_dec)    
    
    for i in range(len(r)):

        G, c = nb.graph_builder_limit(m, r[i])

        density.append(2 * G.number_of_edges()/(len(G)*(len(G)-1)))
        components.append(c)
        
        if nx.is_connected(G) == True:
            average_shortest_path_length.append(nx.average_shortest_path_length(G))     #Average Shortest Path
        else:
            average_shortest_path_length.append(0)
        
        if feature == "normal":
            X = feature_extractor(G) 
        else:
            X = feature_extractorNew(G)    
 
        if model_type == "LR":
            transformer = Normalizer().fit(X)
            X = transformer.transform(X)

        Y = data[end-1,:]- data_seasonal
        Y_test = data[end-1,:]
        reg.fit(X, Y)
        y = reg.predict(X)
        y = y + data_seasonal
        MSE.append(mean_squared_error(Y_test, y))

        
    return(MSE,average_shortest_path_length,components, density)


def model1(data, model_type = "LR", feature = "normal", data_type = "normal", start = 0, end = 841,MA=15,WINTER_ONLY=False, first_dec=0, correlation_type="pearsonr"): 
    
    r = np.arange(0.05,1,0.05)
    MSE = []
    density = []
    average_shortest_path_length = []
    
    if model_type == "NN":
        reg = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,25,10, ), random_state=1)
    elif model_type == "RF":
        reg = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
    else:
        reg = linear_model.LinearRegression() 
        
    if data_type == "PA":
        data_deseasonal, data_seasonal = deseasonal_monthly_anomaly(data[start:end-1,:])
        data_seasonal = data_seasonal[(end-1) % 12,:]
    elif data_type == "MA":
        data_deseasonal, data_seasonal = deseasonal_monthly_anomaly_MA(data[start:end-1,:],MA)
        data_seasonal = data_seasonal[-1,(end-1) % 12,:]
    elif data_type == "STL":
        data_deseasonal, data_seasonal = deseasonal_STL(data)
        data_seasonal = data_seasonal[end-1,:]
    else:
        data_deseasonal = data[start:end-1,:]
        data_seasonal = np.zeros(data.shape[1]) 
 #   elif data_type == "CF":
 #       data_deseasonal, data_seasonal = deseasonal_curve_fitting(data)
 #       data_seasonal = data_seasonal[end-1,:]
 
  #   elif data_type == "X13":
 #       data_deseasonal, data_seasonal = deseasonal_X13(data)
 #       data_seasonal = data_seasonal[end-1,:]

   
    m = nb.weighted_matrix(data_deseasonal,correlation_type,WINTER_ONLY, first_dec)     
    
    
    for i in range(len(r)):


        G, c = nb.graph_builder_limit(m, r[i])

        density.append(2 * G.number_of_edges()/(len(G)*(len(G)-1)))
        
        if nx.is_connected(G) == True:
            average_shortest_path_length.append(nx.average_shortest_path_length(G))     #Average Shortest Path
        else:
            average_shortest_path_length.append(0)
        
        if feature == "normal":
            X = feature_extractor(G) 
        else:
            X = feature_extractorNew(G)    
 
        if model_type == "LR":
            transformer = Normalizer().fit(X)
            X = transformer.transform(X)

        if data_type == "MA":
            Y = data[end-1,:] - data_seasonal
        else:
            Y = data[end-1,:]- data_seasonal
        Y_test = data[end-1,:]
        reg.fit(X, Y)
        y = reg.predict(X)
        y = y + data_seasonal
        MSE.append(mean_squared_error(Y_test, y))

        
    return(MSE,average_shortest_path_length, density)
    

def model2(data, model_type = "LR", feature = "normal", data_type = "normal", start = 0, end = 841,): 
    
    #N = data.shape[1]
    average_degree = []
    second_moment = []
    variance = []
    shannon_entropy = []
    transitivity_pr = []
    average_cluster= []
    average_shortest_path_length = []
    average_closeness = []
    average_betweennes = []
    average_eigenvector = []
    average_pagerank =  []
    r = np.arange(0.05,1,0.05)
    degree = []
 #   R2 = []
    MSE = []
    density = []
    S = []
    components = []
    if model_type == "NN":
        reg = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(50,25,10, ), random_state=1)
    elif model_type == "RF":
        reg = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
    else:
        reg = linear_model.LinearRegression() 
        
    if data_type == "PA":
        data_deseasonal, data_seasonal = deseasonal_monthly_anomaly(data)
        data_seasonal = data_seasonal[(end-1) % 12,:]
    elif data_type == "STL":
        data_deseasonal, data_seasonal = deseasonal_STL(data)
        data_seasonal = data_seasonal[end-1,:]
    else:
        data_deseasonal = data
        data_seasonal = np.zeros(data.shape[1]) 
        
    
    for i in range(len(r)):
        n = data.shape[1]
        m = nb.weighted_matrix(data_deseasonal[start:end-1,:])
        G, c = nb.graph_builder_limit(m, r[i])

        density.append(2 * G.number_of_edges()/(len(G)*(len(G)-1)))
        S.append(len(max(nx.connected_component_subgraphs(G), key=len)))
        components.append(c)
        
        degree = dict(G.degree())
        closseness = nx.closeness_centrality(G)
        kcore = dict(nx.core_number(G))
        betweeness= dict(nx.betweenness_centrality(G))
        pagerank = dict(nx.pagerank(G, alpha=0.85))
        #eigenvector = dict(nx.eigenvector_centrality(G, max_iter = 1000))
        eigenvector = dict(nx.eigenvector_centrality_numpy(G))
        
        average_degree.append(nb.momment_of_degree_distribution(G,1))                  #First Moment
        second_moment.append(nb.momment_of_degree_distribution(G,2))                   #Second Moment
        variance.append(nb.momment_of_degree_distribution(G,2) - nb.momment_of_degree_distribution(G,1)**2)     #Variance
        shannon_entropy.append(nb.shannon_entropy(G))                                  #Shanon Entropy
        transitivity_pr.append(nx.transitivity(G))                                     #Transitivity 
        average_cluster.append(nx.average_clustering(G))
        if nx.is_connected(G) == True:
            average_shortest_path_length.append(nx.average_shortest_path_length(G))     #Average Shortest Path
        else:
            average_shortest_path_length.append(0)
      
        average_closeness.append(np.mean(list(closseness.values())))                       #Average closeness centrality
        average_betweennes.append(np.mean(list(betweeness.values())))                        #Average betweenness centrality
        average_eigenvector.append(np.mean(list(eigenvector.values())))                      #Average eigenvector centrality
        average_pagerank.append(np.mean(list(pagerank.values())))
        
        if feature == "normal":
            X = feature_extractor(G) 
        else:
            X = feature_extractorNew(G)         
               
        Y = data_deseasonal[end-1,:]
        Y_test = data[end-1,:]
        reg.fit(X, Y)
        y = reg.predict(X)
        y = y + data_seasonal
        MSE.append(mean_squared_error(Y_test, y))

        
    return(MSE, average_eigenvector, average_shortest_path_length, density)
    
    
def feature_importance(data):
    importances_list = []
    r = np.arange(0.1,1,0.05)   
    regr = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
    for i in range(len(r)):
        M = data.shape[0]
        m = nb.weighted_matrix(data[:M-2,:])
        G, c = nb.graph_builder_limit(m, r[i])
        G = G.to_undirected()
        G.remove_edges_from(nx.selfloop_edges(G))

        
        X = feature_extractor(G)        
        Y = data[M-1,:]
        #Y_test = data[M-1,:]
        regr.fit(X, Y)
        #y = regr.predict(X)
        importances = regr.feature_importances_
        #std = np.std([tree.feature_importances_ for tree in regr.estimators_],
         #            axis=0)
        #indices = np.argsort(importances)[::-1]
        importances_list.append(importances)
    df_importance = pd.DataFrame(importances_list,columns=['degree','closseness','kcore','betweeness','pagerank', 'eigenvector','clustering'])
    return(df_importance)

def fisher_test(data):
    r = np.arange(0.1,1,0.05) 
    p_limit = []
    z_limit = []
    for i in range(len(r)):
        
        n = data.shape[1]
        m = nb.weighted_matrix(data)
        m = np.absolute(m)
        # G, c = nb.graph_builder_limit(m, r[i])
    
        np.fill_diagonal(m, 0)
        Z = abs(np.arctanh(m))
        N = data.shape[0]
        SE = 1/math.sqrt(N-3)
        Z_real = np.arctanh(0)
        Z = (Z-Z_real)/SE
        p = st.norm.cdf(Z)
    
        adjacency_matrix = np.zeros(m.shape)
        adjacency_matrix[m > r[i]] = 1
        
        if np.count_nonzero(np.where(m > r[i])) > 0:
            p_limit.append(p[adjacency_matrix.astype(int) == 1].min())
            z_limit.append(Z[adjacency_matrix.astype(int) == 1].min())
        else:
            p_limit.append(np.NAN)
            z_limit.append(np.NAN)
            
    return(p_limit, z_limit)
    
def refine_data(data):

    l = data.shape[1]
    result = []
    for i in range(l):
        if data[-1,i] != (-9.969209968386869e+36):
            result.append(data[:,i])
    return(np.transpose(np.array(result)))    

   

def main():
    f_pre = Dataset('precipitation.nc')
    pr = f_pre.variables['precip']
    pr = np.swapaxes(pr,0,2)
    
    r = np.arange(0.05,1,0.05)
     
    data = pr[16:23,10:16,:]
    data = unflatten(data)
    data = np.swapaxes(data,0,1)
    data = refine_data(data)
    MSE_ma,SP_ma, density_ma = model(data,model_type ="LR",feature = "normal",data_type = "MA", MA = 10)
# b = deseasonal_monthly_anomaly_MA(data[:2,:])

if __name__ == "__main__":
    main()