import numpy as np
import math
#from scipy import stats
import networkx as nx
import scipy.stats as st
from fastdtw import fastdtw


    
def winter_only(data,first_dec):
    n = data.shape[0]
    N = data.shape[1]
    start = first_dec+2
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
    
def weighted_matrix(data, correlation_type = "pearson", WINTER_ONLY = False, first_dec = 0):
    if WINTER_ONLY:
        input_data= winter_only(data,first_dec)
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
    
def weighted_matrix_dtw(data):
    N = data.shape[1]
    dtw = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
             dtw[i,j] = fastdtw(data[:,i],data[:,j])[0]
    return(dtw)
        
def graph_builder_limit (weighted_matrix,limit):
    #weighted_matrix = np.exp(-np.sqrt(1 - weighted_matrix))
    weighted_matrix = np.nan_to_num(weighted_matrix)
    weighted_matrix = np.absolute(weighted_matrix)
    np.fill_diagonal(weighted_matrix, 0)
    componenets_number = 0
    adjacency_matrix = np.zeros(weighted_matrix.shape)
    adjacency_matrix[weighted_matrix >= limit] = 1
    G = nx.from_numpy_matrix(adjacency_matrix)
    G = G.to_undirected()
    G.remove_edges_from(G.selfloop_edges())
    #Gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
    #componenets_number = len(Gcc) 
    componenets_number = nx.number_connected_components(G)
    return(G, componenets_number)

def graph_builder (weighted_matrix):
    weighted_matrix = np.exp(-np.sqrt(1 - weighted_matrix))
    componenets_number = 0
    limit = 1.0
    while componenets_number != 1:
        limit -= 0.01
        adjacency_matrix = np.zeros(weighted_matrix.shape)
        adjacency_matrix[weighted_matrix >= limit] = 1
        G = nx.from_numpy_matrix(adjacency_matrix)
        G = G.to_undirected()
        G.remove_edges_from(G.selfloop_edges())
        #Gcc = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)
        #componenets_number = len(Gcc)
        componenets_number = nx.number_connected_components(G)
    return(G, limit)

def degree_distribution(G):
    vk = dict(G.degree())
    vk = list(vk.values()) # we get only the degree values
    maxk = np.max(vk)
    mink = np.min(min)
    kvalues= np.arange(0,maxk+1) # possible values of k
    Pk = np.zeros(maxk+1) # P(k)
    for k in vk:
        Pk[k] = Pk[k] + 1
    Pk = Pk/sum(Pk) # the sum of the elements of P(k) must to be equal to one
    return kvalues,Pk

def momment_of_degree_distribution(G,m):
    k,Pk = degree_distribution(G)
    M = sum((k**m)*Pk)
    return M

def shannon_entropy(G):
    k,Pk = degree_distribution(G)
    H = 0
    for p in Pk:
        if(p > 0):
            H = H - p*math.log(p, 2)
    return H

def degree_builder(dictionary):
    deg = np.empty((10,10), dtype=float)
    lat_number = 10
    lon_number = 10
    for i in range(lat_number):
        for j in range(lon_number):
            index = i*lon_number+j
            if index in dictionary.keys():
                deg[i,j] = dictionary[index]
            else:
                deg[i,j] = np.nan
    return deg

def graph_builder_fisher (weighted_matrix, limit):
    np.fill_diagonal(weighted_matrix, 0)
    Z = abs(np.arctanh(weighted_matrix))
    N = 1000
    SE = 1/math.sqrt(N-3)
    Z = Z/SE
    p = st.norm.cdf(Z)
    
    adjacency_matrix = np.zeros(weighted_matrix.shape)
    adjacency_matrix[p > (1-limit/2)] = 1
    G = nx.from_numpy_matrix(adjacency_matrix)
    G = G.to_undirected()
    G.remove_edges_from(G.selfloop_edges())
    componenets_number = nx.number_connected_components(G)    
    return(G, componenets_number)

