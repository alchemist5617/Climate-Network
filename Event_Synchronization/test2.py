import numpy as np
from netCDF4 import Dataset
from numba import jit, prange
import math

np.random.seed(1360)

@jit(parallel=True)
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
    
@jit(parallel=True)
def ES(t1, t2, t_max):
    t1_index = np.where((t1 == 0) & (np.roll(t1,-1) == 1))[0] + 1
    t2_index = np.where((t2 == 0) & (np.roll(t2,-1) == 1))[0] + 1
    s = 0
    if((len(t1_index)!=0) and (len(t2_index)!=0)): 
        for i in prange(len(t1_index)):
            for j in prange(len(t2_index)):
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

@jit(parallel=True)
def ES_matrix(data, t_max):
    lat_number = data.shape[1]
    lon_number = data.shape[2]
    PR = np.zeros((lat_number*lon_number,lat_number*lon_number))
    
    for i in prange(lat_number*lon_number):
        lat_index = math.floor(i/lon_number)
        lon_index = i % lon_number
        if np.isnan(data[-1,lat_index ,lon_index]): 
            PR[i,:] = np.nan
            continue
        for j in prange(lat_number*lon_number):
            lat_index_sec = math.floor(j/lon_number)
            lon_index_sec = j % lon_number
            if np.isnan(data[-1,lat_index ,lon_index]): 
                PR[i,j] = np.nan
                continue
            else:
                PR[i,j] = ES(data[:,lat_index,lon_index], data[:,lat_index_sec,lon_index_sec], t_max)
    return(PR)

@jit(parallel=True)
def unif_generator(l):
    n = 0
    lst = np.empty(0)
    lst = np.append(lst,np.random.randint(2))
    if (lst[-1] == 1): 
        n = 1
    while (l > n):
        r = np.random.randint(2)
        if(r == 1):
            if(lst[-1]==0):
                lst = np.append(lst,r)
                n = n + 1
        else:
            lst = np.append(lst,r)
    return(lst)
    
@jit(parallel=True)
def event_count(t):
    t_index = np.where((t == 0) & (np.roll(t,-1) == 1))[0] + 1
    return(len(t_index))
    
@jit(parallel=True)
def event_count_matrix(data):
    lat_number = data.shape[1]
    lon_number = data.shape[2]
    result = np.zeros((lat_number,lon_number))
    
    for i in prange(lat_number):
        for j in prange(lon_number):
            result[i,j] = event_count(data[:,i,j])
    return(result)

@jit(parallel=True)
def test(t1_count, t2_count, weight, n, t_max, significance):
    result = np.empty(0)
    for i in prange(n):
        sample1 = unif_generator(t1_count)
        sample2 = unif_generator(t2_count)
        sample_weight = ES(sample1, sample2, t_max)
        result = np.append(result, sample_weight)
    percentile = np.percentile(result, significance) 
    if(weight>percentile):
        return(1)
    else:
        return(0)

@jit(parallel=True)        
def adjacency_matrix(weighted_matrix, count_matrix, n ,t_max, significance):
    N = weighted_matrix.shape[0]
    lat_number = count_matrix.shape[0]
    lon_number = count_matrix.shape[1]
    result = np.zeros((N,N))
    
    for i in prange(N):        
        lat_index = math.floor(i/lon_number)
        lon_index = i % lon_number
        for j in prange(N):
            if(weighted_matrix[i,j] == 0):
                continue;
            lat_index_sec = math.floor(j/lon_number)
            lon_index_sec = j % lon_number
            result[i,j] = test(count_matrix[lat_index, lon_index], count_matrix[lat_index_sec, lon_index_sec],weighted_matrix[i,j], n, t_max, significance)
    return(result)

f = Dataset('spi3_6_12_1deg_cru_ts_3_21_1949_2012.nc') 
spi = f.variables['spi3']

e = np.load('ESMatrixWhole1.npy')
#data = spi[:,49:131,149:241]

data = np.zeros((768,36,72))
for i in range(0,180,5):
    for j in range(0,360,5):
        data[:,int(i/5),int(j/5)] = spi[:,i,j]
        
data[data > -1] = 0
data[data<=-1] = 1
count_matrix = event_count_matrix(data)
r = adjacency_matrix(e, count_matrix, 1000, 3, 0.95)

np.save("AdjacencyMatrix.npy", r)

