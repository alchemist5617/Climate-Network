from functions import *
import numpy as np
from netCDF4 import Dataset
import multiprocessing as mp 
import  ctypes
import math
np.random.seed(1360)


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



N = e.shape[0]
shared_array_base = mp.Array(ctypes.c_double, N*N)
shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
shared_array = shared_array.reshape(N, N)
lon_number = count_matrix.shape[1]



def adjacency_matrix1(start):
    
    for i in range(start, start + 36):        
        lat_index = math.floor(i/lon_number)
        lon_index = i % lon_number
        for j in range(N):
            if(e[i,j] == 0):
                continue;
            lat_index_sec = math.floor(j/lon_number)
            lon_index_sec = j % lon_number
            shared_array[i,j] = test(count_matrix[lat_index, lon_index], count_matrix[lat_index_sec, lon_index_sec],e[i,j], 1000, 3, 0.95)


window_idxs = [i for i in range(0, 2592, 36)]
pool = mp.Pool(16)
pool.map(adjacency_matrix1, window_idxs)

#r = adjacency_matrix(e, count_matrix, 2000, 3, 0.95)

np.save("AdjacencyMatrix.npy", shared_array)

