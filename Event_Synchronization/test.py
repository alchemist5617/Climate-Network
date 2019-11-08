from functions import *
import numpy as np
from netCDF4 import Dataset

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
r = adjacency_matrix(e, count_matrix, 1000, 3, 95)

np.save("AdjacencyMatrix2.npy", r)

