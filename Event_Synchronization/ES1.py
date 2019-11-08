import math
import numpy as np
from netCDF4 import Dataset


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

f = Dataset('spi3_6_12_1deg_cru_ts_3_21_1949_2012.nc') 
spi = f.variables['spi3']

#data = spi[:,49:131,149:241]

data = np.zeros((768,90,180))
for i in range(0,180,2):
    for j in range(0,360,2):
        data[:,int(i/2),int(j/2)] = spi[:,i,j]
        
data[data > -1] = 0
data[data<=-1] = 1

m = ES_matrix(data, 3)
np.save('ESMatrixWhole.npy', m)
