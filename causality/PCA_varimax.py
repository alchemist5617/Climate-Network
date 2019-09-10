from statsmodels.tsa.arima_process import ArmaProcess 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import eye, asarray, dot, sum, diag
from scipy.linalg import svd
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
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

#Data for world
f_pre = Dataset('../precip.nc')
data = f_pre.variables['precip']
lon = f_pre.variables['lon'][:]
lat = f_pre.variables['lat'][:]
time = f_pre.variables['time'][:]
data = np.swapaxes(data,0,2)

#Data for Africa
data = np.load('../data.npy')
lat = np.load('../lat.npy')
lon = np.load('../lon.npy')

result = []
index = []
lat_list = []
lon_list =[]
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if not data[i,j,-1] < 0:
            result.append(data[i,j,:])
            index.append((i,j))
            lon_list.append(lon[i])
            lat_list.append(lat[j])

result = np.matrix(result)
result = result.transpose()
data = pd.DataFrame(result)

scale = StandardScaler()
scaled_data = scale.fit_transform(data)

pca = PCA(n_components=7)
pca_model = pca.fit(scaled_data)

pca_data = pca_model.transform(data)

Matrix = pd.DataFrame(pca_model.components_)
Matrix1 = np.transpose(Matrix)

loading = pd.DataFrame(varimax(Matrix1))
clusters = loading.idxmax(axis=1)
df = pd.DataFrame({"lons":lon_list,"lats":lat_list,
                   "clusters":clusters.values.tolist()})

#colors = {0:'grey', 1:'purple', 2:'blue', 3:'green', 4:'oranges', 
#          5:'red',6:'lightred',7:'lightblue',
#          8:'lightblack',9:'lightgreen'}
#label_color = np.array([colors[l] for l in clusters.values])
#colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
#colors = np.hstack([colors] * 25)


#Data Visualisation    
colors = {0:'b', 1:'g', 2:'r', 3:'c', 4:'m', 5:'y',6:'k'}
label_color = np.array([colors[l] for l in clusters.values])
center_colors = label_color[:len(clusters)]

lon_temp = df["lons"].values
lon_temp[lon_temp > 180] = lon_temp[lon_temp > 180] -360
df["lons"].vlues = lon_temp


#lon = np.arange(-23.75,60.0,2.5)
fig = plt.figure(figsize=(30,15))
# setup Lambert Conformal basemap.
#m = Basemap(projection='cyl',
#              llcrnrlon=lon[0], 
#              llcrnrlat=lat[-1], 
#              urcrnrlon=lon[-1], 
#              urcrnrlat=lat[0],resolution='c')
m = Basemap(projection='merc',llcrnrlat=-80, urcrnrlat=80,             
            llcrnrlon=-180, urcrnrlon=180,lat_ts=20, resolution='c')
# draw coastlines.
m.drawcoastlines()
# draw a boundary around the map, fill the background.
# this background will end up being the ocean color, since
# the continents will be drawn on top.
m.drawmapboundary(fill_color='white')
#m.bluemarble()
# fill continents, set lake color same as ocean color.
#m.fillcontinents(color='lightgrey',lake_color='white', )
#m.drawlsmask(land_color='coral',ocean_color='aqua',lakes=True)

lons = df["lons"].values
lats = df["lats"].values
x,y = m(lons, lats)
plt.scatter(x, y, c=center_colors, s=50)
plt.show()

#Final PCA_Varimax data
loading = np.matrix(loading)
data = np.matrix(data)
result = np.matmul(data,loading)

