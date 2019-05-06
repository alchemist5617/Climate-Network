library("ncdf4")
library('forecast')
library('tseries')

ncin <- nc_open("precipitation.nc")
lon <- ncvar_get(ncin,"lon")
lat <- ncvar_get(ncin,"lat")
time <- ncvar_get(ncin,"time")
data<-ncvar_get(ncin,"precip")

t<-data[19,15,] #Aw AR(1,1,0)          21.25 3.75
t<-data[23,4,] #BWh MA(1)          31.25  31.25
t<-data[22,27,] #Cwb AR(2)         28.75 -26.25
t<-data[142,134,] #Aw AR(1)        45.25 -25.75
t<-data[107,132,] #BSh ARMA(1,0,2) 27.75 -24.75
t<-data[113,60,] #BSh ARMA(3,0,1)  30.75 11.25
t<-data[91,81,] #Af ARMA(1,0,1)  19.75 0.75
t<-data[73,39,] #BWh ARMA(0,0,0)


plot.ts(t)

t = ts(t, frequency=12)
decomp = stl(t, s.window="periodic")
deseasonal_cnt <- seasadj(decomp)
plot(decomp)

adf.test(t, alternative = "stationary")

Acf(t)
Pacf(t)

count_d1 = diff(deseasonal_cnt, differences = 1)
plot(count_d1)
adf.test(count_d1, alternative = "stationary")

auto.arima(deseasonal_cnt, seasonal=FALSE)