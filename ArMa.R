r = length(lat)
c = length(lon)
AR = matrix( rep( 0, len=c*r), nrow = r)
MA = matrix( rep( 0, len=c*r), nrow = r)
d = matrix( rep( 0, len=c*r), nrow = r)

for (i in 1:r){
  for (j in 1:c){
    if (is.na(data[j,i,1])) {
      AR[i,j] <- NA
      d[i,j] <- NA
      MA[i,j] <- NA
    } else {
      t<-data[j,i,]
      t = ts(t, frequency=12)
      decomp = stl(t, s.window="periodic")
      deseasonal_cnt <- seasadj(decomp)
      result = auto.arima(deseasonal_cnt, seasonal=FALSE)
      e = arimaorder(result)
      AR[i,j] <- e[[1]]
      d[i,j] <- e[[2]]
      MA[i,j] <- e[[3]]
    }

  }
}

rotate <- function(x) t(apply(x, 2, rev))
MA<-rotate(MA)
AR<-rotate(AR)
d<-rotate(d)
library(lattice)

levelplot(AR)
levelplot(MA)
levelplot(d)
