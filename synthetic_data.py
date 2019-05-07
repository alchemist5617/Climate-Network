import numpy as np
import math
import random

class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
def flatten(i, j, n):
    return(i * n +j)

def unflatten(k, n):
    j = k % n
    i = int((k-j)/n)
    return(i, j)

def toroidal_distance(p, q, L):
    delta_x = abs(p.x-q.x)
    if(delta_x > 0.5 * L):
        delta_x = L - delta_x
    delta_y = abs(p.y - q.y)
    if(delta_y > 0.5 * L):
        delta_y = L - delta_y
    return(math.sqrt(delta_x**2 + delta_y**2))
    
def synthetic_data_generator(n = 10, M = 1000, a = 0.5, b = 0.5, epsilon = 4.0E-2, denominator = 2.0):
    
    delta_x = 2.0 * math.pi/n
    lamda = 3 * delta_x 
    
    
    x = np.empty((n,n), dtype=object)
    X = np.empty(n*n, dtype=object)
    for i in range(n):
        for j in range(n):
            x[i,j] = point(x = i * delta_x, y =  j * delta_x)
            X[flatten(i,j,n)] = x[i,j]
        
    N = n*n
    phi = np.empty(N, dtype=float)
    c = np.empty(N, dtype=float)
    sigma = np.empty(N, dtype=float)
    alpha = np.empty(N, dtype=float)
    
    
    for i in range(N):
        xc = X[i].x
        yc = X[i].y
        phi[i] = (0.5 + a * math.sin(xc) * math.sin(yc))
        c[i] = 0.0
        sigma[i] = math.sqrt(1.0 - phi[i]**2)
        alpha[i] = 1.0 + b * math.sin(xc/denominator) * math.sin(yc/denominator)
    
    C = np.empty((N,N), dtype=float)
    
    for i in range(N):
        for j in range(N):
            d = toroidal_distance(X[i], X[j], 2*math.pi)
            C[i,j] = math.exp(-d/lamda)
            if(i==j):
                C[i,j] = 1 + epsilon
    
    lamda_C = np.linalg.cholesky(C)
    
    A = np.empty((M,N), dtype=float)
    data = np.empty((M,N), dtype=float)
    
    random.seed(1234)
    
    A[0,:] = np.random.normal(0, 1, N)
    data[0,:] = np.matmul(lamda_C,A[0,:])
    for i in range(1,M):
        
        zeta = sigma * np.random.normal(0, 1, N)
        A[i,] = c + (phi * A[i-1,:]) + zeta
        data[i,] = np.matmul(lamda_C,A[i,])
        
    return(data)    
    