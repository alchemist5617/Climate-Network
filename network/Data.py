#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Jun 22 14:50:38 2019

@author: Mohammad Noorbakhsh
"""

import numpy as np
from netCDF4 import Dataset


class Data:

    """
    Encapsulates general spatio-temporal data.

    """

    def __init__(self,file_name, variable_name, temporal_limits=None, 
                 spatial_limits=None):
        #Set file name
        self._file_name = file_name
        #Set variable name
        self._variable_name = variable_name
        
        def get_variable_name(self):
            return(self._variable_name)
            
        def set_variable_name(self, variable_name):
            self._variable_name = variable_name
            
    @staticmethod
    def Load(self, file_name, variable_name, temporal_limits=None,
             spatial_limits=None):
        """
        Import data from a NetCDF file with a regular and rectangular grid.
        
        Parameters
        ----------
        file_name: str, The name of the data file.
        variable_name:str, The name of the variable.
        temporal_limits:dics, Maximum and minimum values of time
        spatia_limits:dics, Maximum and minimum values of lon and lat
        
        Returns
        -------
        result : 2D matrix, shape (time, index)

        """
        
        data = self._load_data(self, file_name, variable_name,
                             temporal_limits, spatial_limits)

        return data
    
    
    def _load_data(self, file_name, variable_name,temporal_limits=None, 
                   spatial_limits=None):
        """
        Import data from a NetCDF file with a regular and rectangular grid.
        
        Parameters
        ----------
        file_name: str, The name of the data file.
        variable_name:str, The name of the variable.
        temporal_limits:dics, Maximum and minimum values of time
        spatia_limits:dics, Maximum and minimum values of lon and lat
        
        Returns
        -------
        result : 2D matrix, shape (time, index)

        """

        # Open netCDF3 or netCDF4 file
        f = Dataset(file_name)
        # Create reference to observable
        var = f.variables[variable_name]
        # Get time axis from NetCDF file
        time = f.variables['time'][:]
        #Get longtitude and latitude
        lon = f.variables['lon'][:]
        lat = f.variables['lat'][:]
        #Set minimum and maximum of longtitude and latitude
        if spatial_limits is not None:
            if spatial_limits["lon_min"] in lon and spatial_limits["lon_max"] in lon:
                 lon_min = spatial_limits["lon_min"]  
                 lon_max = spatial_limits["lon_max"] 
            else:
                raise IndexError('lon_min or lon_max is not in lon')
            
            if spatial_limits["lat_min"] in lat and spatial_limits["lat_max"] in lat:
                 lat_min = spatial_limits["lat_min"]  
                 lat_max = spatial_limits["lat_max"] 
            else:
                raise IndexError('lat_min or lat_max is not in lat')
        else:
            lon_min = 0 
            lon_max = len(lon)
            lat_min = 0 
            lat_max = len(lat)
        #Set minimum and maximum of time
        if temporal_limits is not None:
            if temporal_limits["time_min"] in time and temporal_limits["time_max"] in time:
                 time_min = temporal_limits["time_min"]  
                 time_max = temporal_limits["time_max"] 
            else:
                raise IndexError('time_min or time_max is not in time')
        else:
            time_min = 0 
            time_max = len(time)            
        #create final data
        var = np.swapaxes(var,0,2)
        data = var[lon_min:lon_max,lat_min:lat_max,time_min:time_max]
        data = self._unflatten(data)
        data = np.swapaxes(data,0,1)
        data = self._refine_data(data)
        
        f.close()
        return data
    
    def _unflatten(data):
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
        
    def _refine_data(data):
        """
        Removes unspecified items from data
        
        Parameters
        ----------
        data : 2D matrix, shape (time, index)
        
        Returns
        -------
        result : 2D matrix, shape (time, index)
            
        """  
        l = data.shape[1]
        result = []
        for i in range(l):
            if data[-1,i] != (-9.969209968386869e+36):
                result.append(data[:,i])
        return(np.transpose(np.array(result)))