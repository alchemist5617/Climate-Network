#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  Jun 22 14:50:38 2019

@author: Mohammad Noorbakhsh
"""

import numpy as np
from netCDF4 import Dataset, num2date, date2num



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
                # Open netCDF3 or netCDF4 file
        f = Dataset(file_name)
        # Create reference to observable
        var = f.variables[variable_name]
        #Get time axis from NetCDF file plus dates
        time = f.variables['time'][:]
        units = f.variables['time'].units
        self._units = units
        #Get mask
        self._mask = (var[:,:,-1]!=-9.969209968386869e+36)
        #Get longtitude and latitude
        lon = f.variables['lon'][:]
        lat = f.variables['lat'][:]  
        #Set minimum and maximum of longtitude and latitude
        if spatial_limits is not None:
            lon_min = spatial_limits["lon_min"]
            lon_max = spatial_limits["lon_max"]
            if  lon_min in lon and lon_max in lon:
                 lon_min_Idx = np.where(lon == lon_min)[0]
                 lon_max_Idx = np.where(lon == lon_max)[0]
                 if lon_min_Idx < lon_max_Idx:
                     self._lon = lon[lon_min_Idx:lon_max_Idx+1]

                 else:
                     raise IndexError('lon_min is not less than lon_max')    
            else:
                raise IndexError('lon_min or lon_max is not in lon')
            
            lat_min = spatial_limits["lat_min"]
            lat_max = spatial_limits["lat_max"]
            if lat_min in lat and lat_max in lat:
                 lat_min_Idx = np.where(lat == lat_min)[0]
                 lat_max_Idx = np.where(lat == lat_max)[0] 
                 if lat_min_Idx < lat_max_Idx: 
                     self._lat = lat[lat_min_Idx:lat_max_Idx+1]
                 else:
                     raise IndexError('lat_min is not less than lat_max')
            else:
                raise IndexError('lat_min or lat_max is not in lat')
        else:
            lon_min_Idx = 0 
            lon_max_Idx = len(lon)-1
            self._lon = lon
            lat_min_Idx = 0 
            lat_max_Idx = len(lat)-1
            self._lat = lat
        #Set minimum and maximum of time
        if temporal_limits is not None:
            t_min = date2num(temporal_limits["time_min"], units)
            t_max = date2num(temporal_limits["time_max"], units)
            if t_min in time and t_max in time:
                t_min_Idx = np.where(time == t_min)[0]
                t_max_Idx = np.where(time == t_max)[0]
                if t_min_Idx < t_max_Idx:
                    self._time = time[t_min_Idx:t_max_Idx+1]
                else:
                    raise IndexError('time_min is not less than time_max')
            else:
                raise IndexError('time_min or time_max is not in Index')
        else:
            t_min_Idx = 0 
            t_max_Idx = len(time)-1
            self._time = time
            self._date = num2date(time, units)
        #create final data
        var = np.swapaxes(var,0,2)
        data = var[lon_min_Idx:lon_max_Idx+1,lat_min_Idx:lat_max_Idx+1,
                   t_min_Idx:t_max_Idx+1]
        data, index2d = self._unflatten_refine(data)           
        
        f.close()
        self._data = data
        self._index2d = index2d
        
    def get_data(self):
        """
        Returns data
                
        Returns
        -------
        data : 2D matrix of dates of shape (time,location)
    
        """
        return(self._data)        
        
    def get_variable_name(self):
        """
        Returns name of climate variable
                
        Returns
        -------
        variable_name : str
    
        """
        return(self._variable_name)
    
    def get_time(self):
        """
        Returns timestamp of data in from of number 
        of days from 01 01 1800
                
        Returns
        -------
        time : 1D array of floats
    
        """
        return(self._time)  
        
    def get_date(self):
        """
        Returns dates of data in from of number 
        of days since 1800-1-1 00:00:00
                
        Returns
        -------
        date : 1D array of floats
    
        """
        return(self._date)

    def get_units(self):
        """
        Returns time units
                
        Returns
        -------
        units: str
    
        """
        return(self._date)

    def get_mask(self):
        """
        Returns mask to specify which entry
        is NA
                
        Returns
        -------
        mask : 2D matrix of bool
    
        """
        return(self._amsk)
        
    def get_lon(self):
        """
        Returns longitude
                
        Returns
        -------
        longitude : 1D array of floats
    
        """
        return(self._lon)

    def get_lat(self):
        """
        Returns latitude
                
        Returns
        -------
        latitude : 1D array of floats
    
        """
        return(self._lat) 
   
    def get_indxe2d(self):
        """
        Returns 2d index of data
                
        Returns
        -------
        index2d : 2d matrix of integer
    
        """
        return(self._index2d)
    
    def _unflatten_refine(self,data):
        """
        Converts 3D spatio-temporal data inot 2D data
        
        Parameters
        ----------
        data : 3D matrix, shape (longtitude, latitude, time)
        
        Returns
        -------
        result : 2D matrix, shape (time, locaion)
            
        """
        result = []
        index2d = {}
        r = data.shape[0]
        c = data.shape[1]
        for i in range(r):
            for j in range(c):
                if data[i,j,-1] != (-9.969209968386869e+36):
                    result.append(data[i,j,:])
                    index2d[len(result)-1] = (i,j)   
        result = np.transpose(np.matrix(result))
        return(result, index2d)