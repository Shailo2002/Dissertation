#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:47:32 2020

@author: max
"""
import netCDF4
from netCDF4 import Dataset
import numpy as np
import math
import os,sys,csv,decimal,itertools

class MTModel:
    def __init__(self):
        self.Northing = np.array([])
        self.Easting = np.array([])
        self.Depth = np.array([])
        self.Conductivity = np.array([])
        self.bg_thickness = np.array([])
        self.bg_conductivity = np.array([])
    def ReadNetCDF(self,filename):
         data_fh = Dataset(filename, "r")
         self.Northing = np.array(data_fh.variables['Northing'][:])
         self.Easting = np.array(data_fh.variables['Easting'][:])
         self.Depth = np.array(data_fh.variables['Depth'][:])
         self.Conductivity = data_fh.variables['Conductivity'][:,:,:]
         north_orig = data_fh.variables['Northing_Origin'][:]
         east_orig = data_fh.variables['Easting_Origin'][:]
         depth_orig = data_fh.variables['Depth_Origin'][:]
         self.bg_conductivity = data_fh.variables['bg_conductivities'][:]
         self.bg_thickness = data_fh.variables['bg_thicknesses'][:]
         self.Northing = np.insert(self.Northing,0,north_orig)
         self.Easting = np.insert(self.Easting,0,east_orig)
         self.Depth = np.insert(self.Depth,0,depth_orig)
         data_fh.close()
    def WriteNetCDF(self,filename):
         data_fh = Dataset(filename, "w")
         nbound = 2
         bounddim = data_fh.createDimension("nbound", nbound)
         Northdim = data_fh.createDimension("Northing", self.Northing.size-1)
         Eastdim = data_fh.createDimension("Easting", self.Easting.size-1)
         Depthdim = data_fh.createDimension("Depth", self.Depth.size-1)
         bgdim = data_fh.createDimension("bg_layers", self.bg_conductivity.size)
         
         vnorth = data_fh.createVariable("Northing","f8",("Northing"))
         vnorth[:] = self.Northing[1:]
         vnorth.units = "m"

         vnorth_orig = data_fh.createVariable("Northing_Origin","f8")
         vnorth_orig[:] = self.Northing[0]
         
         veast = data_fh.createVariable("Easting","f8",("Easting"))
         veast[:] = self.Easting[1:]
         veast.units = "m"
         veast_orig = data_fh.createVariable("Easting_Origin","f8")
         veast_orig[:] = self.Easting[0]
         
         vdepth = data_fh.createVariable("Depth","f8",("Depth"))
         vdepth[:] = self.Depth[1:]
         vdepth.units = "m"
         vdepth_orig = data_fh.createVariable("Depth_Origin","f8")
         vdepth_orig[:] = self.Depth[0]
         
         vcond  = data_fh.createVariable("Conductivity","f8",("Depth","Easting","Northing"))
         vcond[:,:,:] = self.Conductivity[:,:,:]
         vcond.units = "S/m"
         vbgcond = data_fh.createVariable("bg_conductivities","f8",("bg_layers"))
         vbgcond[:] = self.bg_conductivity[:]
         vbgcond.units = "S/m"
         
         vbgthick = data_fh.createVariable("bg_thicknesses","f8",("bg_layers"))
         vbgthick[:] = self.bg_thickness[:]
         vbgthick.units = "m"

         
         data_fh.close()    
    def read_csv(self,filename,delim):

        #Simple function for reading csv files and give out filtered output for given delimiter (delim)

        file_obj = open(filename,'rt',encoding = "utf8") #Creating file object
        file_csv = csv.reader(file_obj,delimiter = delim) #Reading the file object with csv module, delimiter assigned to ','
        data = [] #Creating empty array to append data

        #Appending data from csb object
        for row in file_csv:
            data.append(row)

        #Filtering data for None elements read.
        for j in range(0,len(data)):
            data[j] = list(filter(None,data[j]))
        data = list(filter(None,data))

        return data

    def ReadModEM(self,filename):

        self.ModEM_rho_data = self.read_csv(filename, delim = ' ')

        self.x_num = int(self.ModEM_rho_data[1][0])
        self.y_num = int(self.ModEM_rho_data[1][1])
        self.z_num = int(self.ModEM_rho_data[1][2])

        self.x_grid = np.asarray(self.ModEM_rho_data[2]).astype(np.float)
        self.y_grid = np.asarray(self.ModEM_rho_data[3]).astype(np.float)
        self.z_grid = np.asarray(self.ModEM_rho_data[4]).astype(np.float)

        self.lenxgrid = len(self.x_grid)
        self.lenygrid = len(self.y_grid)
        self.lenzgrid = len(self.z_grid)

        self.rho = []

        for k in range(5,len(self.ModEM_rho_data) - 2 ,self.y_num):
            rhoy = []
            for z in range(k, k + self.y_num):
                rhox = []
                for l in range(0,self.x_num):
                    rhox.append(float(self.ModEM_rho_data[z][l]))
                rhoy.append(rhox)

            self.rho.append(rhoy)

        self.rho = np.exp(np.asarray(self.rho))


        self.z_depth = np.array([0.0])
        self.z_grid = np.cumsum(self.z_grid)
        self.z_depth = np.append(self.z_depth,self.z_grid)


        self.mid_point_x = int(len(self.x_grid) / 2.0)
        self.mid_point_y = int(len(self.y_grid) / 2.0)

        if len(self.x_grid) %2 == 0:
            self.mid_point_x = int(len(self.x_grid) / 2.0)
            beg_x = np.sum(self.x_grid[:self.mid_point_x]) * -1

        elif len(self.x_grid) %2 != 0:
            self.mid_point_x = int(len(self.x_grid) / 2.0) + 1
            beg_x = np.sum(self.x_grid[:self.mid_point_x]) * -1 + (self.x_grid[self.mid_point_x] / 2.0)

        if len(self.y_grid) %2 == 0:
            self.mid_point_y = int(len(self.y_grid) / 2.0)
            beg_y = np.sum(self.y_grid[:self.mid_point_y]) * -1
        elif len(self.y_grid) %2 != 0:
            self.mid_point_y = int(len(self.y_grid) / 2.0) + 1
            beg_y = np.sum(self.y_grid[:self.mid_point_y]) * -1 + (self.y_grid[self.mid_point_y] / 2.0)



        self.Northing = np.append([beg_x],beg_x + np.cumsum(self.x_grid))
        #self.Northing = self.Northing[::-1]
        self.Easting = np.append([beg_y],beg_y + np.cumsum(self.y_grid))
        self.Depth = self.z_depth
        self.Conductivity = 1.0/self.rho[:,:,::-1]