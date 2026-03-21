#!/usr/bin/python
import netCDF4
from netCDF4 import Dataset
import numpy as np
import math
import mtpy.modeling.modem as modem

def ZfromRho(freq, rho):
    Z = rho * np.sqrt((8e-7 *math.pi**2 *freq)/2.0)
    return Z
def CalcRho(freq, ZReal, ZIm ):
   rho = np.ones(ZReal.shape)
   for i in range(0,len(ZReal[:,0])):
      rho[i,:] = (ZReal[i,:]**2 + ZIm[i,:]**2)/(8e-7 *math.pi**2 *freq[i])
   return rho;

def CalcdRho(freq, ZReal, ZIm, dZ):
    drho = np.ones(ZReal.shape)
    for i in range(0,len(ZReal[:,0])):
      drho[i,:] = np.sqrt(ZReal[i,:]**2 + ZIm[i,:]**2) * dZ[i,:] /(4e-7 *math.pi**2 *freq[i])
    return drho

def CalcPhi(ZReal, ZIm):
   phi = np.arctan(ZIm/ZReal) / math.pi * 180.0
   return phi

def CalcPhi2(ZReal, ZIm):
   phi = np.arctan2(ZIm,ZReal) / math.pi * 180.0
   return phi

def CalcdPhi(ZReal, ZIm, dZ):
   dphi = np.divide(dZ,np.sqrt(ZReal**2 + ZIm**2)) / math.pi * 180.0
   return dphi

class MTData:
     def __init__(self):
         self.freq = np.array([])
         self.MeasX = np.array([])
         self.MeasY = np.array([])
         self.MeasZ = np.array([])
         self.Zxx_re = np.array([])
         self.Zxx_im = np.array([])
         self.Zxy_re = np.array([])
         self.Zxy_im = np.array([])
         self.Zyx_re = np.array([])
         self.Zyx_im = np.array([])
         self.Zyy_im = np.array([])
         self.Zyy_re = np.array([])
         self.dZxx = np.array([])
         self.dZxy = np.array([])
         self.dZyx = np.array([])
         self.dZyy = np.array([])
         self.C  = np.array([])
         self.Names = ""
     def ReadNetCDF(self,filename):
         data_fh = Dataset(filename, "r")
         self.freq = np.array(data_fh.variables['Frequency'][:])
         self.MeasX = np.array(data_fh.variables['MeasPosX'][:])
         self.MeasY = np.array(data_fh.variables['MeasPosY'][:])
         self.MeasZ = np.array(data_fh.variables['MeasPosZ'][:])
         self.Zxx_re = np.array(data_fh.variables['Zxx_re'][:,:])
         self.Zxx_im = np.array(data_fh.variables['Zxx_im'][:,:])
         self.Zxy_re = np.array(data_fh.variables['Zxy_re'][:,:])
         self.Zxy_im = np.array(data_fh.variables['Zxy_im'][:,:])
         self.Zyx_re = np.array(data_fh.variables['Zyx_re'][:,:])
         self.Zyx_im = np.array(data_fh.variables['Zyx_im'][:,:])
         self.Zyy_im = np.array(data_fh.variables['Zyy_im'][:,:])
         self.Zyy_re = np.array(data_fh.variables['Zyy_re'][:,:])
         self.dZxx = np.array(data_fh.variables['dZxx'][:,:])
         self.dZxy = np.array(data_fh.variables['dZxy'][:,:])
         self.dZyx = np.array(data_fh.variables['dZyx'][:,:])
         self.dZyy = np.array(data_fh.variables['dZyy'][:,:])
         if 'C' in data_fh.variables:
            self.C = np.array(data_fh.variables['C'][:,:])        
         if 'Names' in data_fh.variables:   
           self.Names = data_fh.variables['Names'][:]
         else:
           self.Names = np.array(["Station {}".format(x) for x in range(len(self.MeasX))])
         if 'RotationAngle' in data_fh.variables:
             self.RotationAngle = np.array(data_fh.variables['RotationAngle'][:]) 
         data_fh.close()
     def ReadModEM(self,filename,center_utm=None):
         convfactor = 4.0 * 1e-4 * math.pi;
         data = modem.Data()         
         data.read_data_file(filename,center_utm)
         self.freq = 1.0/data.period_list
         self.MeasX = data.data_array["north"]
         self.MeasY = data.data_array["east"]
         self.MeasZ = data.data_array["elev"]
         self.Zxx_re = convfactor * np.real(data.data_array["z"][:,:,0,0].T)
         self.Zxx_im = convfactor * np.imag(data.data_array["z"][:,:,0,0].T)
         self.Zxy_re = convfactor * np.real(data.data_array["z"][:,:,0,1].T)
         self.Zxy_im = convfactor * np.imag(data.data_array["z"][:,:,0,1].T)
         self.Zyx_re = convfactor * np.real(data.data_array["z"][:,:,1,0].T)
         self.Zyx_im = convfactor * np.imag(data.data_array["z"][:,:,1,0].T)
         self.Zyy_re = convfactor * np.real(data.data_array["z"][:,:,1,1].T)
         self.Zyy_im = convfactor * np.imag(data.data_array["z"][:,:,1,1].T)
         self.dZxx = convfactor * data.data_array["z_err"][:,:,0,0].T
         self.dZxy = convfactor * data.data_array["z_err"][:,:,0,1].T
         self.dZyx = convfactor * data.data_array["z_err"][:,:,1,0].T
         self.dZyy = convfactor * data.data_array["z_err"][:,:,1,1].T
         self.Names = data.data_array["station"]
         self.RotationAngle = np.zeros(self.MeasX.shape)
         self.C = np.zeros((self.MeasX.size,4))
         self.C[:,0] = 1.0
         self.C[:,3] = 1.0
     def WriteNetCDF(self,filename):
         data_fh = Dataset(filename, "w")
         nstat = self.MeasX.size
         nfreq = self.freq.size
         statdim = data_fh.createDimension("StationNumber", nstat)
         freqdim = data_fh.createDimension("Frequency", nfreq)
         cdim = data_fh.createDimension("Celem", 4)
         vnames = data_fh.createVariable("Names", str, ("StationNumber"))
         vnames[:] = self.Names
         vfreq = data_fh.createVariable("Frequency","f8",("Frequency"))
         vfreq[:] = self.freq
         vmeasx = data_fh.createVariable("MeasPosX","f8",("StationNumber"))
         vmeasx.units = "m"
         vmeasx[:] = self.MeasX[:]
         vmeasy = data_fh.createVariable("MeasPosY","f8",("StationNumber"))
         vmeasy.units = "m"
         vmeasy[:] = self.MeasY[:]
         vmeasz = data_fh.createVariable("MeasPosZ","f8",("StationNumber"))
         vmeasz.units = "m"
         vmeasz[:] = self.MeasZ[:]
         vzxx_re = data_fh.createVariable("Zxx_re","f8",("Frequency","StationNumber"))
         vzxx_re.units = "Ohm"
         vzxx_re[:,:] = self.Zxx_re[:,:]
         vzxx_im = data_fh.createVariable("Zxx_im","f8",("Frequency","StationNumber"))
         vzxx_im.units = "Ohm"
         vzxx_im[:,:] = self.Zxx_im[:,:]
         vzxy_re = data_fh.createVariable("Zxy_re","f8",("Frequency","StationNumber"))
         vzxy_re.units = "Ohm"
         vzxy_re[:,:] = self.Zxy_re[:,:]
         vzxy_im = data_fh.createVariable("Zxy_im","f8",("Frequency","StationNumber"))
         vzxy_im.units = "Ohm"
         vzxy_im[:,:] = self.Zxy_im[:,:]
         vzyx_re = data_fh.createVariable("Zyx_re","f8",("Frequency","StationNumber"))
         vzyx_re.units = "Ohm"
         vzyx_re[:,:] = self.Zyx_re[:,:]
         vzyx_im = data_fh.createVariable("Zyx_im","f8",("Frequency","StationNumber"))
         vzyx_im.units = "Ohm"
         vzyx_im[:,:] = self.Zyx_im[:,:]
         vzyy_re = data_fh.createVariable("Zyy_re","f8",("Frequency","StationNumber"))
         vzyy_re.units = "Ohm"
         vzyy_re[:,:] = self.Zyy_re[:,:]
         vzyy_im = data_fh.createVariable("Zyy_im","f8",("Frequency","StationNumber"))
         vzyy_im.units = "Ohm"
         vzyy_im[:,:] = self.Zyy_im[:,:]
         vdzxx = data_fh.createVariable("dZxx","f8",("Frequency","StationNumber"))
         vdzxx.units = "Ohm"
         vdzxx[:,:] = self.dZxx[:,:]
         vdzxy = data_fh.createVariable("dZxy","f8",("Frequency","StationNumber"))
         vdzxy.units = "Ohm"
         vdzxy[:,:] = self.dZxy[:,:]
         vdzyx = data_fh.createVariable("dZyx","f8",("Frequency","StationNumber"))
         vdzyx.units = "Ohm"
         vdzyx[:,:] = self.dZyx[:,:]
         vdzyy = data_fh.createVariable("dZyy","f8",("Frequency","StationNumber"))
         vdzyy.units = "Ohm"
         vdzyy[:,:] = self.dZyy[:,:]
         if self.C.size > 0:
             vc = data_fh.createVariable("C","f8",("StationNumber","Celem"))
             vc[:,:] = self.C[:,:]
    
         if self.RotationAngle.size > 0:
             vc = data_fh.createVariable("RotationAngle","f8",("StationNumber"))
             vc[:] = self.RotationAngle[:]
         data_fh.close()
     def RemoveStations(self,indices):
         self.MeasX = np.delete(self.MeasX,indices)
         self.MeasY = np.delete(self.MeasY,indices)
         self.MeasZ = np.delete(self.MeasZ,indices)
         self.Zxx_re = np.delete(self.Zxx_re,indices,1)
         self.Zxx_im = np.delete(self.Zxx_im,indices,1)
         self.Zxy_re = np.delete(self.Zxy_re,indices,1)
         self.Zxy_im = np.delete(self.Zxy_im,indices,1)
         self.Zyx_re = np.delete(self.Zyx_re,indices,1)
         self.Zyx_im = np.delete(self.Zyx_im,indices,1)
         self.Zyy_re = np.delete(self.Zyy_re,indices,1)
         self.Zyy_im = np.delete(self.Zyy_im,indices,1)
         self.dZxx = np.delete(self.dZxx,indices,1)
         self.dZxy = np.delete(self.dZxy,indices,1)
         self.dZyx = np.delete(self.dZyx,indices,1)
         self.dZyy = np.delete(self.dZyy,indices,1)
         self.C = np.delete(self.C,indices,0)
         self.Names = np.delete(self.Names,indices)        
         self.RotationAngle = np.delete(self.RotationAngle,indices)        
     def AddStations(self,otherdata):
        self.MeasX = np.concatenate((self.MeasX,otherdata.MeasX))
        self.MeasY = np.concatenate((self.MeasY,otherdata.MeasY))
        self.MeasZ = np.concatenate((self.MeasZ,otherdata.MeasZ))
        self.Zxx_re = np.concatenate((self.Zxx_re,otherdata.Zxx_re),axis=1)
        self.Zxx_im = np.concatenate((self.Zxx_im,otherdata.Zxx_im),axis=1)
        self.Zxy_re = np.concatenate((self.Zxy_re,otherdata.Zxy_re),axis=1)
        self.Zxy_im = np.concatenate((self.Zxy_im,otherdata.Zxy_im),axis=1)
        self.Zyx_re = np.concatenate((self.Zyx_re,otherdata.Zyx_re),axis=1)
        self.Zyx_im = np.concatenate((self.Zyx_im,otherdata.Zyx_im),axis=1)
        self.Zyy_re = np.concatenate((self.Zyy_re,otherdata.Zyy_re),axis=1)
        self.Zyy_im = np.concatenate((self.Zyy_im,otherdata.Zyy_im),axis=1)
        self.dZxx = np.concatenate((self.dZxx,otherdata.dZxx),axis=1)
        self.dZxy = np.concatenate((self.dZxy,otherdata.dZxy),axis=1)
        self.dZyx = np.concatenate((self.dZyx,otherdata.dZyx),axis=1)
        self.dZyy = np.concatenate((self.dZyy,otherdata.dZyy),axis=1)
        self.Names = np.concatenate((self.Names,otherdata.Names))
        self.RotationAngle = np.concatenate((self.RotationAngle,otherdata.RotationAngle))

        self.C = np.concatenate((self.C,otherdata.C),axis=0)


        
     def RhoXX(self):
         return CalcRho(self.freq, self.Zxx_re, self.Zxx_im)
     def RhoXY(self):
         return CalcRho(self.freq, self.Zxy_re, self.Zxy_im)
     def RhoYX(self):
         return CalcRho(self.freq, self.Zyx_re, self.Zyx_im)
     def RhoYY(self):
         return CalcRho(self.freq, self.Zyy_re, self.Zyy_im)

     def dRhoXX(self):
         return CalcdRho(self.freq, self.Zxx_re, self.Zxx_im, self.dZxx)
     def dRhoXY(self):
         return CalcdRho(self.freq, self.Zxy_re, self.Zxy_im, self.dZxy)
     def dRhoYX(self):
         return CalcdRho(self.freq, self.Zyx_re, self.Zyx_im, self.dZyx)
     def dRhoYY(self):
         return CalcdRho(self.freq, self.Zyy_re, self.Zyy_im, self.dZyy)


     def PhiXX(self):
         return CalcPhi2(self.Zxx_re,self.Zxx_im)
     def PhiXY(self):
         return CalcPhi2(self.Zxy_re,self.Zxy_im)
     def PhiYX(self):
         return CalcPhi2(self.Zyx_re,self.Zyx_im)
     def PhiYY(self):
         return CalcPhi2(self.Zyy_re,self.Zyy_im)

     def Phi90XX(self):
         return CalcPhi(self.Zxx_re,self.Zxx_im)
     def Phi90XY(self):
         return CalcPhi(self.Zxy_re,self.Zxy_im)
     def Phi90YX(self):
         return CalcPhi(self.Zyx_re,self.Zyx_im)
     def Phi90YY(self):
         return CalcPhi(self.Zyy_re,self.Zyy_im)


     def dPhiXX(self):
         return CalcdPhi(self.Zxx_re,self.Zxx_im, self.dZxx)
     def dPhiXY(self):
         return CalcdPhi(self.Zxy_re,self.Zxy_im, self.dZxy)
     def dPhiYX(self):
         return CalcdPhi(self.Zyx_re,self.Zyx_im, self.dZyx)
     def dPhiYY(self):
         return CalcdPhi(self.Zyy_re,self.Zyy_im, self.dZyy)


class TipperData:
     def __init__(self):
         self.freq = np.array([])
         self.MeasX = np.array([])
         self.MeasY = np.array([])
         self.MeasZ = np.array([])
         self.Tx_re = np.array([])
         self.Tx_im = np.array([])
         self.Ty_re = np.array([])
         self.Ty_im = np.array([])
         self.dTx = np.array([])
         self.dTy = np.array([])         
         self.Names = ""
     def ReadNetCDF(self,filename):
         data_fh = Dataset(filename, "r")
         self.freq = np.array(data_fh.variables['Frequency'][:])
         self.MeasX = np.array(data_fh.variables['MeasPosX'][:])
         self.MeasY = np.array(data_fh.variables['MeasPosY'][:])
         self.MeasZ = np.array(data_fh.variables['MeasPosZ'][:])
         self.Tx_re = np.array(data_fh.variables['Tx_re'][:,:])
         self.Tx_im = np.array(data_fh.variables['Tx_im'][:,:])
         self.Ty_re = np.array(data_fh.variables['Ty_re'][:,:])
         self.Ty_im = np.array(data_fh.variables['Ty_im'][:,:])
         self.dTx = np.array(data_fh.variables['dTx'][:,:])
         self.dTy = np.array(data_fh.variables['dTy'][:,:])             
         if 'Names' in data_fh.variables:   
           self.Names = data_fh.variables['Names'][:]
         else:
           self.Names = np.array(["Station {}".format(x) for x in range(len(self.MeasX))])
         data_fh.close()
     def WriteNetCDF(self,filename):
         data_fh = Dataset(filename, "w")
         nstat = self.MeasX.size
         nfreq = self.freq.size
         statdim = data_fh.createDimension("StationNumber", nstat)
         freqdim = data_fh.createDimension("Frequency", nfreq)     
         vnames = data_fh.createVariable("Names", str, ("StationNumber"))
         vnames[:] = self.Names
         vfreq = data_fh.createVariable("Frequency","f8",("Frequency"))
         vfreq[:] = self.freq
         vmeasx = data_fh.createVariable("MeasPosX","f8",("StationNumber"))
         vmeasx.units = "m"
         vmeasx[:] = self.MeasX[:]
         vmeasy = data_fh.createVariable("MeasPosY","f8",("StationNumber"))
         vmeasy.units = "m"
         vmeasy[:] = self.MeasY[:]
         vmeasz = data_fh.createVariable("MeasPosZ","f8",("StationNumber"))
         vmeasz.units = "m"
         vmeasz[:] = self.MeasZ[:]
         vtx_re = data_fh.createVariable("Tx_re","f8",("Frequency","StationNumber"))
         vtx_re.units = ""
         vtx_re[:,:] = self.Tx_re[:,:]
         vtx_im = data_fh.createVariable("Tx_im","f8",("Frequency","StationNumber"))
         vtx_im.units = ""
         vtx_im[:,:] = self.Tx_im[:,:]
         vty_re = data_fh.createVariable("Ty_re","f8",("Frequency","StationNumber"))
         vty_re.units = ""
         vty_re[:,:] = self.Ty_re[:,:]
         vty_im = data_fh.createVariable("Ty_im","f8",("Frequency","StationNumber"))
         vty_im.units = ""
         vty_im[:,:] = self.Ty_im[:,:]
         
         vdtx = data_fh.createVariable("dTx","f8",("Frequency","StationNumber"))
         vdtx.units = ""
         vdtx[:,:] = self.dTx[:,:]
         vdty = data_fh.createVariable("dTy","f8",("Frequency","StationNumber"))
         vdty.units = ""
         vdty[:,:] = self.dTy[:,:]
         
         data_fh.close()
     def RemoveStations(self,indices):
         self.MeasX = np.delete(self.MeasX,indices)
         self.MeasY = np.delete(self.MeasY,indices)
         self.MeasZ = np.delete(self.MeasZ,indices)
         self.Tx_re = np.delete(self.Tx_re,indices,1)
         self.Tx_im = np.delete(self.Tx_im,indices,1)
         self.Ty_re = np.delete(self.Ty_re,indices,1)
         self.Ty_im = np.delete(self.Ty_im,indices,1)
         
         self.dTx = np.delete(self.dTx,indices,1)
         self.dTy = np.delete(self.dTy,indices,1)
         
         self.Names = np.delete(self.Names,indices)
     def AddStations(self,otherdata):
        self.MeasX = np.concatenate((self.MeasX,otherdata.MeasX))
        self.MeasY = np.concatenate((self.MeasY,otherdata.MeasY))
        self.MeasZ = np.concatenate((self.MeasZ,otherdata.MeasZ))
        self.Tx_re = np.concatenate((self.Tx_re,otherdata.Tx_re),axis=1)
        self.Tx_im = np.concatenate((self.Tx_im,otherdata.Tx_im),axis=1)
        self.Ty_re = np.concatenate((self.Ty_re,otherdata.Ty_re),axis=1)
        self.Ty_im = np.concatenate((self.Ty_im,otherdata.Ty_im),axis=1)
        
        self.dTx = np.concatenate((self.dTx,otherdata.dTx),axis=1)
        self.dTy = np.concatenate((self.dTy,otherdata.dTy),axis=1)
        self.Names = np.concatenate((self.Names,otherdata.Names))
        