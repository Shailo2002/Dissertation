#!/usr/bin/python3
import netCDF4
from netCDF4 import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pyproj import Proj
import rasterio
import mtmodel

myProj = Proj("+proj=utm +zone=34k +ellps=WGS84 +datum=WGS84 +units=m +no_defs +south")
cy, cx = myProj(22.41878,-24.79278)


model = mtmodel.MTModel()
model.ReadNetCDF('jif3D_MaxData/jif3D_SA2022_Max.nc')



res = 1.0 /model.Conductivity
x1, y1 = np.meshgrid(model.Easting+cy, model.Northing+cx)
bounds =[np.min(x1),np.max(x1),np.min(y1),np.max(y1)]
  

def TiffSlice(lon, lat, Depth, res, index):
  fig = plt.figure()
  z = np.transpose(res[index,:,:])
  m = plt.pcolormesh(x1, y1, z,cmap='viridis_r',norm=matplotlib.colors.LogNorm(),linewidth=0,rasterized=True,vmin=5,vmax=5000)
  plt.xticks([])
  plt.yticks([])
  return fig
  
  
  
  
def ToGeoTiff(input, output, bounds):
   dataset = rasterio.open(input, 'r')
   bands = [1, 2, 3]
   data = dataset.read(bands)
   transform = rasterio.transform.from_bounds(bounds[0], bounds[2], bounds[1], bounds[3], data.shape[1], data.shape[2])
   crs = {'init': 'EPSG:32734'}

   with rasterio.open(output, 'w', driver='GTiff',
                   width=data.shape[1], height=data.shape[2],
                   count=3, dtype=data.dtype, nodata=0,
                   transform=transform, crs=crs) as dst:
        dst.write(data, indexes=bands)


    

for index in range(0,model.Depth.size-1):
   src = 'slice_{:.2f}.png'.format(model.Depth[index]/1000.0)
   dstDS = 'slice_{:.2f}.tif'.format(model.Depth[index]/1000.0)
   f = TiffSlice(x1,y1,model.Depth,res,index)
   f.savefig(src,bbox_inches='tight',dpi=600,pad_inches=0.0)
   ToGeoTiff(src,dstDS,bounds)
   plt.close()


