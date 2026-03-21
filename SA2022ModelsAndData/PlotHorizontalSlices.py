    #!/usr/bin/env python3

import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from pyproj import Proj
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import mtmodel
import mtclass
import matplotlib.patheffects as PathEffects
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages

import geopandas as gpd
from shapely.geometry import Polygon

bounds =[14,31,-32.5,-16.5]

arc = gpd.read_file('mccourt_2013_tectonic_units/archaean.shp')
lim = gpd.read_file('mccourt_2013_tectonic_units/limpopo.shp')
meso = gpd.read_file('mccourt_2013_tectonic_units/mesoproterozoic_belts.shp')
neo = gpd.read_file('mccourt_2013_tectonic_units/neoproterozoic.shp')
paleo = gpd.read_file('mccourt_2013_tectonic_units/paleoproterozoic_belts.shp')
paleos = gpd.read_file('mccourt_2013_tectonic_units/paleoproterozoic_shields.shp')
#faults.crs = {'init': 'epsg:4283'}
#f = faults.to_crs(crs='epsg:32654')
polygon = Polygon([(bounds[0], bounds[2]), (bounds[0], bounds[3]), (bounds[1], bounds[3]), (bounds[1], bounds[2]),
                   (bounds[0], bounds[2])])
#fc= gpd.clip(faults,polygon)


cm_data = np.loadtxt('imola.txt')
color_map = colors.LinearSegmentedColormap.from_list('imola',cm_data)

myProj = Proj("+proj=utm +zone=34k +ellps=WGS84 +datum=WGS84 +units=m +no_defs +south")
cy, cx = myProj(22.41878,-24.79278)

mtdata = mtclass.MTData()
mtdata.ReadModEM("ModEM_MaxData/ObservedData_Max.dat",center_utm=(cy,cx))
mlon, mlat = myProj(mtdata.MeasY,mtdata.MeasX,inverse=True)







def PlotHorSlice(Fig, Northing, Easting, Conductivity,stlat,stlon, index, xlabel=True, ylabel=True):
  ax_mt = Fig #mt_fig.add_subplot(111,projection=ccrs.Mercator(central_longitude=24))
  y1, x1 = np.meshgrid(Easting, Northing)
  lon, lat = myProj(y1,x1,inverse=True)

  rho = 1.0 / Conductivity
  ax_mt.set_extent(bounds, crs=ccrs.PlateCarree())
  m = ax_mt.pcolormesh(lon, lat, np.transpose(rho[index,:,:]),cmap='viridis_r',norm=matplotlib.colors.LogNorm(),linewidth=0,rasterized=True,vmin=5,vmax=5000,transform=ccrs.PlateCarree())
  #cs = ax_mt.contour(pseislon,pseislat,seisano[seisindex,:,:],np.arange(-1.0,1.0,0.2),transform=ccrs.PlateCarree(),colors='white')
  #ax_mt.clabel(cs)
  current_cmap = matplotlib.cm.get_cmap()
  current_cmap.set_bad(color='gray')
  ax_mt.scatter(stlon,stlat, transform=ccrs.PlateCarree(),c='black',s=1)      
  if xlabel:
    ax_mt.set_xlabel('Longitude')
  if ylabel:
    ax_mt.set_ylabel('Latitude')
  ax_mt.tick_params(axis='both', which='major', labelsize=8)
  ax_mt.grid(which = 'both')

  arc.boundary.plot(ax=ax_mt,color='black', transform=ccrs.PlateCarree())
  lim.boundary.plot(ax=ax_mt,color='black', transform=ccrs.PlateCarree())
  meso.boundary.plot(ax=ax_mt,color='black', transform=ccrs.PlateCarree())
  neo.boundary.plot(ax=ax_mt,color='black', transform=ccrs.PlateCarree())
  paleo.boundary.plot(ax=ax_mt,color='black', transform=ccrs.PlateCarree())
  paleos.boundary.plot(ax=ax_mt,color='black', transform=ccrs.PlateCarree())


  txt= ax_mt.text(24.5,-27.5,"KR",color="white", transform=ccrs.PlateCarree(),fontsize="medium",fontweight="bold")
  txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='black')])
  txt= ax_mt.text(19,-21,"DCB",color="white", transform=ccrs.PlateCarree(),fontsize="medium",ha="center",fontweight="bold")
  txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='black')])

  txt= ax_mt.text(22,-30,"SKC",color="white", transform=ccrs.PlateCarree(),fontsize="medium",ha="center",fontweight="bold")
  txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='black')])

  txt= ax_mt.text(24,-25,"MFC",color="white", transform=ccrs.PlateCarree(),fontsize="medium",ha="center",fontweight="bold")
  txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='black')])

  txt= ax_mt.text(28,-25,"BC",color="white", transform=ccrs.PlateCarree(),fontsize="medium",ha="center",fontweight="bold")
  txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='black')])
  ax_mt.set_facecolor('gray')
  #plt.colorbar(m)
        #self.ax_mt.set_ylim((self.min_x,self.max_x))

#mlon, mlat = myProj(np.array(self.station_posy)+cy,np.array(self.station_posx)+cx,inverse=True)
#ax_mt.scatter(mlon, mlat,marker = 'v',color = 'k',s = 1.5,transform=ccrs.PlateCarree())
  ax_mt.coastlines()
  gl = ax_mt.gridlines(draw_labels=True)
  gl.xlabels_top, gl.ylabels_right = False, False
  gl.xlocator = mticker.FixedLocator([16, 20, 24, 28])
  gl.ylocator = mticker.FixedLocator([-28,-24,-20])

  gl.xlines = False
  gl.ylines = False
  gl.xformatter = LONGITUDE_FORMATTER
  gl.yformatter = LATITUDE_FORMATTER
  gl.xlabel_style = {'fontsize': 15}
  gl.ylabel_style = {'fontsize': 15}

  return m




widths = [1, 1, 0.05]
heights = [1, 1]
model = mtmodel.MTModel()
model.ReadNetCDF('jif3D_Selected/jif3D_SA2022_selected.nc')


model2 = mtmodel.MTModel()
model2.ReadModEM('ModEM_Selected/ModEM_SA2022_Selected.rho')

model3 = mtmodel.MTModel()
model3.ReadNetCDF('jif3D_MaxData/jif3D_SA2022_Max.nc')

model4 = mtmodel.MTModel()
model4.ReadModEM('ModEM_MaxData/ModEM_SA2022_Max.rho')

sdata = mtclass.MTData()
sdata.ReadNetCDF("jif3D_Selected/jif3D_SA2022_selected.dist_imp.nc")
slon, slat = myProj(sdata.MeasY+cy,sdata.MeasX+cx,inverse=True)

Easting_jif3D = np.array(model.Easting) + cy
Northing_jif3D = np.array(model.Northing) + cx

Easting_modem = np.array(model2.Easting) + cy
Northing_modem = np.array(model2.Northing) + cx

with PdfPages('Horslices.pdf') as pdf:  
 for i in range(0,model.Depth.size-1):
  mt_fig = plt.figure(figsize = (11,10))
  mt_fig.suptitle('MT models at {:.2f} km depth'.format(model.Depth[i]/1000.0))
  spec = mt_fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths,
                          height_ratios=heights)

  



  fig1 = mt_fig.add_subplot(spec[0,0],projection=ccrs.Mercator(central_longitude=24))
  PlotHorSlice(fig1,Northing_jif3D, Easting_jif3D, model.Conductivity, slat, slon, i,False,True)
  fig1.set_title('jif3D selected')

  fig2 = mt_fig.add_subplot(spec[0,1],projection=ccrs.Mercator(central_longitude=24))
  PlotHorSlice(fig2,Northing_modem, Easting_modem, model2.Conductivity,  slat, slon, i,False,False)
  fig2.set_title('ModEM selected')


  fig3 = mt_fig.add_subplot(spec[1,0],projection=ccrs.Mercator(central_longitude=24))
  m = PlotHorSlice(fig3,Northing_jif3D, Easting_jif3D, model3.Conductivity,  mlat, mlon, i,True,True)
  fig3.set_title('jif3D all data')


  fig4 = mt_fig.add_subplot(spec[1,1],projection=ccrs.Mercator(central_longitude=24))
  m = PlotHorSlice(fig4,Northing_modem, Easting_modem, model4.Conductivity, mlat, mlon,i,True,False)
  fig4.set_title('ModEM all data')

  axes = plt.subplot(spec[0,2])
  plt.colorbar(m,cax=axes)
  axes.set_ylabel(r'$\rho$ [$\Omega m$]' )

  pdf.savefig(mt_fig)
  plt.close()


