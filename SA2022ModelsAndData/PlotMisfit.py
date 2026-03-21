#!~/bin/python3
import netCDF4
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import mtclass
from matplotlib.backends.backend_pdf import PdfPages
from pyproj import Proj

plt.style.use('seaborn-poster')

  


def plotrhophi(MTObs,MTSynth,i):    
  Rho_xx = MTObs.RhoXX()
  Rho_xy = MTObs.RhoXY()
  Rho_yx = MTObs.RhoYX()
  Rho_yy = MTObs.RhoYY()

  Phi_xx = MTObs.PhiXX()
  Phi_xy = MTObs.Phi90XY()
  Phi_yx = MTObs.Phi90YX()
  Phi_yy = MTObs.PhiYY()
  
  dRho_xx = MTObs.dRhoXX()
  dPhi_xx = MTObs.dPhiXX()
  dRho_xy = MTObs.dRhoXY()
  dPhi_xy = MTObs.dPhiXY()
  dRho_yx = MTObs.dRhoYX()
  dPhi_yx = MTObs.dPhiYX()
  dRho_yy = MTObs.dRhoYY()
  dPhi_yy = MTObs.dPhiYY()
  MeasX = MTObs.MeasX
  MeasY = MTObs.MeasY
  
  fig = plt.figure(constrained_layout=True,figsize=(15,15))
  gs = fig.add_gridspec(2, 3)
  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[1, 0])
  ax3 = fig.add_subplot(gs[0, 1])   
  ax4 = fig.add_subplot(gs[1, 1])   
  ax5 = fig.add_subplot(gs[1, 2])   
 
  
  ax5.scatter(MeasY,MeasX,s=5)
  ax5.scatter(MeasY[i],MeasX[i],s=50,color='r')
  ax5.set_xlabel('Easting')
  ax5.set_ylabel('Northing')
  ax5.yaxis.set_label_position("right")
  ax5.yaxis.tick_right()
  ax5.set_aspect(1.0)

  if (i % 20 == 0):
    print ('Station Number: ', i+1, ' out of ',nstats)

  fig.suptitle('{}: {} X {} Y {}'.format(i,MTObs.Names[i], MTObs.MeasX[i], MTObs.MeasY[i])) 
  ind1 = (Rho_xy[:,i] < 1e7) 
  rxy=ax1.errorbar(MTObs.freq[ind1],Rho_xy[ind1,i],dRho_xy[ind1,i],linestyle='None',marker='o',mec='blue',label='xy',mfc='white',linewidth=2,mew=1.5,ecolor='blue',picker=5,zorder=0)
  ind2 = (Rho_yx[:,i] < 1e7) 
  ryx=ax1.errorbar(MTObs.freq[ind2],Rho_yx[ind2,i],dRho_yx[ind2,i],linestyle='None',marker='s',mec='red',label='yx',mfc='white',linewidth=2,mew=1.5,ecolor='red',picker=5,zorder=0)

  ax1.set_ylabel(r'$\rho_a$ ($\Omega$ m)')
  ax1.set_xscale('log')
  ax1.set_yscale('log')
  ax1.invert_xaxis()
  legend = ax1.legend(loc='upper right',numpoints=1) 


  pxy=ax2.errorbar(MTObs.freq[ind1],Phi_xy[ind1,i],dPhi_xy[ind1,i],linestyle='None',marker='o',mec='blue',mfc='white',linewidth=2,mew=1.5,ecolor='blue',picker=5)
  pyx=ax2.errorbar(MTObs.freq[ind2],Phi_yx[ind2,i],dPhi_yx[ind2,i],linestyle='None',marker='s',mec='red',mfc='white',linewidth=2,mew=1.5,ecolor='red',picker=5)
  ax2.set_xscale('log')
  ax2.set_yscale('linear')
  ax2.set_ylabel(r'$\Phi$ (deg)')
  ax2.set_xlabel(r'f [Hz]')
  ax2.invert_xaxis()
  ax2.set_ylim([0,90])
  ax2.set_yticks([0,15,30,45,60,75,90])

  ind3 = (Rho_xx[:,i] < 1e7)
  ind4 = (Rho_yy[:,i] < 1e7)

  rxx=ax3.errorbar(MTObs.freq[ind3],Rho_xx[ind3,i],dRho_xx[ind3,i],linestyle='None',marker='o',mec='blue',label='xx',mfc='white',linewidth=2,mew=1.5,ecolor='blue',picker=5,zorder=0)
  rYY=ax3.errorbar(MTObs.freq[ind4],Rho_yy[ind4,i],dRho_yy[ind4,i],linestyle='None',marker='o',mec='red',label='yy',mfc='white',linewidth=2,mew=1.5,ecolor='red',picker=5,zorder=0)

  ax3.yaxis.tick_right()
  ax3.set_xscale('log')
  ax3.set_yscale('log')
  ax3.invert_xaxis()
#  ax3.set_ylim([1e-4,10])
  legend = ax3.legend(loc='lower right',numpoints=1) 


  
  pxx=ax4.errorbar(MTObs.freq[ind3],Phi_xx[ind3,i],dPhi_xx[ind3,i],linestyle='None',marker='o',mec='blue',mfc='white',linewidth=2,mew=1.5,ecolor='blue',picker=5)
  pyy=ax4.errorbar(MTObs.freq[ind4],Phi_yy[ind4,i],dPhi_yy[ind4,i],linestyle='None',marker='s',mec='red',mfc='white',linewidth=2,mew=1.5,ecolor='red',picker=5)
  ax4.set_xscale('log')
  ax4.set_yscale('linear')
  ax4.yaxis.tick_right()
  ax4.set_xlabel(r'f [Hz]')
  ax4.invert_xaxis()
  ax4.set_ylim([-180,180])
  ax4.set_yticks([-180,-90,0,90,180])
  

 
  ax1.plot(MTSynth.freq,MTSynth.RhoXY()[:,i],color='blue',linewidth=2, zorder=3)
  ax1.plot(MTSynth.freq,MTSynth.RhoYX()[:,i],color='red',linewidth=2, zorder=3)

  ax2.plot(MTSynth.freq,MTSynth.Phi90XY()[:,i],color='blue',linewidth=2, zorder=3)
  ax2.plot(MTSynth.freq,MTSynth.Phi90YX()[:,i],color='red',linewidth=2, zorder=3)

  ax3.plot(MTSynth.freq,MTSynth.RhoXX()[:,i],color='blue',linewidth=2, zorder=3)
  ax3.plot(MTSynth.freq,MTSynth.RhoYY()[:,i],color='red',linewidth=2, zorder=3)

  ax4.plot(MTSynth.freq,MTSynth.PhiXX()[:,i],color='blue',linewidth=2, zorder=3)
  ax4.plot(MTSynth.freq,MTSynth.PhiYY()[:,i],color='red',linewidth=2, zorder=3)


  return fig


print ('Creating file: mtfit_jif3D_selected.pdf') 
MTObs = mtclass.MTData()
MTObs.ReadNetCDF('jif3D_Selected/jif3D_SA2022_selected.dist_imp.nc')

MTSynth = mtclass.MTData()
MTSynth.ReadNetCDF('jif3D_Selected/jif3D_SA2022_selected.inv_imp.nc')

nstats = MTObs.MeasX.size
with PdfPages('mtfit_jif3D_selected.pdf') as pdf:
 for index in range(nstats):
   f = plotrhophi(MTObs,MTSynth,index)
   pdf.savefig(f)
   plt.close()


print ('Creating file: mtfit_jif3D_max.pdf')
MTObs = mtclass.MTData()
MTObs.ReadNetCDF('jif3D_MaxData/jif3D_SA2022_Max.dist_imp.nc')

MTSynth = mtclass.MTData()
MTSynth.ReadNetCDF('jif3D_MaxData/jif3D_SA2022_Max.inv_imp.nc')

nstats = MTObs.MeasX.size
with PdfPages('mtfit_jif3D_max.pdf') as pdf:
 for index in range(nstats):
   f = plotrhophi(MTObs,MTSynth,index)
   pdf.savefig(f)
   plt.close()
   

myProj = Proj("+proj=utm +zone=34k +ellps=WGS84 +datum=WGS84 +units=m +no_defs +south")
cy, cx = myProj(22.41878,-24.79278)


print ('Creating file: mtfit_ModEM_max.pdf')
MTObs = mtclass.MTData()
MTObs.ReadModEM('ModEM_MaxData/ObservedData_Max.dat',center_utm=(cy,cx))

MTSynth = mtclass.MTData()
MTSynth.ReadModEM('ModEM_MaxData/ModEM_SA2022_Max.dat',center_utm=(cy,cx))

nstats = MTObs.MeasX.size
with PdfPages('mtfit_ModEM_maxdata.pdf') as pdf:
 for index in range(nstats):
   f = plotrhophi(MTObs,MTSynth,index)
   pdf.savefig(f)
   plt.close()
   
   
   
   
print ('Creating file: mtfit_Modem_selected.pdf')
MTObs = mtclass.MTData()
MTObs.ReadModEM('ModEM_Selected/ObservedData_Selected.dat',center_utm=(cy,cx))

MTSynth = mtclass.MTData()
MTSynth.ReadModEM('ModEM_Selected/ModEM_SA2022_Selected.dat',center_utm=(cy,cx))

nstats = MTObs.MeasX.size
with PdfPages('mtfit_ModEM_selected.pdf') as pdf:
 for index in range(nstats):
   f = plotrhophi(MTObs,MTSynth,index)
   pdf.savefig(f)
   plt.close()  
   
   
print ('Creating file: mtfit_Modem_median.pdf')
MTObs = mtclass.MTData()
MTObs.ReadModEM('ModEM_Median/ObservedData_Median.dat',center_utm=(cy,cx))

MTSynth = mtclass.MTData()
MTSynth.ReadModEM('ModEM_Median/ModEM_SA2022_Median.dat',center_utm=(cy,cx))

nstats = MTObs.MeasX.size
with PdfPages('mtfit_ModEM_median.pdf') as pdf:
 for index in range(nstats):
   f = plotrhophi(MTObs,MTSynth,index)
   pdf.savefig(f)
   plt.close()  

   

