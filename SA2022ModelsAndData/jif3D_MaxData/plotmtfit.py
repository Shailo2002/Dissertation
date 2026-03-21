#!~/bin/python3

from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import mtclass
from matplotlib.backends.backend_pdf import PdfPages
from pyproj import Proj




plt.style.use('seaborn-poster')

  
MTObs = mtclass.MTData()
MTObs.ReadNetCDF('jif3D_SA2022_Max.nc.dist_imp.nc')

MTComp = mtclass.MTData()
MTComp.ReadNetCDF('jif3D_SA2022_Max.nc.inv_imp.nc')

MTDiff = mtclass.MTData()
MTDiff.ReadNetCDF('jif3D_SA2022_Max.nc.diff_imp.nc')
Zxxdiff = np.sqrt((MTDiff.Zxx_re[:,:]**2 + MTDiff.Zxx_im[:,:]**2)/2)
Zxydiff = np.sqrt((MTDiff.Zxy_re[:,:]**2 + MTDiff.Zxy_im[:,:]**2)/2)
Zyxdiff = np.sqrt((MTDiff.Zyx_re[:,:]**2 + MTDiff.Zyx_im[:,:]**2)/2)
Zyydiff = np.sqrt((MTDiff.Zyy_re[:,:]**2 + MTDiff.Zyy_im[:,:]**2)/2)

period = 1.0/MTObs.freq
modified = False
nstats = MTObs.MeasX.size


i = 0

Rho_xx = MTObs.RhoXX()
Rho_xy = MTObs.RhoXY()
Rho_yx = MTObs.RhoYX()
Rho_yy = MTObs.RhoYY()

Phi_xx = MTObs.PhiXX()
Phi_xy = MTObs.Phi90XY()
Phi_yx = MTObs.Phi90YX()
Phi_yy = MTObs.PhiYY()

print (Phi_yx)

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
def plotrhophi(i):    
  fig = plt.figure(constrained_layout=True,figsize=(15,10))
  gs = fig.add_gridspec(2, 4,height_ratios=(1, 1), width_ratios=(1, 1, 0.1,1 ))
  ax1 = fig.add_subplot(gs[0, 0])
  ax2 = fig.add_subplot(gs[1, 0])
  ax3 = fig.add_subplot(gs[0, 1])   
  ax4 = fig.add_subplot(gs[1, 1])   
  ax5 = fig.add_subplot(gs[1, 2:4])   
  ax6 = fig.add_subplot(gs[0, 2])   


  
  ax5.scatter(MeasY,MeasX,s=5)
  ax5.scatter(MeasY[i],MeasX[i],s=50,color='r')
  ax5.set_xlabel('Easting')
  ax5.set_ylabel('Northing')
  ax5.yaxis.set_label_position("right")
  ax5.yaxis.tick_right()
  ax5.set_aspect(1.0)


  print ('Station Number: ', i+1, ' out of ',nstats)

  fig.suptitle('{}: {} X {} Y {}'.format(i,MTObs.Names[i], MTObs.MeasX[i], MTObs.MeasY[i])) 
  rxy=ax1.errorbar(period,Rho_xy[:,i],dRho_xy[:,i],linestyle='None',marker='o',mec='blue',label='xy obs',mfc='white',linewidth=2,mew=1.5,ecolor='blue',picker=5,zorder=0)
  ryx=ax1.errorbar(period,Rho_yx[:,i],dRho_yx[:,i],linestyle='None',marker='s',mec='red',label='yx obs',mfc='white',linewidth=2,mew=1.5,ecolor='red',picker=5,zorder=0)

  mf = ax1.scatter(period,Rho_xy[:,i],c=Zxydiff[:,i],cmap='plasma',vmin=0,vmax=10)
  ax1.scatter(period,Rho_yx[:,i],c=Zyxdiff[:,i],cmap='plasma',vmin=0,vmax=10)
  ax1.set_ylabel(r'$\rho_a$ ($\Omega$ m)')
  ax1.set_xscale('log')
  ax1.set_yscale('log')
  ax1.set_ylim([1,1e6])
  legend = ax1.legend(loc='lower left',numpoints=1) 
  c = plt.colorbar(mf,cax=ax6)
  c.set_label("RMS")

  # Set useblit=True on most backends for enhanced performance.
  pxy=ax2.errorbar(period,Phi_xy[:,i],dPhi_xy[:,i],linestyle='None',marker='o',mec='blue',mfc='white',linewidth=2,mew=1.5,ecolor='blue',picker=5)
  pyx=ax2.errorbar(period,Phi_yx[:,i],dPhi_yx[:,i],linestyle='None',marker='s',mec='red',mfc='white',linewidth=2,mew=1.5,ecolor='red',picker=5)
  ax2.set_xscale('log')
  ax2.set_yscale('linear')
  ax2.set_ylabel(r'$\Phi$ (deg)')
  ax2.set_xlabel(r'T (s)')
  ax2.set_ylim([0,90])
 
  rxx=ax3.errorbar(period,Rho_xx[:,i],dRho_xx[:,i],linestyle='None',marker='o',mec='blue',label='xx obs',mfc='white',linewidth=2,mew=1.5,ecolor='blue',picker=5,zorder=0)
  rYY=ax3.errorbar(period,Rho_yy[:,i],dRho_yy[:,i],linestyle='None',marker='o',mec='red',label='yy obs',mfc='white',linewidth=2,mew=1.5,ecolor='red',picker=5,zorder=0)

  sc = ax3.scatter(period,Rho_xx[:,i],c=Zxxdiff[:,i],cmap='plasma',vmin=0,vmax=10)
  ax3.scatter(period,Rho_yy[:,i],c=Zyydiff[:,i],cmap='plasma',vmin=0,vmax=10)
  ax3.yaxis.tick_right()
  ax3.set_xscale('log')
  ax3.set_yscale('log')
#  ax3.set_ylim([1e-4,10])
  legend = ax3.legend(loc='lower right',numpoints=1) 


  
  pxx=ax4.errorbar(period,Phi_xx[:,i],dPhi_xx[:,i],linestyle='None',marker='o',mec='blue',mfc='white',linewidth=2,mew=1.5,ecolor='blue',picker=5)
  pyy=ax4.errorbar(period,Phi_yy[:,i],dPhi_yy[:,i],linestyle='None',marker='s',mec='red',mfc='white',linewidth=2,mew=1.5,ecolor='red',picker=5)
  ax4.set_xscale('log')
  ax4.set_yscale('linear')
  ax4.yaxis.tick_right()
  ax4.set_xlabel(r'T (s)')
  ax4.set_ylim([-180,180])

  
  
  CRho_xx = MTComp.RhoXX()
  CRho_xy = MTComp.RhoXY()
  CRho_yx = MTComp.RhoYX()
  CRho_yy = MTComp.RhoYY()

  CPhi_xx = MTComp.PhiXX()

  CPhi_xy = MTComp.Phi90XY()
  CPhi_yx = MTComp.Phi90YX()

  CPhi_yy = MTComp.PhiYY()
  Cfreq = MTComp.freq
  Cperiod = 1.0 / Cfreq

  CdRho_xx = MTComp.dRhoXX()
  CdPhi_xx = MTComp.dPhiXX()
  CdRho_xy = MTComp.dRhoXY()
  CdPhi_xy = MTComp.dPhiXY()
  CdRho_yx = MTComp.dRhoYX()
  CdPhi_yx = MTComp.dPhiYX()
  CdRho_yy = MTComp.dRhoYY()
  CdPhi_yy = MTComp.dPhiYY()
  ax1.plot(Cperiod,CRho_xy[:,i],color='blue',linewidth=2, zorder=3)
  ax1.plot(Cperiod,CRho_yx[:,i],color='red',linewidth=2, zorder=3)

  ax2.plot(Cperiod,CPhi_xy[:,i],color='blue',linewidth=2, zorder=3, label = 'xy synth')
  ax2.plot(Cperiod,CPhi_yx[:,i],color='red',linewidth=2, zorder=3, label = 'yx synth')
  ax2.legend(loc='lower left',numpoints=1)

  ax3.plot(Cperiod,CRho_xx[:,i],color='blue',linewidth=2, zorder=3)
  ax3.plot(Cperiod,CRho_yy[:,i],color='red',linewidth=2, zorder=3)

  ax4.plot(Cperiod,CPhi_xx[:,i],color='blue',linewidth=2, zorder=3, label = 'xx synth')
  ax4.plot(Cperiod,CPhi_yy[:,i],color='red',linewidth=2, zorder=3, label = 'yy synth')
  ax4.legend(loc='lower left',numpoints=1)
  #fig.canvas.draw_idle()
  return fig

  


with PdfPages('mtfit.pdf') as pdf:
 for index in range(nstats):
   f = plotrhophi(index)
   pdf.savefig(f)
   plt.close()




