import numpy as np
import scipy as sp
from scipy import io
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle
import cmaps
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from scipy import signal
from matplotlib import ticker
from   scipy.stats import linregress
from cartopy.util import add_cyclic_point
import matplotlib.colors as mcolors
import statsmodels.api as sm
from scipy import stats
import matplotlib as mpl
from tqdm import tqdm
import glob
import os
from scipy.stats.mstats import ttest_ind

####################################################
#MIROC-ES2H PDINT temperature
####################################################
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/MIROC/MIROC-ES2H/PDINT/r1i1p1f1/mon/ta/gn/v20241223/"
df = xr.open_dataset(filesdir+"ta_mon_MIROC-ES2H_PDINT_r1i1p1f1_gn_200001-208912.nc")
print(df)
lev = df["plev"]
print(lev)
ta_sel = df['ta'].loc[:,50000.:500.,:,:]
obs_mean = ta_sel.groupby('time.year').mean(dim='time')
print(obs_mean)
obs_mean_climate = obs_mean.mean(dim=["lon"])
print(obs_mean_climate)
PDINT_climate_series = obs_mean_climate.loc["2000":"2029",:,:]
PDINT_climate = (obs_mean_climate.loc["2000":"2029",:,:]).mean(dim=["year"])
print(PDINT_climate)
del filesdir,ta_sel,df,lev,obs_mean,obs_mean_climate
####################################################
#MIROC-ES2H PDNINT temperature
####################################################
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/MIROC/MIROC-ES2H/PDNINT/r1i1p1f1/mon/ta/gn/v20241223/"
df = xr.open_dataset(filesdir+"ta_mon_MIROC-ES2H_PDNINT_r1i1p1f1_gn_200001-202912.nc")
ta_sel = df['ta'].loc[:,50000.:500.,:,:]
obs_mean = ta_sel.groupby('time.year').mean(dim='time')
#计算气候态
PDNINT_climate_series = obs_mean.mean(dim=["lon"])
print(PDNINT_climate_series)
PDNINT_climate = PDNINT_climate_series.mean(dim=["year"])
print(np.max(PDNINT_climate))
print(np.min(PDNINT_climate))
del filesdir,df,ta_sel,obs_mean

###############################################################
#MIROC-ES2H difference  
###############################################################
MIROC_PDINT_PDNINT_climate = PDINT_climate - PDNINT_climate
print(MIROC_PDINT_PDNINT_climate)
print(np.max(MIROC_PDINT_PDNINT_climate))
print(np.min(MIROC_PDINT_PDNINT_climate))
_,MIROC_p_nino = ttest_ind(PDINT_climate_series,PDNINT_climate_series,equal_var=False)
del PDINT_climate,PDNINT_climate,PDINT_climate_series,PDNINT_climate_series
print(MIROC_p_nino)

############################################################
#E3SMv3 PDINT temperature
############################################################
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/E3SMv3/QUOCA/LLNL/E3SM/PDINT/r1i1p1f1/mon/ta/gr/v20250728/"
df = xr.open_dataset(filesdir+"ta_mon_E3SM_PDINT_r1i1p1f1_gr_001101-010012.nc")
print(df)
lev = df["plev"]
print(lev)
ta_sel = df['ta'].loc[:,50000.:500.,:,:]
obs_mean = ta_sel.groupby('time.year').mean(dim='time')
print(obs_mean)
PDINT_climate_series  = obs_mean.loc["11":"40",:,:].mean(dim=["lon"])
print(PDINT_climate_series)
PDINT_climate = PDINT_climate_series.mean(dim=["year"])
print(PDINT_climate)
del filesdir,ta_sel,df,obs_mean,lev

############################################################
#E3SMv3 PDNINT temperature
############################################################
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/E3SMv3/QUOCA/LLNL/E3SM/PDNINT/r1i1p1f1/mon/ta/gr/v20250727/"
df = xr.open_dataset(filesdir+"ta_mon_E3SM_PDNINT_r1i1p1f1_gr_001101-004012.nc")
ta_sel = df['ta'].loc[:,50000.:500.,:,:]
obs_mean = ta_sel.groupby('time.year').mean(dim='time')
print(obs_mean)
#计算气候态
PDNINT_climate_series = obs_mean.mean(dim=["lon"])
print(PDNINT_climate_series)
PDNINT_climate = PDNINT_climate_series.mean(dim=["year"])
print(PDNINT_climate)
print(np.max(PDNINT_climate))
print(np.min(PDNINT_climate))
del filesdir,ta_sel,df,obs_mean

###############################################################
#E3SMv3 difference  
###############################################################
E3SMv3_PDINT_PDNINT_climate = PDINT_climate - PDNINT_climate
print(E3SMv3_PDINT_PDNINT_climate)
print(np.max(E3SMv3_PDINT_PDNINT_climate))
print(np.min(E3SMv3_PDINT_PDNINT_climate))
_,E3SMv3_p_nino = ttest_ind(PDINT_climate_series,PDNINT_climate_series)


#子函数
def add_down_cax(ax, pad, height):
    '''
    在一个ax上边追加与之等宽的cax.
    pad是cax与ax的间距.
    height是cax的宽度.
    '''
    axposition = ax.get_position()
    caxposition = mpl.transforms.Bbox.from_extents(
        axposition.x0,
        axposition.y0 - pad - height,
        axposition.x1,
        axposition.y0 - pad 
    )
    cax = ax.figure.add_axes(caxposition)
    return cax

#python画组图
import cmaps
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
from matplotlib.legend import Legend
from matplotlib.ticker import MultipleLocator
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as patches
config = {
            "font.family": 'STIXGeneral',
            "font.size": 2.0,
            "mathtext.fontset": 'stix',
            "font.serif": [''],#宋体
            'axes.unicode_minus': True # 处理负号
         }
rcParams.update(config)
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["hatch.linewidth"] = "0.14"
plt.rcParams["hatch.color"] = "black"
#plt.rcParams["hatch.color"] = "white"
plt.rc("axes",linewidth=0.3)

#画组图
fig = plt.figure(figsize=(1.0,0.5),dpi=5000)
ax1=plt.subplot(1,2,1)
ax2=plt.subplot(1,2,2)

#画图1
ax1.set_yscale("log")
y=[500.,300., 200., 100.,70.,  50.,  30.,  20.,10.,5.]
ax1.set_yticks(y)
ax1.set_yticklabels(y)
ax1.invert_yaxis()
# ax.set_ylim(200, 70)
ax1.xaxis.set_major_formatter (LatitudeFormatter())
ax1.set_xticks(np.arange(-90,120,30))
ax1.set_xlim(-90,90)
ax1.set_ylim(500,10)
for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
ax1.set_ylabel('Plev(hPa)',fontsize=1.8,labelpad=0.7)
ax1.set_xlabel('Lat',fontsize=1.8,labelpad=0.4)

#绘图1
lat = MIROC_PDINT_PDNINT_climate["lat"]
level = MIROC_PDINT_PDNINT_climate["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax1.contourf(date2d,lev2d,MIROC_PDINT_PDNINT_climate,extend='both',cmap=cmaps.MPL_bwr,levels=np.arange(-1.6,1.61,0.1))
ax1.contourf(date2d,lev2d,MIROC_p_nino,levels=[0,0.1,1], hatches=['...............',None],zorder=1,colors="none")
#
ax1.tick_params(labelsize=1.7, pad=0.3)
ax1.spines['top'].set_linewidth(0.2)
ax1.spines['bottom'].set_linewidth(0.2)
ax1.spines['left'].set_linewidth(0.2)
ax1.spines['right'].set_linewidth(0.2)
ax1.tick_params(axis="both", which="major", direction="out", width=0.2, length=0.7)
ax1.tick_params(axis="y", which="minor", direction="out", width=0.0, length=0.0)
y_formatter = FormatStrFormatter('%1.0f')
ax1.yaxis.set_major_formatter(y_formatter)
chartBox = ax1.get_position()
ax1.set_position([chartBox.x0, chartBox.y0+0.15,chartBox.width*0.95,chartBox.height*0.85])
#添加色标
cax = add_down_cax(ax1, pad=0.13, height=0.021)
cbar = fig.colorbar(c1,cax,ticks=np.arange(-1.6,1.61,0.4),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$T$"+r" (K)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax1.set_title('(a) PDINT-PDNINT(MIROC-ES2H)',fontsize=1.6,loc='center',pad=1.1,weight='bold')

#画图2
ax2.set_yscale("log")
y=[500.,300., 200., 100.,70.,  50.,  30.,  20.,10.,5.]
ax2.set_yticks(y)
ax2.set_yticklabels(y)
ax2.invert_yaxis()
# ax.set_ylim(200, 70)
ax2.xaxis.set_major_formatter (LatitudeFormatter())
ax2.set_xticks(np.arange(-90,120,30))
ax2.set_xlim(-90,90)
ax2.set_ylim(500,10)
for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
ax2.set_ylabel('Plev(hPa)',fontsize=1.8,labelpad=0.7)
ax2.set_xlabel('Lat',fontsize=1.8,labelpad=0.4)

#绘图2
lat = E3SMv3_PDINT_PDNINT_climate["lat"]
level = E3SMv3_PDINT_PDNINT_climate["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax2.contourf(date2d,lev2d,E3SMv3_PDINT_PDNINT_climate,extend='both',cmap=cmaps.MPL_bwr,levels=np.arange(-1.6,1.61,0.1))
ax2.contourf(date2d,lev2d,E3SMv3_p_nino,levels=[0,0.1,1], hatches=['...............',None],zorder=1,colors="none")
#
ax2.tick_params(labelsize=1.7, pad=0.3)
ax2.spines['top'].set_linewidth(0.2)
ax2.spines['bottom'].set_linewidth(0.2)
ax2.spines['left'].set_linewidth(0.2)
ax2.spines['right'].set_linewidth(0.2)
ax2.tick_params(axis="both", which="major", direction="out", width=0.2, length=0.7)
ax2.tick_params(axis="y", which="minor", direction="out", width=0.0, length=0.0)
y_formatter = FormatStrFormatter('%1.0f')
ax2.yaxis.set_major_formatter(y_formatter)
chartBox = ax2.get_position()
ax2.set_position([chartBox.x0, chartBox.y0+0.15,chartBox.width*0.95,chartBox.height*0.85])
#添加色标
cax = add_down_cax(ax2, pad=0.13, height=0.021)
cbar = fig.colorbar(c1,cax,ticks=np.arange(-1.6,1.61,0.4),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$T$"+r" (K)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax2.set_title('(b) PDINT-PDNINT(E3SMv3)',fontsize=1.6,loc='center',pad=1.1,weight='bold')


ax1.set_rasterized(True)
ax2.set_rasterized(True)
plt.savefig("PDINT_PDNIT_ta_climate_diff_E3SMv3_MIROC.png",dpi=8000,format="png",facecolor="white")
plt.show()

