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
from scipy.stats.mstats import ttest_ind
import os

#添加色标
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

#PDINT
filesdir = "/home/users/zhangjk/code/2026-10/QBO/MIROC-ES2H/out/"
df = xr.open_dataset(filesdir+"MIROC-ES2H_ua_PDINT_WQBO_EQBO_diff_bootstrap_r1.nc",decode_times=False)
EQBO_WQBO_diff_PDINT = df["QBO"]
p_nino_PDINT = df["QBO_P"]
QBO_PDINT = df["EQQBO_WQBO_diff"]
QBO_PDINT_mean = QBO_PDINT.mean(dim=["ensemble"])
print(QBO_PDINT)
del df,filesdir
#PDNINT
filesdir = "/home/users/zhangjk/code/2026-10/QBO/MIROC-ES2H/out/"
df = xr.open_dataset(filesdir+"MIROC-ES2H_ua_PDNINT_WQBO_EQBO_diff_bootstrap_r1.nc",decode_times=False)
EQBO_WQBO_diff_PDNINT = df["QBO"]
p_nino_PDNINT = df["QBO_P"]
QBO_PDNINT = df["EQQBO_WQBO_diff"]
QBO_PDNINT_mean = QBO_PDNINT.mean(dim=["ensemble"])
print(QBO_PDNINT)
del df,filesdir

#计算差异
EQBO_PDINT_PDNINT = QBO_PDINT_mean-QBO_PDNINT_mean 
_,p_nino = ttest_ind(QBO_PDINT,QBO_PDNINT,equal_var=False)
print(EQBO_PDINT_PDNINT)

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
fig = plt.figure(figsize=(1.6,0.5),dpi=5000)
ax1=plt.subplot(1,3,1)
ax2=plt.subplot(1,3,2)
ax3=plt.subplot(1,3,3)

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
ax1.set_ylim(300,5)
for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
ax1.set_ylabel('Plev(hPa)',fontsize=1.8,labelpad=0.7)
ax1.set_xlabel('Lat',fontsize=1.8,labelpad=0.4)

#绘图1
lat = EQBO_WQBO_diff_PDINT["lat"]
level = EQBO_WQBO_diff_PDINT["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax1.contourf(date2d,lev2d,EQBO_WQBO_diff_PDINT,extend='both',cmap=cmaps.MPL_bwr,levels=np.arange(-20.0,20.1,2.0))
ax1.contourf(date2d,lev2d,p_nino_PDINT,levels=[0,0.1,1], hatches=['...............',None],zorder=1,colors="none")
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
ax1.set_position([chartBox.x0+0.02, chartBox.y0+0.15,chartBox.width*0.95,chartBox.height*0.85])
#添加色标
cax = add_down_cax(ax1, pad=0.13, height=0.021)
cbar = fig.colorbar(c1,cax,ticks=np.arange(-20.0,20.1,10.0),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$U$"+r" (m/s)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax1.set_title('(a) PDINT(EQBO-WQBO)',fontsize=1.6,loc='center',pad=1.1,weight='bold')

#图2
ax2.set_yscale("log")
y=[500.,300., 200., 100.,70.,  50.,  30.,  20.,10.,5.]
ax2.set_yticks(y)
ax2.set_yticklabels(y)
ax2.invert_yaxis()
# ax.set_ylim(200, 70)
ax2.xaxis.set_major_formatter (LatitudeFormatter())
ax2.set_xticks(np.arange(-90,120,30))
ax2.set_xlim(-90,90)
ax2.set_ylim(300,5)
for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
ax2.set_ylabel('Plev(hPa)',fontsize=1.8,labelpad=0.7)
ax2.set_xlabel('Lat',fontsize=1.8,labelpad=0.4)

#绘图1
lat = EQBO_WQBO_diff_PDNINT["lat"]
level = EQBO_WQBO_diff_PDNINT["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax2.contourf(date2d,lev2d,EQBO_WQBO_diff_PDNINT,extend='both',cmap=cmaps.MPL_bwr,levels=np.arange(-20.0,20.1,2.0))
ax2.contourf(date2d,lev2d,p_nino_PDNINT,levels=[0,0.1,1], hatches=['...............',None],zorder=1,colors="none")
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
ax2.set_position([chartBox.x0+0.02, chartBox.y0+0.15,chartBox.width*0.95,chartBox.height*0.85])
#添加色标
cax = add_down_cax(ax2, pad=0.13, height=0.021)
cbar = fig.colorbar(c1,cax,ticks=np.arange(-20.0,20.1,10.0),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$U$"+r" (m/s)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax2.set_title('(b) PDNINT(EQBO-WQBO)',fontsize=1.6,loc='center',pad=1.1,weight='bold')

#画图3
ax3.set_yscale("log")
y=[500.,300., 200., 100.,70.,  50.,  30.,  20.,10.,5.]
ax3.set_yticks(y)
ax3.set_yticklabels(y)
ax3.invert_yaxis()
# ax.set_ylim(200, 70)
ax3.xaxis.set_major_formatter (LatitudeFormatter())
ax3.set_xticks(np.arange(-90,120,30))
ax3.set_xlim(-90,90)
ax3.set_ylim(300,5)
for spine in ax3.spines.values():
        spine.set_linewidth(1.2)
ax3.set_ylabel('Plev(hPa)',fontsize=1.8,labelpad=0.7)
ax3.set_xlabel('Lat',fontsize=1.8,labelpad=0.4)

#绘图3
lat = EQBO_PDINT_PDNINT["lat"]
level = EQBO_PDINT_PDNINT["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax3.contourf(date2d,lev2d,EQBO_PDINT_PDNINT,extend='both',cmap=cmaps.MPL_bwr,levels=np.arange(-6.0,6.1,0.5))
ax3.contourf(date2d,lev2d,p_nino,levels=[0,0.05,1], hatches=['...............',None],zorder=1,colors="none")
#
ax3.tick_params(labelsize=1.7, pad=0.3)
ax3.spines['top'].set_linewidth(0.2)
ax3.spines['bottom'].set_linewidth(0.2)
ax3.spines['left'].set_linewidth(0.2)
ax3.spines['right'].set_linewidth(0.2)
ax3.tick_params(axis="both", which="major", direction="out", width=0.2, length=0.7)
ax3.tick_params(axis="y", which="minor", direction="out", width=0.0, length=0.0)
y_formatter = FormatStrFormatter('%1.0f')
ax3.yaxis.set_major_formatter(y_formatter)
chartBox = ax3.get_position()
ax3.set_position([chartBox.x0+0.02, chartBox.y0+0.15,chartBox.width*0.95,chartBox.height*0.85])
#添加色标
cax = add_down_cax(ax3, pad=0.13, height=0.021)
cbar = fig.colorbar(c1,cax,ticks=np.arange(-6.0,6.1,2.0),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$U$"+r" (m/s)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax3.set_title('(c) PDINT-PDNINT',fontsize=1.6,loc='center',pad=1.1,weight='bold')

ax1.set_rasterized(True)
ax2.set_rasterized(True)
ax3.set_rasterized(True)
plt.savefig("MIROC-ES2H_ua_PDINT-PDNINT_diff_bootsrrtap_r1.png",dpi=8000,format="png",facecolor="white")
plt.show()
