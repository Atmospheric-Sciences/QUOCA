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

#面积加权
def AreaWeightMean(data2D, lat, lon):
    '''
    data2D: 要进行区域加权平均的变量  2D: [lat, lon]
    lat: data2D对应的纬度 1D -90° 和 90°  因为NCL 和 Python 计算 np.cos(90 * rad) 值差的很大，故使用89.9°进行替代计算
    lon: data2D对应的经度 1D
    '''
    if lat.data[0] == 90:
        lat.data[0] = 89.9
        pass
    if lat.data[-1] == -90:
        lat.data[-1] = -89.9
        pass

    jlat = lat.shape[0]
    rad = 4.0 * np.arctan(1.0) / 180.0
    re = 6371220.0
    rr = re * rad
    dlon = np.abs(lon[2] - lon[1]) * rr
    dx = dlon * np.cos(lat * rad)
    dy = np.zeros(jlat)
    dy[0] = np.abs(lat[2] - lat[1]) * rr
    dy[1: jlat - 1] = np.abs(lat[2: jlat].values-lat[0: jlat - 2].values)*rr * 0.5
    dy[jlat - 1] = abs(lat[jlat - 1] - lat[jlat - 2]) * rr
    area = dx * dy
    
    weight = np.expand_dims(area, 1).repeat(len(lon), axis=1)
    new_data = np.average(data2D, weights=weight)
    return new_data

#计算数据异常
def slove_anomaly(date,clm):
    #计算数据的异常-新建数组
    date_clm = np.empty((len(date["year"]),len(date["plev"]),len(date["lat"]),len(date["lon"])))
    date_clm = xr.DataArray(date_clm, coords=[date["year"],date["plev"],date["lat"],\
                                  date["lon"]], dims=['year',"plev","lat","lon"])
    #计算数据的异常
    for m in range(len(date["year"])):
        for n in range(len(date["plev"])):
            date_clm[m,n,:,:] = date[m,n,:,:]-clm[n,:,:]
    #返回数据的异常
    return date_clm

#算面积加权
def weight_word(model_date):
    #定义数组保存加权平均后的数据
    if model_date.ndim >= 4:
        date_clm = np.empty((len(model_date["year"]),len(model_date["plev"])))
        date_clm = xr.DataArray(date_clm, coords=[model_date["year"],model_date["plev"]], dims=['year', 'plev'])
        for mm in range(len(model_date["year"])):
            for nn in range(len(model_date["plev"])):
                date_clm[mm,nn] = AreaWeightMean(model_date[mm,nn,:,:],model_date["lat"],model_date["lon"])
        return date_clm
    else:
        date_clm = np.empty((len(model_date["year"])))
        date_clm = xr.DataArray(date_clm, coords=[model_date["year"]], dims=['year'])
        for mm in range(len(model_date["year"])):
            date_clm[mm] = AreaWeightMean(model_date[mm,:,:],model_date["lat"],model_date["lon"])
        return date_clm

#滑动平均
def np_move_avg(a,n,mode):
    return(np.convolve(a, np.repeat(1/n, n), mode=mode))

####################################################
#PDINT temperature
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
PDINT_climate_series = obs_mean_climate.loc["2000":"2089",:,:]
PDINT_climate = (obs_mean_climate.loc["2000":"2089",:,:]).mean(dim=["year"])
print(PDINT_climate)
del filesdir,ta_sel,df,lev,obs_mean,obs_mean_climate
####################################################
#PDNINT temperature
####################################################
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/MIROC/MIROC-ES2H/PDNINT/r1i1p1f1/mon/ta/gn/v20241223/"
df = xr.open_dataset(filesdir+"ta_mon_MIROC-ES2H_PDNINT_r1i1p1f1_gn_200001-202912.nc")
ta_sel = df['ta'].loc[:,50000.:500.,:,:]
obs_mean_r1 = ta_sel.groupby('time.year').mean(dim='time')
del ta_sel,df
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/MIROC/MIROC-ES2H/PDNINT/r2i1p1f1/mon/ta/gn/v20241223/"
df = xr.open_dataset(filesdir+"ta_mon_MIROC-ES2H_PDNINT_r2i1p1f1_gn_203001-205912.nc")
ta_sel = df['ta'].loc[:,50000.:500.,:,:]
obs_mean_r2 = ta_sel.groupby('time.year').mean(dim='time')
del ta_sel,df
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/MIROC/MIROC-ES2H/PDNINT/r3i1p1f1/mon/ta/gn/v20241223/"
df = xr.open_dataset(filesdir+"ta_mon_MIROC-ES2H_PDNINT_r3i1p1f1_gn_206001-208912.nc")
ta_sel = df['ta'].loc[:,50000.:500.,:,:]
obs_mean_r3 = ta_sel.groupby('time.year').mean(dim='time')
del ta_sel,df
#contact
obs_mean = xr.concat([obs_mean_r1, obs_mean_r2, obs_mean_r3], dim='year')
print(obs_mean)
del obs_mean_r1, obs_mean_r2, obs_mean_r3
#计算气候态
obs_mean_climate = obs_mean.mean(dim=["lon"])
print(obs_mean_climate)
PDNINT_climate = (obs_mean_climate.loc["2000":"2089",:,:]).mean(dim=["year"])
PDNINT_climate_series = obs_mean_climate.loc["2000":"2089",:,:]
print(np.max(PDNINT_climate))
print(np.min(PDNINT_climate))
del obs_mean,obs_mean_climate

###############################################################
#difference  
###############################################################
PDINT_PDNINT_climate = PDINT_climate - PDNINT_climate
print(PDINT_PDNINT_climate)
print(np.max(PDINT_PDNINT_climate))
print(np.min(PDINT_PDNINT_climate))
_,p_nino = ttest_ind(PDINT_climate_series,PDNINT_climate_series,equal_var=False)

#draw
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
ax1.set_ylim(500,10)
for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
ax1.set_ylabel('Plev(hPa)',fontsize=1.8,labelpad=0.7)
ax1.set_xlabel('Lat',fontsize=1.8,labelpad=0.4)

#绘图1
lat = PDINT_climate["lat"]
level = PDINT_climate["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax1.contourf(date2d,lev2d,PDINT_climate,extend='both',cmap=cmaps.WhiteYellowOrangeRed,levels=np.arange(180.0,300.1,5))
#ax1.contourf(date2d,lev2d,ch4_slope[0,:,:],levels=[0,0.1,1], hatches=['...............',None],zorder=1,colors="none")
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
cbar = fig.colorbar(c1,cax,ticks=np.arange(180.0,300.1,20),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$T$"+r" (K)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax1.set_title('(a) PDINT(MIROC-ES2H)',fontsize=1.6,loc='center',pad=1.1,weight='bold')

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
lat = PDNINT_climate["lat"]
level = PDNINT_climate["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax2.contourf(date2d,lev2d,PDNINT_climate,extend='both',cmap=cmaps.WhiteYellowOrangeRed,levels=np.arange(180.0,300.1,5))
#ax1.contourf(date2d,lev2d,ch4_slope[0,:,:],levels=[0,0.1,1], hatches=['...............',None],zorder=1,colors="none")
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
cbar = fig.colorbar(c1,cax,ticks=np.arange(180.0,300.1,20),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$T$"+r" (K)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax2.set_title('(b) PDNINT(MIROC-ES2H)',fontsize=1.6,loc='center',pad=1.1,weight='bold')

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
ax3.set_ylim(500,10)
for spine in ax3.spines.values():
        spine.set_linewidth(1.2)
ax3.set_ylabel('Plev(hPa)',fontsize=1.8,labelpad=0.7)
ax3.set_xlabel('Lat',fontsize=1.8,labelpad=0.4)

#绘图3
lat = PDINT_PDNINT_climate["lat"]
level = PDINT_PDNINT_climate["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax3.contourf(date2d,lev2d,PDINT_PDNINT_climate,extend='both',cmap=cmaps.MPL_bwr,levels=np.arange(-1.6,1.61,0.1))
ax3.contourf(date2d,lev2d,p_nino,levels=[0,0.1,1], hatches=['...............',None],zorder=1,colors="none")
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
ax3.set_position([chartBox.x0, chartBox.y0+0.15,chartBox.width*0.95,chartBox.height*0.85])
#添加色标
cax = add_down_cax(ax3, pad=0.13, height=0.021)
cbar = fig.colorbar(c1,cax,ticks=np.arange(-1.6,1.61,0.4),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$T$"+r" (K)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax3.set_title('(c) PDINT-PDNINT',fontsize=1.6,loc='center',pad=1.1,weight='bold')



ax1.set_rasterized(True)
ax2.set_rasterized(True)
ax3.set_rasterized(True)
plt.savefig("PDINT_PDNIT_ta_climate_diff_MIROC.png",dpi=8000,format="png",facecolor="white")
plt.show()

