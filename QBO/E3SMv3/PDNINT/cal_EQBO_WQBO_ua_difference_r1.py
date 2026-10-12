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
        date_clm = np.empty((len(model_date["time"]),len(model_date["plev"])))
        date_clm = xr.DataArray(date_clm, coords=[model_date["time"],model_date["plev"]], dims=['time', 'plev'])
        for mm in range(len(model_date["time"])):
            for nn in range(len(model_date["plev"])):
                date_clm[mm,nn] = AreaWeightMean(model_date[mm,nn,:,:],model_date["lat"],model_date["lon"])
        return date_clm
    else:
        date_clm = np.empty((len(model_date["time"])))
        date_clm = xr.DataArray(date_clm, coords=[model_date["time"]], dims=['time'])
        for mm in range(len(model_date["time"])):
            date_clm[mm] = AreaWeightMean(model_date[mm,:,:],model_date["lat"],model_date["lon"])
        return date_clm

#滑动平均
def np_move_avg(a,n,mode):
    return(np.convolve(a, np.repeat(1/n, n), mode=mode))

#选取冬季的数据
def convert_DJF(dataset):
    dataset_group = dataset.groupby('time.month')
    print(dataset_group)
    m1 = list(dataset_group)[0][1]
    m2 = list(dataset_group)[1][1]
    m12 = list(dataset_group)[11][1]
    #将12个月份加1
    time = m12["time"]
    new_time = pd.date_range(start="2001-12-01", periods=len(time), freq="YE").values
    m12["time"] = new_time
    #合并到统一的年份
    merge = xr.concat([m1,m2,m12],dim="time")
    DJF_year = merge.groupby("time.year").mean(dim=["time"]).rename({"year":"time"})
    return DJF_year

#数据标准化
def Z_score_normalize(data):
    mean = np.mean(data,axis=0)
    std_dev = np.std(data,axis=0)
    normalized_data = (data-mean)/std_dev
    return normalized_data

#定义文件的路径
#ua
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/E3SMv3/QUOCA/LLNL/E3SM/PDNINT/r1i1p1f1/mon/ua/gr/v20250727/"
df = xr.open_dataset(filesdir+"ua_mon_E3SM_PDNINT_r1i1p1f1_gr_001101-004012.nc")
obs_mean_r1 = df['ua'].loc[:,5000.,-10.:10.,:]
del df
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/E3SMv3/QUOCA/LLNL/E3SM/PDNINT/r2i1p1f1/mon/ua/gr/v20250726/"
df = xr.open_dataset(filesdir+"ua_mon_E3SM_PDNINT_r2i1p1f1_gr_004101-007012.nc")
obs_mean_r2 = df['ua'].loc[:,5000.,-10.:10.,:]
del df
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/E3SMv3/QUOCA/LLNL/E3SM/PDNINT/r3i1p1f1/mon/ua/gr/v20250725/"
df = xr.open_dataset(filesdir+"ua_mon_E3SM_PDNINT_r3i1p1f1_gr_007101-010012.nc")
obs_mean_r3 = df['ua'].loc[:,5000.,-10.:10.,:]
del df
#contact
obs_mean = xr.concat([obs_mean_r1, obs_mean_r2, obs_mean_r3], dim='time')
obs_mean ['time'] = pd.date_range(start=str(2000)+"-01-01", periods=obs_mean.sizes['time'], freq='MS')
del obs_mean_r1, obs_mean_r2, obs_mean_r3
print(obs_mean)
ua_sel_DJF = convert_DJF(obs_mean.loc["2000-12-01":"2029-02-01",-10.:10.,:])
print(ua_sel_DJF)
print(obs_mean.loc["2000-12-01":"2029-02-01",-10.:10.,:])
del obs_mean
# exit()

#ua
#定义文件的路径
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/E3SMv3/QUOCA/LLNL/E3SM/PDNINT/r1i1p1f1/mon/ua/gr/v20250727/"
df = xr.open_dataset(filesdir+"ua_mon_E3SM_PDNINT_r1i1p1f1_gr_001101-004012.nc")
obs_mean_r1 = df['ua'].loc[:,50000.:100.,:,:]
del df
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/E3SMv3/QUOCA/LLNL/E3SM/PDNINT/r2i1p1f1/mon/ua/gr/v20250726/"
df = xr.open_dataset(filesdir+"ua_mon_E3SM_PDNINT_r2i1p1f1_gr_004101-007012.nc")
obs_mean_r2 = df['ua'].loc[:,50000.:100.,:,:]
del df
filesdir = "/gws/ssde/j25b/qboi/quoca/phase1/E3SMv3/QUOCA/LLNL/E3SM/PDNINT/r3i1p1f1/mon/ua/gr/v20250725/"
df = xr.open_dataset(filesdir+"ua_mon_E3SM_PDNINT_r3i1p1f1_gr_007101-010012.nc")
obs_mean_r3 = df['ua'].loc[:,50000.:100.,:,:]
del df
#contact
obs_mean = xr.concat([obs_mean_r1, obs_mean_r2, obs_mean_r3], dim='time')
obs_mean['time'] = pd.date_range(start=str(2000)+"-01-01", periods=obs_mean.sizes['time'], freq='MS')
del obs_mean_r1, obs_mean_r2, obs_mean_r3
print(obs_mean)
ua_plev_sel_DJF = convert_DJF(obs_mean.loc["2000-12-01":"2029-02-01",50000.:100,:,:])
print(ua_plev_sel_DJF)
del obs_mean

#计算纬度的加权平均
weights = np.cos(np.deg2rad(ua_sel_DJF.lat))
ua_historical_word_mean = ua_sel_DJF.weighted(weights).mean(dim=["lat", "lon"])
#ua_historical_word_mean = ua_sel_DJF.mean(dim=["lat","lon"])#weight_word(ua_sel_DJF)
print(ua_historical_word_mean)

ua_historical_word_mean_plev = ua_plev_sel_DJF.mean(dim=["lon"])#weight_word(obs_mean)
print(ua_historical_word_mean_plev)

#QBO指数标准化
ua_historical_word_mean_normal = Z_score_normalize(ua_historical_word_mean)
print(ua_historical_word_mean_normal)

#选折挑选出的年份
times = ua_historical_word_mean_normal["time"]
mask_postivate  = ua_historical_word_mean_normal>= 0.5
mask_negative  = ua_historical_word_mean_normal<= -0.5
true_indices_postivate = np.where(mask_postivate.values)[0]  # 获取 True 的索引
true_years_postivate = times[true_indices_postivate]  # 对应的年份
true_indices_negative = np.where(mask_negative.values)[0]  # 获取 True 的索引
true_years_negative = times[true_indices_negative]  # 对应的年份
print(true_years_postivate)
print(true_years_negative)

#选取EQBO的数据
WQBO_datesel = ua_historical_word_mean_plev.loc[true_years_postivate,:,:]
print(WQBO_datesel)
#选取WQBO的数据
EQBO_datesel = ua_historical_word_mean_plev.loc[true_years_negative,:,:]
print(EQBO_datesel)

##选取数据
print(true_years_negative[0:17])

#Conduct synthesis analysis
EQBO_datesel_ano = EQBO_datesel.mean(dim=["time"])
WQBO_datesel_ano = WQBO_datesel.mean(dim=["time"])
print(EQBO_datesel_ano)
EQBO_WQBO_diff = EQBO_datesel_ano-WQBO_datesel_ano
_,p_nino = ttest_ind(EQBO_datesel,WQBO_datesel,equal_var=False)
p_nino = xr.DataArray(p_nino, coords=[EQBO_WQBO_diff["plev"],EQBO_WQBO_diff["lat"]], dims=["plev","lat"])

#############################################################################
#E3SMv3 PDINT bootstrap
#############################################################################
#进行bootstrap重采样
iter = 1000
EQQBO_WQBO_diff_bootstrap = np.empty((iter,len(ua_historical_word_mean_plev["plev"]),len(ua_historical_word_mean_plev["lat"])))
EQQBO_WQBO_diff_bootstrap = xr.DataArray(EQQBO_WQBO_diff_bootstrap, coords=[np.arange(1,1001,1),ua_historical_word_mean_plev["plev"],\
                                ua_historical_word_mean_plev["lat"]], dims=['ensemble',"plev","lat"])
for mm in range(iter):
    true_years_postivate_sel = np.random.choice(true_years_postivate, size=len(true_years_postivate))
    #选取EQBO的数据
    WQBO_datesel_bootstrap = ua_historical_word_mean_plev.loc[true_years_postivate_sel,:,:]
    print(WQBO_datesel_bootstrap)
    #选取WQBO的数据
    true_years_negative_sel = np.random.choice(true_years_negative, size=len(true_years_postivate))
    EQBO_datesel_bootstrap = ua_historical_word_mean_plev.loc[true_years_negative_sel,:,:]
    print(EQBO_datesel_bootstrap)
    ##选取数据
    # print(true_years_negative[0:17])
    #作合成分析
    EQBO_datesel_ano_bootstrap = EQBO_datesel_bootstrap.mean(dim=["time"])
    WQBO_datesel_ano_bootstrap = WQBO_datesel_bootstrap.mean(dim=["time"])
    print(EQBO_datesel_ano_bootstrap)
    EQQBO_WQBO_diff_bootstrap[mm,:,:] = EQBO_datesel_ano_bootstrap-WQBO_datesel_ano_bootstrap
    del true_years_postivate_sel,WQBO_datesel_bootstrap,true_years_negative_sel,EQBO_datesel_bootstrap
    del EQBO_datesel_ano_bootstrap,WQBO_datesel_ano_bootstrap
    # _,p_nino = ttest_ind(EQBO_datesel,WQBO_datesel,equal_var=False)

#保存数据
dir_out = "/home/users/zhangjk/code/2026-10/QBO/E3SMv3/out/"
hist= xr.Dataset({'QBO': EQBO_WQBO_diff,"QBO_P":p_nino,"EQQBO_WQBO_diff":EQQBO_WQBO_diff_bootstrap})
hist.to_netcdf(dir_out+'E3SMv3_ua_PDNINT_WQBO_EQBO_diff_bootstrap_r1.nc')
# exit()

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
# ax1.set_ylim(200, 5)
ax1.xaxis.set_major_formatter (LatitudeFormatter())
ax1.set_xticks(np.arange(-90,120,30))
ax1.set_xlim(-90,90)
ax1.set_ylim(300,5)
for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
ax1.set_ylabel('Plev(hPa)',fontsize=1.8,labelpad=0.7)
ax1.set_xlabel('Lat',fontsize=1.8,labelpad=0.4)

#绘图1
lat = EQBO_datesel_ano["lat"]
level = EQBO_datesel_ano["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax1.contourf(date2d,lev2d,EQBO_datesel_ano,extend='both',cmap=cmaps.MPL_bwr,levels=np.arange(-60.0,60.1,5))
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
cbar = fig.colorbar(c1,cax,ticks=np.arange(-60.0,60.1,20),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$U$"+r" (m/s)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax1.set_title('(a) EQBO(E3SMv3)',fontsize=1.6,loc='center',pad=1.1,weight='bold')

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
ax2.set_ylim(300,5)
for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
ax2.set_ylabel('Plev(hPa)',fontsize=1.8,labelpad=0.7)
ax2.set_xlabel('Lat',fontsize=1.8,labelpad=0.4)

#绘图2
lat = WQBO_datesel_ano["lat"]
level = WQBO_datesel_ano["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax2.contourf(date2d,lev2d,WQBO_datesel_ano,extend='both',cmap=cmaps.MPL_bwr,levels=np.arange(-60.0,60.1,5))
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
cbar = fig.colorbar(c1,cax,ticks=np.arange(-60.0,60.1,20),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$U$"+r" (m/s)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax2.set_title('(b) WQBO(E3SMv3)',fontsize=1.6,loc='center',pad=1.1,weight='bold')

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
lat = EQBO_WQBO_diff["lat"]
level = EQBO_WQBO_diff["plev"]/100
date2d,  lev2d= np.meshgrid(lat, level)
c1 = ax3.contourf(date2d,lev2d,EQBO_WQBO_diff,extend='both',cmap=cmaps.MPL_bwr,levels=np.arange(-20.0,20.1,2.0))
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
ax3.set_position([chartBox.x0+0.02, chartBox.y0+0.15,chartBox.width*0.95,chartBox.height*0.85])
#添加色标
cax = add_down_cax(ax3, pad=0.13, height=0.021)
cbar = fig.colorbar(c1,cax,ticks=np.arange(-20.0,20.1,10.0),fraction=0.55,pad=0.1,shrink=0.45,orientation="horizontal")  #aspect代表长宽比
cbar.ax.tick_params(size=0.6,width=0.2,pad=0.8,labelsize=1.3)
cbar.set_label(r"$U$"+r" (m/s)",fontsize=1.2,labelpad=0.1)
cbar.outline.set_linewidth(0.2)
ax3.set_title('PDNINT(EQBO-WQBO)',fontsize=1.6,loc='center',pad=1.1,weight='bold')

ax1.set_rasterized(True)
ax2.set_rasterized(True)
ax3.set_rasterized(True)
plt.savefig("ua_EQBO_WQBO_PDNINT_diff_r1.png",dpi=8000,format="png",facecolor="white")
plt.show()