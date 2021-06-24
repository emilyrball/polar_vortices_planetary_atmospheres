# Trappist-1e PV cross-section

import numpy as np
import xarray as xr
import os, sys
import glob

import colorcet as cc

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors)
import matplotlib.path as mpath

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def laitscale(PV,theta,theta0, **kwargs):
    r"""Perform Lait scaling of PV
    kwargs
    ------
    kappa: R/c_p, optional, defaults to 0.25.
    """
    kappa = kwargs.pop('kappa', 0.25)
    ret = PV*(theta/theta0)**(-(1+1/kappa))

    return ret

if __name__ == "__main__":

    thetalevs=[200., 250., 300., 350., 400., 450., 500., 550., 600., 650., 700., 750., 800., 850., 900., 950.]

    theta_0 = 200.
    kappa = 0.286
    p0 = 1.e5

    plev = 5

    inpath = '/export/silurian/array-01/xz19136/'

    figpath = 'trappist-1e_figs/'

    d = xr.open_mfdataset(inpath + '*trappist1e*.nc*', decode_times=False, concat_dim='time',
                           combine='nested',chunks={'time':'auto'})

    d["air_pressure"] = d.air_pressure/100

    ens_list = []
    tmp1 = d.sel(longitude=-178.8,method='nearest')
    tmp1 = tmp1.assign_coords({'longitude':181.2})
    ens_list.append(d)
    ens_list.append(tmp1)
    d = xr.concat(ens_list, dim='longitude')
    d = d.where(d.latitude>=0,drop=True)
    d = d.sel(air_pressure=plev, method='nearest').mean(dim='time')

    d.to_netcdf('link-to-anthro/Mitchell_etal_2021/data_trappist-1e_map.nc')

    # plot set-up
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)



    fig, axs = plt.subplots(nrows=1,ncols=1, figsize = (8,8),
                            subplot_kw = {'projection':ccrs.NorthPolarStereo()})


    boundaries = list(np.arange(186.75,189.5,0.25))
    cmap_viridis = cm.get_cmap('cet_coolwarm',len(boundaries)+1)
    colours = list(cmap_viridis(np.arange(len(boundaries)+1)))

    cmap = colors.ListedColormap(colours[1:-1],"")
    cmap.set_over(colours[-1])
    cmap.set_under(colours[0])

    norm = colors.BoundaryNorm(boundaries, ncolors=len(boundaries)-1,clip=False)

    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    #gl = axs.gridlines(draw_labels=True)
    gl = axs.gridlines(crs=ccrs.PlateCarree(),linewidth=1,
                          linestyle='--',color='black',alpha=0.3,)
    axs.set_boundary(circle, transform=axs.transAxes)
    gl.xlocator = ticker.FixedLocator([-180,-120,-60,0,60,120,180])
    gl.ylocator = ticker.FixedLocator([np.max(d.latitude),60,30,0])


    t = d.air_temperature


    c0 = axs.contourf(t.longitude,t.latitude,t,
                    cmap=cmap,transform=ccrs.PlateCarree(),
                    norm=norm,levels=[-25]+boundaries+[300])

    cb = fig.colorbar(cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=axs, label='temperature (K)', extend='both',
                 #orientation='horizontal',
                 shrink=0.7, pad=.03,ticks=boundaries[slice(1,None,2)])

    cb.set_label(label='Temperature [K]',
                 fontsize=15)
    cb.ax.tick_params(labelsize=15)

    d = d.chunk({'latitude':36,'longitude':36})

    u = d.x_wind


    u = u.sel(latitude = list(np.arange(10,75,10)),method='nearest')
    u = u.sel(longitude = list(np.arange(-180,180,30)),method='nearest')

    v = d.y_wind
    

    v = v.sel(latitude = list(np.arange(10,75,10)),method='nearest')
    v = v.sel(longitude = list(np.arange(-180,180,30)),method='nearest')

    Q = axs.quiver(u.longitude.values, u.latitude.values,u.values,v.values,
                    transform=ccrs.PlateCarree(),color='black')

    axs.quiverkey(Q, 0.9, 0.9, 40, r'40 ms$^{-1}$',
                  fontproperties = {'size':14})

    plt.savefig(figpath+'average_temp_'+str(plev)+'hPa.png', bbox_inches='tight',
                pad_inches=0.05)