# Python script to plot PV cross-section for Earth, Mars and Titan
# Output is a plot showing latitude-pressure cross-section of PV for
# winter polar vortices of different planets.
# Written by Emily Ball, 02/11/2020
# Tested on Anthropocene.

# NOTES and known problems
# Data for Earth are from ERA5, for Mars from OpenMARS and for
# Titan from Sharkey et al. 2020.

# Updates
# 11/11/2019 Initial upload to GitHub EB

# Begin script ========================================================================================================

# PV cross-section for Titan, data from Sharkey et al 2020

import numpy as np

import netCDF4
import pandas as pd
import xarray as xr

import colorcet as cc
import os, sys
import PVmodule as PVmod
import string

from cartopy import crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import (cm, colors)

def make_colourmap(vmin, vmax, step, **kwargs):
    '''
    Makes a colormap from ``vmin`` (inclusive) to ``vmax`` (exclusive) with
    boundaries incremented by ``step``. Optionally includes choice of color and
    to extend the colormap.
    '''
    col = kwargs.pop('col', 'viridis')
    extend = kwargs.pop('extend', 'both')

    boundaries = list(np.arange(vmin, vmax, step))

    if extend == 'both':
        cmap_new = cm.get_cmap(col, len(boundaries) + 1)
        colours = list(cmap_new(np.arange(len(boundaries) + 1)))
        cmap = colors.ListedColormap(colours[1:-1],"")
        cmap.set_over(colours[-1])
        cmap.set_under(colours[0])

    elif extend == 'max':
        cmap_new = cm.get_cmap(col, len(boundaries))
        colours = list(cmap_new(np.arange(len(boundaries))))
        cmap = colors.ListedColormap(colours[:-1],"")
        cmap.set_over(colours[-1])

    elif extend == 'min':
        cmap_new = cm.get_cmap(col, len(boundaries))
        colours = list(cmap_new(np.arange(len(boundaries))))
        cmap = colors.ListedColormap(colours[1:],"")
        cmap.set_under(colours[0])

    norm = colors.BoundaryNorm(boundaries, ncolors = len(boundaries) - 1,
                               clip = False)

    return boundaries, cmap_new, colours, cmap, norm

def laitMars(PV,theta,theta0, **kwargs):
    r"""Perform Lait scaling of PV
    kwargs
    ------
    kappa: R/c_p, optional, defaults to 0.25.
    """
    kappa = kwargs.pop('kappa', 0.25)
    ret = PV*(theta/theta0)**(-(1+1/kappa))

    return ret


def laitTitan(PV,theta,theta0, **kwargs):
    r"""Perform Lait scaling of PV
    kwargs
    ------
    kappa: R/c_p, optional, defaults to 0.25.
    """
    kappa = kwargs.pop('kappa', 0.25)
    ret = PV*(theta/theta0)**(-7/2)#(1+1/kappa))

    return ret

def fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

class nf(float):
    def __repr__(self):
        s = f'{self:.1f}'
        return f'{self:.0f}' if s[-1] == '0' else s

def plot_setup(axs, ymin, ymax, title, norm, cmap, boundaries):
    axs.set_yscale('log')#, subsy = [])
    axs.set_xlim([0,90])
    axs.set_ylim([ymax,ymin])
    axs.minorticks_off()
    axs.yaxis.set_major_formatter(ticker.ScalarFormatter())
    axs.set_title(title,fontsize=25,weight='bold', y=1.04)

    cb = fig.colorbar(cm.ScalarMappable(norm,cmap),ax=axs,extend='max',
                      orientation='horizontal',
                      ticks=boundaries[slice(None,None,2)])
    cb.set_label(label='Lait PV (10$^{-5}$ K m$^2$ kg$^{-1}$ s$^{-1}$)',
                 fontsize=18)

    cb.ax.tick_params(labelsize=18)

    axs.tick_params(labelsize=18)
    axs.set_xlabel('latitude ($^{\circ}$N)',fontsize=20)

    return cb

def scale(val, src, dst):
        """
        Scale the given value from the scale of src to the scale of dst.
        """
        return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]



if __name__ == "__main__":


    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'


    figpath = 'Thesis/'
    fig, ax2 = plt.subplots(nrows=1,ncols=3,figsize=(26, 8))

    boundaries2, _, _, cmap2, norm2 = make_colourmap(-1, 10.1, 0.5,
                                            col = 'OrRd', extend = 'max')

    boundaries1, _, _, cmap1, norm1 = make_colourmap(-5, 65.1, 2.5,
                                            col = 'OrRd', extend = 'max')

    boundaries0, _, _, cmap0, norm0 = make_colourmap(-0.5, 4.6, 0.25,
                                            col = 'OrRd', extend = 'max')


    for i, ax in enumerate(fig.axes):
        ax.text(-0.05, 1.05, string.ascii_lowercase[i], transform=ax.transAxes, 
                size=20, weight='bold')


    ax2[2].set_ylabel('potential temperature (K)',fontsize=18)
    ax2[2].set_xlabel('latitude ($^{\circ}$N)',fontsize=20)

    ax2[0].set_ylabel('pressure (hPa)',fontsize=20)
    


    plt.subplots_adjust(wspace=0.15)
    plt.minorticks_off()

    # Titan data plotted on potential temperature surfaces
    axs = ax2[2].twinx()  # instantiate a second axes that shares the same x-axis
    axs.yaxis.set_label_position('left')
    axs.yaxis.tick_left()



    ax2[2].yaxis.set_label_position('right')
    ax2[2].yaxis.tick_right()
    ax2[2].set_ylim([-20, 1250])
    ax2[2].tick_params(labelsize=18)


    cb0 = plot_setup(ax2[0], 1, 1000, 'Earth', norm0, cmap0, boundaries0)
    cb1 = plot_setup(ax2[1], 0.005, 6.1, 'Mars', norm1, cmap1, boundaries1)
    cb2 = plot_setup(axs, 0.005, 80, 'Titan', norm2, cmap2, boundaries2)

    ax2[1].set_yticks([6,1,0.1,0.01])
    axs.set_yticks([35,20,10,5,0.3,0.01])
    axs.set_yticklabels([1500,100,10,1,0.1,0.01])
    ax2[2].set_yticks([200,400,600,800,1000,1200])

    d = .005  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=axs.transAxes, color='k', clip_on=False)
    
    axs.plot((0, +d), (0, + 2*d), **kwargs)  # bottom-left diagonal
    axs.plot((d, -d), (2*d, 4*d), **kwargs)
    axs.plot((-d, d), (4*d, 6*d), **kwargs)
    axs.plot((d, -d), (6*d, 8*d), **kwargs)
    axs.plot((-d, d), (8*d, 10*d), **kwargs)
    axs.plot((d, 0), (10*d, 12*d), **kwargs)
    
    axs.plot((-d, +d), (0.06 - d, 0.06 + d), **kwargs)  # bottom-right diagonal

    axs.plot((1, 1+d), (0, +2*d), **kwargs)  # bottom-left diagonal
    axs.plot((1+d, 1-d), (2*d, 4*d), **kwargs)
    axs.plot((1-d, 1+d), (4*d, 6*d), **kwargs)
    axs.plot((1+d, 1-d), (6*d, 8*d), **kwargs)
    axs.plot((1-d, 1+d), (8*d, 10*d), **kwargs)
    axs.plot((1+d, 1), (10*d, 12*d), **kwargs)
    
    axs.plot((-2*d, +2*d), (0.06 - d, 0.06 + d), **kwargs)  # bottom-right diagonal
    axs.plot((-2*d, +2*d), (- d, + d), **kwargs)  # bottom-right diagonal
    axs.plot((1-2*d, 1+2*d), (0.06 - d, 0.06 + d), **kwargs)  # bottom-right diagonal
    axs.plot((1-2*d, 1+2*d), ( - d, + d), **kwargs)  # bottom-right diagonal



    #axs.plot((0, 0), (0,0.1), transform=axs.transAxes, color='blue',linewidth=2)

    # constants for Titan calculation of potential temperature
    theta_0 = 200.
    kappa = 0.281
    p0 = 1.e3

    # min/max solar longitudes (corr. Fig 9 of Sharkey et al. 2020)
    Lsmin = 325
    Lsmax = 345

    inpath = 'link-to-silurian/Titan_Jason/'

    plt.savefig(figpath+'PV_cross_section_Ls'+str(Lsmin)+'-'+str(Lsmax)+'.png',
                bbox_inches='tight', pad_inches=0)

    # Load Titan data =============================================================


    d1  = np.loadtxt(inpath + 'CIRS_T.txt')
    d1.transpose()



    plt.savefig(figpath+'PV_cross_section_Ls'+str(Lsmin)+'-'+str(Lsmax)+'.png',
                bbox_inches='tight', pad_inches=0)

    new_d = np.split(d1, np.where(np.diff(d1[:,1]))[0]+1)
    Ls1 = np.empty(len(new_d))

    for i in range(len(new_d)):
        Ls1[i] = new_d[i][0][1]

    new_d2 = np.split(new_d[0], np.where(np.diff(new_d[0][:,3]))[0]+1)

    p1 = np.empty(len(new_d2))
    for i in range(len(new_d2)):
        p1[i] = new_d2[i][0][3]

    lats1 = np.arange(-90.,91.,1.)

    #for i in len(new_d):
    T = np.empty((len(Ls1),len(p1),len(lats1)))

    for i in range(len(Ls1)):
        di = new_d[i]      # selects all values with Ls = Ls[i]
        new_di = np.split(di, np.where(np.diff(di[:,3]))[0]+1)  # splits into pressure levels

        m = []
        a=0
        for j in range(len(new_di)):
            while new_di[j][0][3] < p1[a]:
                T[i,a,:] = np.nan
                m.append(1)
                a += 1

            dj = new_di[j]     # selects all values with p = p[j]
            k=0
            l = []
            for ind in range(len(dj[:,2])):
                while dj[:,2][ind] > lats1[k]:
                    l.append(np.nan)
                    k += 1
                l.append(dj[:,5][ind])
                k+=1

            while len(l) < len(lats1):
                l.append(np.nan)
            T[i,j+len(m),:]=l
            a+=1


    #-----------------------------------------------------------------------------#
    d1  = np.loadtxt(inpath + 'CIRS_U.txt')
    d1.transpose()


    new_d = np.split(d1, np.where(np.diff(d1[:,1]))[0]+1)
    Ls2 = np.zeros(len(new_d))

    for i in range(len(Ls2)):
        Ls2[i] = new_d[i][0][1]

    new_d2 = np.split(new_d[0], np.where(np.diff(new_d[0][:,3]))[0]+1)

    p2 = np.zeros(len(new_d2))
    for i in range(len(p2)):
        p2[i]=new_d2[i][0][3]

    lats2 = np.arange(-90.,91.,1.)

    U = np.empty((len(Ls2),len(p2),len(lats2)))

    for i in range(len(Ls2)):
        di = new_d[i]      # selects all values with Ls = Ls[i]
        new_di = np.split(di, np.where(np.diff(di[:,3]))[0]+1)  # splits into pressure levels

        m = []
        a=0
        for j in range(len(new_di)):
            while new_di[j][0][3] < p2[a]:
                U[i,a,:] = np.nan
                m.append(1)
                a += 1

            dj = new_di[j]     # selects all values with p = p[j]
            k=0
            l = []
            for ind in range(len(dj[:,2])):
                while dj[:,2][ind] > lats2[k]:
                    l.append(np.nan)
                    k += 1
                l.append(dj[:,5][ind])
                k+=1

            while len(l) < len(lats2):
                l.append(np.nan)
            U[i,j+len(m),:]=l
            a+=1

    #-----------------------------------------------------------------------------#

    d1 = np.loadtxt(inpath + 'CIRS_PV.txt')
    d1.transpose()

    new_d = np.split(d1, np.where(np.diff(d1[:,1]))[0]+1)
    Ls3 = np.zeros(len(new_d))

    for i in range(len(Ls3)):
        Ls3[i] = new_d[i][0][1]

    new_d2 = np.split(new_d[1], np.where(np.diff(new_d[1][:,3]))[0]+1)

    theta = np.zeros(len(new_d2))
    for i in range(len(theta)):
        theta[i]=new_d2[i][0][3]

    lats3 = np.arange(-90.,91.,1.)

    #for i in len(new_d):
    prs = np.empty((len(Ls3),len(theta),len(lats3)))
    PV = np.empty((len(Ls3),len(theta),len(lats3)))

    #for i in range(len(Ls)):
    for i in range(len(Ls3)):
        di = new_d[i]      # selects all values with Ls = Ls[i]
        new_di = np.split(di, np.where(np.diff(di[:,3]))[0]+1)  # splits into theta levels

        m = []
        a=0
        for j in range(len(new_di)):
            while new_di[j][0][3] > theta[a]:
                PV[i,a,:] = np.nan
                prs[i,a,:] = np.nan
                m.append(1)
                a += 1

            dj = new_di[j]     # selects all values with p = p[j]
            k=0
            l = []
            n = []
            for ind in range(len(dj[:,2])):
                while dj[:,2][ind] > lats3[k]:
                    l.append(np.nan)
                    n.append(np.nan)
                    k += 1
                l.append(dj[:,4][ind])
                n.append(dj[:,6][ind])
                k+=1

            while len(l) < len(lats3):
                l.append(np.nan)
                n.append(np.nan)
            PV[i,j+len(m),:]=n
            prs[i,j+len(m),:]=l

            a+=1

    temp = xr.Dataset({"temp" : (("Ls", "plev", "lat"), T)},
                        coords = {"Ls": Ls1,
                                  "plev": p1*1e5,
                                  "lat": lats1})

    uwind = xr.Dataset({"uwind" : (("Ls", "plev", "lat"), U)},
                        coords = {"Ls": Ls2,
                                  "plev": p2*1e5,
                                  "lat": lats2})

    d = xr.Dataset({"PV"  : (("Ls", "theta", "lat"), PV),
                    "prs" : (("Ls", "theta", "lat"), prs*1e5)},
                        coords = {"Ls"   : Ls3,
                                  "theta": theta,
                                  "lat"  : lats3})



    thta = PVmod.potential_temperature(temp.plev, temp, kappa=kappa, p0=p0)
    lait_PV = laitTitan(d.PV, d.theta, theta_0, kappa=kappa)

    thta = thta.where(Lsmin <= thta.Ls, drop = True)
    thta = thta.where(thta.Ls <= Lsmax, drop = True)
    thta = thta.where(thta.lat >= 0, drop=True)

    lait_PV = lait_PV.where(Lsmin <= lait_PV.Ls, drop = True)
    lait_PV = lait_PV.where(lait_PV.Ls <= Lsmax, drop = True)
    lait_PV = lait_PV.where(lait_PV.lat >= 0, drop = True)

    uwnd = uwind.where(Lsmin <= uwind.Ls, drop = True)
    uwnd = uwnd.where(uwnd.Ls <= Lsmax, drop = True)
    uwnd = uwnd.where(uwnd.lat >= 0, drop = True)

    lsmin=str(int(thta.Ls.min(dim='Ls').values))
    lsmax=str(int(thta.Ls.max(dim='Ls').values))

    # Smooth over latitudes ===============================
    thta = thta.mean(dim='Ls')
    thta = thta.rolling(lat=8,center=True)
    thta = thta.mean()

    lait_PV = lait_PV.rolling(lat=8,center=True)
    lait_PV = lait_PV.mean()
    lait_PV = lait_PV.mean(dim='Ls',skipna=True)*10**5


    uwnd = uwnd.mean(dim='Ls')
    uwnd = uwnd.rolling(lat=8,center=True)
    uwnd = uwnd.mean()


    cf = ax2[2].contourf(lait_PV.lat, lait_PV.theta,
                      lait_PV.transpose('theta','lat').squeeze(),
                      levels=boundaries2+[90],norm=norm2,cmap=cmap2)

    axs.contour(thta.lat, thta.plev/100,
                thta.transpose('plev','lat').to_array().squeeze(),
                levels=[200,300,400,500,600,700,800,900,1000,1100,1200,1300],
                linestyles = '--', colors='black', linewidths=1)

    cs = axs.contour(uwnd.lat, uwnd.plev/100,
                     uwnd.transpose('plev','lat').to_array().squeeze(),
                     levels=[0,50,100,150], colors='black',linewidths=1)
    # Recast levels to new class
    cs.levels = [nf(val) for val in cs.levels]
    axs.clabel(cs, cs.levels, inline=1, fmt=fmt, fontsize=14)



    plt.savefig(figpath+'PV_cross_section_Ls'+lsmin+'-'+lsmax+'.png',
                bbox_inches='tight', pad_inches=0)


    # Martian constants for potential temperature calculation =======================================
    Lsmin = 255
    Lsmax = 285

    theta_0 = 200.
    kappa = 1/4.0
    p0 = 610.

    inpath = 'link-to-anthro/OpenMARS/Isobaric/'


    d = xr.open_mfdataset(inpath + '*mars_my*', decode_times=False, concat_dim='time',
                           combine='nested',chunks={'time':'auto'})

    d = d.where(Lsmin <= d.Ls, drop = True)
    d = d.where(d.Ls <= Lsmax, drop = True)

    lait = laitMars(d.PV, d.theta, theta_0, kappa=kappa)

    pv = lait.mean(dim='time').mean(dim='lon') *10**5

    t = d.theta.mean(dim='time').mean(dim='lon')

    u = d.uwnd.mean(dim='time').mean(dim='lon')

    p = d.plev/100
    lat = d.lat

    print('Plotting Mars')




    cf = ax2[1].contourf(lat, p, pv.transpose('plev','lat'),
                      levels=boundaries1+[90],norm=norm1,cmap=cmap1)
    ax2[1].contour(lat, p, t.transpose('plev','lat'),
                levels=[200,300,400,500,600,700,800,900,1000,1100],
                linestyles = '--', colors='black', linewidths=1)
    cs = ax2[1].contour(lat, p, u.transpose('plev','lat'),
                    levels=[0,50,100,150], colors='black',linewidths=1)
    # Recast levels to new class
    cs.levels = [nf(val) for val in cs.levels]

    # Label levels with specially formatted floats
    if plt.rcParams["text.usetex"]:
        fmt = r'%r \%'
    else:
        fmt = '%r'

    ax2[1].clabel(cs, cs.levels, inline=1, fmt=fmt, fontsize=14)

    plt.savefig(figpath+'PV.png',
                bbox_inches='tight', pad_inches=0)




    print('Beginning ERA5')

    theta_0 = 475.
    kappa = 0.286
    p0 = 101325.

    inpath = '/export/anthropocene/array-01/xz19136/era5data/'
    figpath = 'Thesis/'

    d1 = xr.open_mfdataset(inpath + 'PV_mean.nc', concat_dim='time',
                           combine='nested', chunks={'latitude':'auto',
                                                     'longitude':'auto'})

    d2 = xr.open_mfdataset(inpath + 'U_mean.nc', concat_dim='time',
                           combine='nested', chunks={#'time' : 'auto',
                                                     'latitude':'auto',
                                                     'longitude':'auto'})

    d3 = xr.open_mfdataset(inpath + 'T_mean.nc', concat_dim='time',
                           combine='nested', chunks={#'time' : 'auto',
                                                     'latitude':'auto',
                                                     'longitude':'auto'})

    d11 = xr.open_mfdataset(inpath + 'PV_mean_2015-2017.nc', concat_dim='time',
                           combine='nested', chunks={'latitude':'auto',
                                                     'longitude':'auto'})

    d21 = xr.open_mfdataset(inpath + 'U_mean_2015-2017.nc', concat_dim='time',
                           combine='nested', chunks={#'time' : 'auto',
                                                     'latitude':'auto',
                                                     'longitude':'auto'})

    d31 = xr.open_mfdataset(inpath + 'T_mean_2015-2017.nc', concat_dim='time',
                           combine='nested', chunks={#'time' : 'auto',
                                                     'latitude':'auto',
                                                     'longitude':'auto'})


    d1 = d1.transpose('level', 'latitude', 'longitude', 'time')
    d2 = d2.transpose('level', 'latitude', 'longitude', 'time')
    d3 = d3.transpose('level', 'latitude', 'longitude', 'time')

    d11 = d11.transpose('level', 'latitude', 'longitude', 'time')
    d21 = d21.transpose('level', 'latitude', 'longitude', 'time')
    d31 = d31.transpose('level', 'latitude', 'longitude', 'time')


    ens1 = [d1, d11]
    ens2 = [d2, d21]
    ens3 = [d3, d31]

    d1 = xr.concat(ens1, dim='time')
    d2 = xr.concat(ens2, dim='time')
    d3 = xr.concat(ens3, dim='time')

    d1 = d1.mean(dim='time')
    d2 = d2.mean(dim='time')
    d3 = d3.mean(dim='time')


    #d1["level"] = d1.level*100
    d2["level"] = d2.level*100             # convert hPa to Pa
    d3["level"] = d3.level*100

    R = 287.04
    cp = 1004.6

    print('Calculating potential temperature...')
    thta = PVmod.potential_temperature(d3.level, d3.t,
                            p0 = 101325, kappa = R/cp)


    lait = d1.__xarray_dataarray_variable__*(thta/theta_0)**(-(1+cp/R))

    pv = lait.mean(dim='longitude') *10**5
    pv = pv.squeeze()

    t = thta.mean(dim='longitude')

    u = d2.mean(dim='longitude')

    p = d1.level/100
    lat = d1.latitude



    cf = ax2[0].contourf(lat, p, pv.transpose('level','latitude'),
                      levels=boundaries0+[90],norm=norm0,cmap=cmap0)
    ax2[0].contour(lat, p, t.transpose('level','latitude'),
                levels=[200,300,400,500,600,700,800,900,1000,1100,1200,1300,
                        1400,1500,1600],
                linestyles = '--', colors='black', linewidths=1)
    cs = ax2[0].contour(lat, p, u.transpose('level','latitude').to_array().squeeze(),
                    levels=[0,10,20,30], colors='black',linewidths=1)
    # Recast levels to new class
    cs.levels = [nf(val) for val in cs.levels]
    # Label levels with specially formatted floats

    ax2[0].clabel(cs, cs.levels, inline=1, fmt=fmt, fontsize=14)


    plt.savefig(figpath+'PV_cross_section_all.png',
                bbox_inches='tight', pad_inches=0)
