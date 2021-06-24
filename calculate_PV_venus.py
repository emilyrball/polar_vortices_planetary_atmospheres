'''
Calculates potential vorticity on isobaric levels from Isca data. Optionally
interpolates the data to isentropic coordinates.
'''

from multiprocessing import Pool, cpu_count
import numpy as np
import xarray as xr
import os, sys
import PVmodule as PV
import glob
import matplotlib.pyplot as plt

def netcdf_prep(ds):
    '''
    Appends longitude 360 to file and reduces file to only variables necessary
    for PV calculation. Also converts pressure to Pa.
    '''
    ens_list = []
    tmp1 = ds.sel(lon=0.)
    tmp1 = tmp1.assign_coords({'lon':359.9999})
    ens_list.append(ds)
    ens_list.append(tmp1)

    d = xr.concat(ens_list, dim='lon')
    d = d.astype('float32')
    d = d[["ucomp", "vcomp", "temp", "mars_solar_long"]]
    # pressure is in hPa, must be in Pa for calculations
    d["pfull"] = d.pfull*100
    return d

def get_ls(d):
    '''
    Extracts solar longitude from dataset.
    '''
    Ls = d.mars_solar_long.sel(lon=0).squeeze()
    return Ls

def wind_prep(d):
    '''
    Reformats uwind, vwind and temperature to be in correct shape for
    Windspharm calculations.
    '''
    uwnd = d.uwind.transpose('lat','lon','plev','time')
    vwnd = d.vwind.transpose('lat','lon','plev','time')
    tmp = d.temp.transpose('lat','lon','plev','time')

    return uwnd, vwnd, tmp

def save_PV_isobaric(d, outpath):
    '''
    Saves potential vorticity on isobaric levels to ``outpath``.
    '''

    #print('Saving PV on isobaric levels to '+outpath)
    d["plev"] = d.plev/100  # data back to hPa
    d.to_netcdf(outpath+'_PV.nc')
    d["plev"] = d.plev*100  # back to Pa for isentropic interpolation

    return d


def calculate_PV_all(runs, **kwargs):
    
    ds = xr.open_mfdataset(f+'.nc', decode_times=False,
                           concat_dim='time', combine='nested',
                           chunks={'time':'auto'})
    d = netcdf_prep(ds)
    Ls = get_ls(d)
    theta = PV.potential_temperature(d.pfull, d.temp,
                                     kappa = kappa, p0 = p0)
    uwnd_trans,vwnd_trans,tmp_trans = wind_prep(d)
    PV_iso = PV.potential_vorticity_baroclinic(uwnd_trans, vwnd_trans,
              theta, 'pfull', omega = omega, g = g, rsphere = rsphere)
    PV_iso = PV_iso.transpose('time','pfull','lat','lon')
    d["PV"] = PV_iso
    d = save_PV_isobaric(d, outpath)

def ext_exner_function(levs, **kwargs):
    '''
    Calculates the extended Exner function.

    Input
    -----
    levs  : pressure levels, array-like
    p0    : reference pressure in Pascals, optional. Default: 610. Pa
    '''
    p0 = kwargs.pop('p0', 100000.)
    T0 = kwargs.pop('T0', 350.)
    A = 2.5223
    B = 0.77101e-2
    C = -0.03981e-5
    Cp0 = A + B*T0 + C*T0**2

    kappa = 1/Cp0

    return (p0/levs)**kappa

def Cp(T):
    A = 2.5223
    B = 0.77101e-2
    C = -0.3981e-5
    R = 188.92
    #R = 8.3143
    return R*(A + B*T + C*(T**2))

def calculate_tau(t, t0):
    A = 2.5223
    B = 0.77101e-2
    C = -0.3981e-5
    return t0 * np.exp(A/Cp(t0) * np.log(t/t0 *np.exp(B/A*(t-t0) \
                         + C/(2*A)*(t**2-t0**2))))

def ext_potential_temp(p, t, **kwargs):
    '''
    Calculates extended potential temperature theta according to
    Garate-Lopez et al. (2016)

    Input
    -----
    tmp   : temperature, array-like
    levs  : pressure levels, array-like
    p0    : reference pressure in Pascals, optional. Default: 610. Pa
    '''
    p0 = kwargs.pop('p0', 100000.)
    t0 = kwargs.pop('t0', 350.)

    #R = 8.33143
    R = 188.92
    return calculate_tau(t,t0) * (p/p0)**(-R/Cp(t0))



if __name__ == "__main__":

    #d = [xr.open_dataset(i) for i in glob.glob('link-to-anthro/' + \
    #                            'vex-analysis/*.nc')]
    #ds = [d[i].assign_coords(time = i) for i in range(len(d))]
    # Venus-specific
    d = xr.open_dataset('link-to-anthro/vex-analysis/venus.nc')
    
    xr.plot.plot(d.t.sel(time=d.time[0]).mean(dim="lon",skipna=True),
                    )#levels = [150,200,250,300])
    #plt.xlim([-60,-90])
    plt.ylim([1,0])
    plt.savefig('Figs/venus_temp_sigma.png')
    plt.clf()
    theta0 = 350. # reference temperature
    p0 = 100000. # reference pressure
    omega = -2.9923e-7 # planetary rotation rate
    g = 8.81 # gravitational acceleration
    rsphere = 6.0518e6 # mean planetary radius

    outpath = 'link-to-anthro/vex-analysis/venus_isobaric'
    pfull = d.lev * d.ps.squeeze()
    pfull = pfull.transpose("time", "lev", "lat", "lon")
    theta = ext_potential_temp(pfull, d.t, p0 = p0, t0 = theta0)
    
    #xr.plot.contour(theta.mean(dim="lon",skipna=True).mean(dim="time",skipna=True),
    #                levels = [300,350,400,450,500,550,600,650,700,750,850])
    #plt.xlim([-60,-90])
    #plt.ylim([1,0])
    xr.plot.plot(theta.sel(time=theta.time[0]).mean(dim="lon",skipna=True),
                    )#levels = [150,200,250,300])
    #plt.yscale('log')
    plt.ylim([1,0])
    plt.savefig('Figs/test_venus_sigma.png')
    plt.clf()


    d = d[["t", "u", "v"]]#, "Q", "CW"]]
    d = d.squeeze().transpose("time", "lev", "lat", "lon")

    plevs = np.logspace(7,-3, 55)
    tmp, uwnd, vwnd = PV.log_interpolate_1d(plevs, pfull.values, d.t.values,
                    d.u.values, d.v.values, axis = 1)

    d_iso = xr.Dataset({
        "temp"  : (("time", "plev", "lat", "lon"), tmp),
        "uwind" : (("time", "plev", "lat", "lon"), uwnd),
        "vwind" : (("time", "plev", "lat", "lon"), vwnd),
        },
        coords = {
            "time" : d.time,
            "plev" : plevs,
            "lat"  : d.lat,
            "lon"  : d.lon,
        }
    )
    theta = ext_potential_temp(d_iso.plev, d_iso.temp, p0 = p0, t0 = theta0)
    
    xr.plot.plot(theta.mean(dim="lon",skipna=True).mean(dim="time" \
                ,skipna=True).where(theta.plev > 100,drop = True).where(theta.plev < 150000),
                    )#levels = [300,350,400,450,500,550,600,650,700,750,850])
    #plt.xlim([-60,-90])
    plt.ylim([150000,100])
    plt.yscale('log')
    plt.savefig('Figs/test_venus.png')
    plt.clf()

    xr.plot.plot(d_iso.uwind.mean(dim="lon",skipna=True).mean(dim="time" \
                ,skipna=True).where(theta.plev > 100,drop = True).where(theta.plev < 150000),
                    )#levels = [300,350,400,450,500,550,600,650,700,750,850])
    #plt.xlim([-60,-90])
    plt.ylim([150000,100])
    plt.yscale('log')
    plt.savefig('Figs/test_winds.png')
    plt.clf()

    xr.plot.plot(d_iso.temp.mean(dim="lon",skipna=True).mean(dim="time" \
                ,skipna=True).where(theta.plev > 100,drop = True).where(theta.plev < 150000),
                    )#levels = [300,350,400,450,500,550,600,650,700,750,850])
    #plt.xlim([-60,-90])
    plt.ylim([150000,100])
    plt.yscale('log')
    plt.savefig('Figs/test_temp.png')

    uwnd_trans,vwnd_trans,tmp_trans = wind_prep(d_iso)
    PV_iso = PV.potential_vorticity_baroclinic(uwnd_trans, vwnd_trans,
              theta, 'plev', omega = omega, g = g, rsphere = rsphere)
    PV_iso = PV_iso.transpose('time','plev','lat','lon')
    theta = theta.transpose('time','plev','lat','lon')
    d_iso["PV"] = PV_iso
    d_iso["theta"] = theta
    #d_iso = d_iso.mean(time="time")
    d_iso = save_PV_isobaric(d_iso, outpath)
    
    