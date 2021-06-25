# polar_vortices_planetary_atmos
Enclosed are the (Python) scripts used to plot Figures 1, 5, and 8 in Mitchell et al. (2021) Polar vortices in planetary atmospheres.

## planetary_polar_vortices_cross_section.py
Plots Figure 1. Data used for Venus were provided by Norihiko Sugimoto, and the relevant paper is Sugimoto et al. (2019) Impact of Data Assimilation on THermal Tides in the Case of Venus Express Wind Observation (https://doi.org/10.1029/2019GL082700). Data used for Earth are from ERA5 (https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form) and for Mars fom OpenMARS (https://ordo.open.ac.uk/collections/OpenMARS_database/4278950/1). The data used to plot the polar vortices on Titan were provided to us by Jason Sharkey, and the relevant paper is Sharkey et al. (2021) Potential Vorticity Structure in Titan's Polar Vortices from Cassini CIRS Observations (https://doi.org/10.1016/j.icarus.2020.114030).

## planetary_polar_vortices_mars_map.py
Plots Figure 5. Data used are from the OpenMARS reanalysis (https://ordo.open.ac.uk/collections/OpenMARS_database/4278950/1).

## planetary_polar_vortices_trappist_1e.py
Plots Figure 8. Data is taken from the HAB1 Unified Model experiment from Fauchez et al. (2020) TRAPPIST-1 Habitable Atmosphere Intercomparison (THAI): motivations and protocol version 1.0 (https://doi.org/10.5194/gmd-13-707-2020).

## PVmodule.py
Potential vorticity is calculated using the script PVmodule.py, along with other helpful functions.

## analysis_functions.py
Various useful functions, including Lait-scaling for potential vorticity.

## calculate_PV_venus.py
Calculates PV from AFES-Venus reanalysis data.

