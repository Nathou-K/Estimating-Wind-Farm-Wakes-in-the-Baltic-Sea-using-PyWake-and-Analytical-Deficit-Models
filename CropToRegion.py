import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, Transformer
import warnings

warnings.filterwarnings("ignore", message="Input array is not C_CONTIGUOUS.*")

latituderef = 55.5
longituderef = -8

proj_string_lcc = f"+proj=lcc +lat_0={latituderef} +lon_0={longituderef} +lat_1=55.5 +x_0=1527525.1216817154 +y_0=1588681.226267193 +R=6371229.0 +units=m +no_defs"
proj_lcc = Proj(proj_string_lcc)

proj_latlon = Proj(proj="latlong", datum="WGS84")

# DEFINING A "RECTANGLE" LATITUDE/LONGITUDE REGION
###KRIEGERS FLAK###
#latmin = 54.5
#latmax = 55.8
#lonmin = 12.2
#lonmax = 13.5
###EAST ANGLIA###
#latmin = 51.7
#latmax = 52.8
#lonmin = 1.8
#lonmax = 3.3
###HORNS REV 3###
latmin = 55.2
latmax = 56.2
lonmin = 7.
lonmax = 8.3

lat_reg = np.arange(latmin, latmax, 0.005) # Resolution: 0.005°
lon_reg = np.arange(lonmin, lonmax, 0.005)
lat_reg, lon_reg = np.meshgrid(lat_reg, lon_reg)

transformer_lccTolatlon = Transformer.from_proj(proj_lcc, proj_latlon) #

data_folder = '/scratch/project_2010671/DMI-HAMOWIOWF_temp/detail/'
work_dir = 'code/'
output_dir = '/scratch/project_2010671/DMI-HAMOWIOWF_temp/HornsRev/'

month = '202604' # a commodity
for j in range(1, 32):
    for i in range(0, 8): # We were using 8 timestamps per day every 3 hours from midnight
        h = 3*i
        
        fname = data_folder + 'withoutOWF/' + month + str(j).zfill(2) + str(h).zfill(2) + '.nc'
        hamo_noWF = xr.open_dataset(fname)
        
        fname = data_folder + 'withOWF/' + month + str(j).zfill(2) + str(h).zfill(2) + '.nc'
        hamo_WF = xr.open_dataset(fname)
        
        hamo_WF = hamo_WF.squeeze()
        hamo_noWF = hamo_noWF.squeeze()
        
        y, x = np.meshgrid(hamo_WF['y'].values, hamo_WF['x'].values)
        #print(x)
        #print(x.shape)
        lon_re, lat_re = transformer_lccTolatlon.transform(x, y)
        #print(lat_re.shape)
        hamo_WF = hamo_WF.assign_coords({'lat':(('x', 'y'), np.ascontiguousarray(lat_re)), 'lon':(('x', 'y'), np.ascontiguousarray(lon_re))})
        
        y, x = np.meshgrid(hamo_noWF['y'].values, hamo_noWF['x'].values)
        lon_re, lat_re = transformer_lccTolatlon.transform(x, y)
        hamo_noWF = hamo_noWF.assign_coords({'lat':(('x', 'y'), np.ascontiguousarray(lat_re)), 'lon':(('x', 'y'), np.ascontiguousarray(lon_re))})
        
        ds_reg=xr.Dataset({"lat":xr.DataArray(np.ascontiguousarray(lat_reg), dims=('x','y')),"lon":xr.DataArray(np.ascontiguousarray(lon_reg), dims=('x','y'))})
        
        ds_reg = ds_reg.assign_coords({'lat':(('x', 'y'), np.ascontiguousarray(lat_reg)), 'lon':(('x', 'y'), np.ascontiguousarray(lon_reg))})
        
        hamo_WF = hamo_WF.astype(np.float32, order='F', casting='unsafe')
        hamo_noWF = hamo_noWF.astype(np.float32, order='F', casting='unsafe')
        ds_reg = ds_reg.astype(np.float32, order='F', casting='unsafe')

        print("Regridding...")
        
        if (j == 1) and (i == 0):
            regridder_hamo_WF = xe.Regridder(hamo_WF, ds_reg, "bilinear", filename = work_dir + 'hamo_to_regular_bilinear.nc', reuse_weights = False)
        else :
            regridder_hamo_WF = xe.Regridder(hamo_WF, ds_reg, "bilinear", filename = work_dir + 'hamo_to_regular_bilinear.nc', reuse_weights = True)
        hamo_WF = regridder_hamo_WF(hamo_WF)
        
        regridder_hamo_noWF = xe.Regridder(hamo_noWF, ds_reg, "bilinear", filename = work_dir + 'hamo_to_regular_bilinear.nc', reuse_weights = True)
        hamo_noWF = regridder_hamo_noWF(hamo_noWF)
        
        # TOTAL WIND SPEED
        hamo_WF['ws'] = np.sqrt(hamo_WF['v']**2 + hamo_WF['u']**2)
        hamo_WF['10ws'] = np.sqrt(hamo_WF['10v']**2 + hamo_WF['10u']**2)
        hamo_noWF['ws'] = np.sqrt(hamo_noWF['v']**2 + hamo_noWF['u']**2)
        hamo_noWF['10ws'] = np.sqrt(hamo_noWF['10v']**2 + hamo_noWF['10u']**2)
        
        # WIND DIRECTION
        hamo_noWF['wdir'] = np.arctan2(hamo_noWF['u'], hamo_noWF['v']) * 180/np.pi + 180
        hamo_WF['wdir'] = np.arctan2(hamo_WF['u'], hamo_WF['v']) * 180/np.pi + 180
        
        hamo_WF = hamo_WF.rename({'lat':'latitude', 'lon':'longitude'})
        hamo_noWF = hamo_noWF.rename({'lat':'latitude', 'lon':'longitude'})
        
        hamo_WF.to_netcdf(path = output_dir + 'withOWF/crop' + month + str(j).zfill(2) + str(h).zfill(2) + '.nc')
        print("Completed hamo_WF")
        
        hamo_noWF.to_netcdf(path = output_dir + 'withoutOWF/crop' + month + str(j).zfill(2) + str(h).zfill(2) + '.nc')
        print("Completed hamo_noWF")

        print("Done with regridding: " + str(3*i).zfill(2) + "/23")
    print("Done with day: " + str(j).zfill(2) + "/31")
print("Done with everything!")