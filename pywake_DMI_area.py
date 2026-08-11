#srun --ntasks=1 --time=01:00:00 --mem=8000 --account=project_2010748 --partition=small --cpus-per-task=1 --pty $SHELL
import py_wake as pw
import numpy as np
import xarray as xr
import dask.distributed
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import pandas as pd
import xesmf as xe
from scipy.spatial import cKDTree
from scipy.special import erf
from joblib import Parallel, delayed
import glob
import dask
import os
import socket
from sys import argv
from copy import deepcopy

from pyproj import Proj, Transformer, CRS
#from memory_profiler import profile

from py_wake.site import XRSite
from py_wake.site.shear import Shear, PowerShear, MOSTShear
from numpy import newaxis as na
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit,IEA37SimpleBastankhahGaussianDeficit
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_farm_models import PropagateUpDownIterative, PropagateDownwind
from py_wake.superposition_models import LinearSum
from py_wake.deficit_models import VortexDipole
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.site.distance import StraightDistance
from py_wake.utils.numpy_utils import Numpy32

def wake(data, wt_x, wt_y, windTurbines, x_reg, y_reg,
        height=100,  k = 0.03):
    """
    The actual calculation using PyWake.

    PARAMETERS :
        data : xArray.dataset
            Wind data at hub-height featuring dims (x, y) ; coords [x] [y] [latitude] [longitude] ; variables [10v] [10u] [100v] [100u] [wdir] [mask]
        wt_x : 1Darray(float)
        wt_y : 1Darray(float)
            Coordinates of each turbine in the local coordinate system
        windTurbines : WindTurbines
        
        x_reg : 1Darray(float)
        y_reg : 1Darray(float)
            Center points of the local grid
        height : float
            Hub-height
        k : float
            Growth rate coefficient
    RETURNS :
        ds_final : xArray.dataset
            Output of PyWake's calculation featuring dims/coords (x, y) ; variables [ws] [wd] [mask] [ws_orig]
    """

    amb10 = np.sqrt(data['10u']**2 + data['10v']**2) #
    amb100 = np.sqrt(data['100u']**2 + data['100v']**2) #
    
    x_reg2 = xr.DataArray(x_reg, coords={'x':x_reg})
    #print(x_reg2.values)
    #print(x_reg)
    
    y_reg2 = xr.DataArray(y_reg, coords={'y':y_reg})
    
    """WIND DIRECTION"""
    data['wdir'] = np.arctan2(data['100u'], data['100v']) * 180/np.pi + 180
    #print(amb100)
    
    # create output data
    ds_final = xr.merge([xr.DataArray(amb100.values, dims = ('x', 'y')).to_dataset(name = 'ws'), #
                         xr.DataArray(data.wdir.values, dims = ('x', 'y')).to_dataset(name = 'wd'),
                         xr.DataArray(data.mask.values, dims = ('x', 'y')).to_dataset(name = 'mask'),
                         xr.DataArray(amb100.values.copy(), dims = ('x', 'y')).to_dataset(name = 'ws_orig'), #
    ]).assign_coords({'latitude':data.latitude, 'longitude':data.longitude, 'x':data.x, 'y':data.y})
    
    # create output grid
    grid = pw.HorizontalGrid(y = y_reg2.astype(np.float32), x = x_reg2.astype(np.float32), h = height)
    
    print('create input dataset')
    
    ###################################################################
    #THE DIMENSIONS NEED TO BE X-Y NOT Y-X
    # create input dataset
    ds=xr.Dataset(data_vars = {'WS':amb100.astype(np.float32), #
                               'WD':data.wdir.astype(np.float32),
                               'P':1,
                               'TI':np.float32(0.05)},
    ).assign_coords({'x':data['100u'].x.astype(np.float32), 'y':data['100u'].y.astype(np.float32)}).load() #
    #print(ds)
    
    print('define wind farm model')
    
    site     = XRSite(ds, distance = StraightDistance(wind_direction = 'WD_i'), bounds = 'ignore'
    )
    wf_model = PropagateDownwind(site, windTurbines,
                                        wake_deficitModel=BastankhahGaussianDeficit(k = k, ceps = 1.0, ctlim = 0.9, use_effective_ws=True) #, shear = MOSTShear(h_ref=100, z0=.5, h_zeta=0.0, Cm1=5.0, Cm2=-19.3, interp_method='nearest'),
                                        )
    print('wind farm model starting')
    wf     = wf_model(wt_x.astype(np.float32), wt_y.astype(np.float32), ws=10)
    
    print('return data on a grid')
    wf_out = wf.flow_map(grid) #.XYGrid(resolution=50000)) #.drop(['wd','ws'])
    #print(wf_out)
    
    
    print('actual calculation is done')
    # Replace with computed value where mask
    jinds, iinds = np.where(ds_final.mask)
    ws_final    = ds_final['ws'].values #
    ws_region   = wf_out.WS_eff.drop_vars(['wd', 'ws']).squeeze()
    #print(ws_final.shape)
    ws_final[jinds, iinds] = (ws_region.values[iinds, jinds]).flatten()
    ds_final['ws'] = xr.DataArray(ws_final, dims=('x', 'y')) #
    
    data.close()
    del wf, wf_model
    return ds_final
    
if __name__ == '__main__':
    WFname = str(argv[1]) # Name of the wind farm
    month = str(argv[2]) # ex: 202601
    print(f"Calculation for {WFname} in {month}")
    
    output_dir = '/scratch/project_2010671/DMI-HAMOWIOWF_temp/' + WFname + '/PyWake/' # where you want to put your output
    work_dir = 'code/'
    data_folder = '/scratch/project_2010671/DMI-HAMOWIOWF_temp/'

    """TURBINES"""
    print('loop over turbines')
    fnamedico = work_dir + 'turbines_offshore.csv' # path to the turbine location csv
    wt_array = pd.read_csv(fnamedico, header=0, index_col=False, usecols=[0, 1], dtype={'longitude':np.float64, 'latitude':np.float64}).to_numpy()
    
    # Select only the turbines of interest
    if WFname == 'KriegersFlak' :
        ###KRIEGERS FLAK###
        wt_sel = wt_array[(wt_array[:, 0] < 13.3) & (wt_array[:, 0] > 12.7)]
        wt_sel = wt_sel[(wt_sel[:, 1] < 55.15) & (wt_sel[:, 1] > 54.9)]
        D = 167
        y1, y2 = (45, 163) # Square region around the wind farm (for the vertical shear ratio)
        x1, x2 = (75, 250)
    elif WFname == 'Anglia' :
        ###EAST ANGLIA###
        wt_sel = wt_array[(wt_array[:, 0] < 2.7) & (wt_array[:, 0] > 2.4)]
        wt_sel = wt_sel[(wt_sel[:, 1] < 52.4) & (wt_sel[:, 1] > 52.1)]
        D = 154
        y1, y2 = (65, 160)
        x1, x2 = (60, 225)
    elif WFname == 'HornsRev' :
        ###HORNS REV 3###
        wt_sel = wt_array[(wt_array[:, 0] < 7.9) & (wt_array[:, 0] > 7.4)]
        wt_sel = wt_sel[(wt_sel[:, 1] < 55.8) & (wt_sel[:, 1] > 55.55)]
        D = 93
        y1, y2 = (63, 140)
        x1, x2 = (45, 215)
    else :
        raise NameError(f"Name '{WFname}' is not known to the program.")
    wt_lon = wt_sel[:, 0]
    wt_lat = wt_sel[:, 1]
            
    print('done with the turbines')
    print('Number of turbines', len(wt_lon))
    
    IEAturbine_carac = pd.read_csv(work_dir + 'IEA_Reference_15MW_240.csv', delimiter=',')
    power_IEA = IEAturbine_carac['Power [kW]'].values
    ct = IEAturbine_carac['Ct [-]'].values
    u_wt = IEAturbine_carac['Wind Speed [m/s]'].values
    h = 100
    P = 15e6
    
    region='DMI'
    
    # we need to do smaller subdomains
    # For each region, the slices in the local coordinate system
    # (used for the mask)
    ###KRIEGERS FLAK###
    #x_slice = slice(-0.2E6, 0.17E6)
    #y_slice = slice(-0.27E6, 0.28E6)
    ###EAST ANGLIA###
    x_slice = slice(-0.047E6, 0.047E6)
    y_slice = slice(-0.1E6, 0.1E6)
    print('Slices defined')

    if (month == '202512') or (month == '202601') or (month == '202603') :
        list_j = range(1,32)
    elif (month == '202602') :
        list_j = range(1,29)
    elif (month == '202604') :
        list_j = range(1,31)
    else :
        raise ValueError("Month must be '202512', '202601', '202602', '202603' or '202604'.") 
    for j in list_j :
        for i in ['00', '03', '06', '09', '12', '15', '18', '21'] :
            fname = month + str(j).zfill(2) + i

            h_noWF = xr.open_dataset(data_folder + WFname + '/withoutOWF/crop'+fname+'.nc').squeeze() # Path to data
            #h_noWF = h_noWF.rename({'lat':'latitude', 'lon':'longitude'})
            
            # Drop unwanted dimensions
            if 'pres' in h_noWF.data_vars :
                h_noWF = h_noWF.drop_vars(['pres'])
            if 'u' in h_noWF.data_vars :
                h_noWF = h_noWF.drop_vars(['u', 'v'])
            if 'height_4' in h_noWF.dims :
                h_noWF = h_noWF.drop_dims(['height_4'])
            if 'height_2' in h_noWF.dims :
                h_noWF = h_noWF.drop_dims(['height_2'])
            #print(h_noWF)
            
            ### ---------- AREA ADJUSTMENT -------------###
            ### RATIO ### Ratio between 10m and 100m wind speeds
            area = h_noWF.isel(x = slice(x1, x2), y = slice(y1, y2))
            mean_ratio = np.mean(np.sqrt(area['10v'].values**2 + area['10u'].values**2) / np.sqrt(area['100v'].values**2 + area['100u'].values**2))
            
            ### WS100 ### Mean of the wind speed received by each turbine at 100m
            tree = cKDTree(np.column_stack([h_noWF.longitude.values.ravel(), h_noWF.latitude.values.ravel()]))
            dist, idx = tree.query(wt_sel)
            ix, iy = np.unravel_index(idx, h_noWF.longitude.values.shape)
            hnoWF_wt = h_noWF.isel(x = ix, y = iy)
        
            mean_u = hnoWF_wt['100u'].mean(dim = ('x', 'y'))
            mean_v = hnoWF_wt['100v'].mean(dim = ('x', 'y'))
        
            ws100 = np.sqrt(hnoWF_wt['100u']**2 + hnoWF_wt['100v']**2).mean(dim = ('x', 'y'))
            
            ### NEW ROTOR DIAMETER ###
            alpha = np.exp(-(ws100 - 11.60)**2/27.55)
            
            coeff_D = np.clip(4.066 + np.log((40.21/2662)/(1+erf(-1.116e-3 * (ws100 - 1.828))) \
                             * np.exp(4.190e-2 * (ws100 - 11.05)**2 - (4.578 * alpha - 0.4232) * (mean_ratio + (0.3827 * alpha - 1.543))) - 1.851e-2), 0, 30)
            
            k = (0.04 * (coeff_D + 0.01)**0.15)
            
            IEA15MW = WindTurbine(name='IEA15MW',
                                diameter= D * coeff_D,
                                hub_height=h,
                                powerCtFunction=PowerCtTabular(u_wt,power_IEA,'kW', ct))
        
            windTurbines = WindTurbines.from_WindTurbine_lst([IEA15MW])

            # REGRIDDING
            lon2d = h_noWF['longitude'].values
            lat2d = h_noWF['latitude'].values
            #print(lat2d)
            #print(h_noWF)
            
            lon_reg = lon2d[:, 0] # Center points
            dlon = np.diff(lon_reg)/2
            lon_b = np.concatenate(([lon_reg[0] - dlon[0]], lon_reg[:-1] + dlon, [lon_reg[-1] + dlon[-1]])) # Bondaries
            
            lat_reg = lat2d[0, :] # Center points
            dlat = np.diff(lat_reg)/2
            lat_b = np.concatenate(([lat_reg[0] - dlat[0]], lat_reg[:-1] + dlat, [lat_reg[-1] + dlat[-1]])) # Bondaries
            
            ds_regular_latlon = xr.merge([xr.DataArray(lon_reg, dims = ('x')).to_dataset(name = 'lon'),
                                          xr.DataArray(lat_reg, dims = ('y')).to_dataset(name = 'lat'),
                                          xr.DataArray(lon_b, dims = ('x_b')).to_dataset(name = 'lon_b'),
                                          xr.DataArray(lat_b, dims = ('y_b')).to_dataset(name = 'lat_b')])
            #print(ds_regular_latlon)
            
            # define a regular xy grid reference to the mid-point of the regular lat/lon grid
            latituderef = np.median(lat_reg)
            longituderef = np.median(lon_reg)
            
            # Projections & Transforms
            proj_string = f"+proj=aeqd +lat_0={latituderef} +lon_0={longituderef} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            proj_local = Proj(proj_string)
            proj_latlon = Proj(proj = "latlong", datum = "WGS84")

            transformer_locTolatlon = Transformer.from_proj(proj_local, proj_latlon)
            transformer_latlonToloc = Transformer.from_proj(proj_latlon, proj_local)

            xmin, ymin = transformer_latlonToloc.transform(lon_reg[0], lat_reg[0])
            xmax, ymax = transformer_latlonToloc.transform(lon_reg[-1], lat_reg[-1])

            wt_x, wt_y = transformer_latlonToloc.transform(wt_lon, wt_lat) #Turbines in local coordinates

            dx = dy = 5e2 # Resolution of the local grid: 500m
            
            x_reg = np.arange(xmin, xmax + dx, dx) # Center points
            x_b = np.arange(xmin - dx/2, xmax + dx + dx/2, dx) # Bondaries
            #print(x_reg)
            y_reg = np.arange(ymin, ymax + dy, dy) # Center points
            y_b = np.arange(ymin - dy/2, ymax + dy + dy/2, dy) # Bondaries
            #print(y_reg)
            
            y2d, x2d = np.meshgrid(y_reg, x_reg)
            y_b2d, x_b2d = np.meshgrid(y_b, x_b) 
            lon2dxy, lat2dxy = transformer_locTolatlon.transform(x2d, y2d)
            lon_b2dxy, lat_b2dxy = transformer_locTolatlon.transform(x_b2d, y_b2d)
            #print(x2d)
            #print(x2d.shape)
            
            # create datasets for both
            ds_regular = xr.merge([xr.DataArray(lat2d, dims = ('x', 'y')).to_dataset(name = 'latitude'),
                                   xr.DataArray(lon2d, dims = ('x', 'y')).to_dataset(name = 'longitude')]
            )
            ds_regular_xy = xr.merge([xr.DataArray(lat2dxy, dims = ('x', 'y')).to_dataset(name = 'latitude'),
                                      xr.DataArray(lon2dxy, dims = ('x', 'y')).to_dataset(name = 'longitude'),
                                      xr.DataArray(lat_b2dxy, dims = ('x_b', 'y_b')).to_dataset(name = 'lat_b'),
                                      xr.DataArray(lon_b2dxy, dims = ('x_b', 'y_b')).to_dataset(name = 'lon_b')]
                                )
            #print(ds_regular_xy)
            #print(h_noWF['100u'])
            
            # regrid h_noWF to the regular xy grid
            if j == 1 and i == '00' : # Initialize weights
                regridder_h_noWF = xe.Regridder(h_noWF, ds_regular_xy, "bilinear", filename=work_dir + 'h_to_regular_precise.nc', reuse_weights=False)
                regridder_xy2latlon = xe.Regridder(ds_regular_xy, ds_regular_latlon, "conservative", filename=work_dir + 'h_XY_to_regular_conserv.nc', reuse_weights=False)
            else :
                regridder_h_noWF = xe.Regridder(h_noWF, ds_regular_xy, "bilinear", filename=work_dir + 'h_to_regular_precise.nc', reuse_weights=True) # Increasing resolution
                regridder_xy2latlon = xe.Regridder(ds_regular_xy, ds_regular_latlon, "conservative", filename=work_dir + 'h_XY_to_regular_conserv.nc', reuse_weights=True) # Decreasing resolution
            
            data = regridder_h_noWF(h_noWF)
            
            #print(data['100u'])

            data = data.assign_coords({'longitude':ds_regular_xy.longitude, 'latitude':ds_regular_xy.latitude,
                                    'x':xr.DataArray(x_reg, dims = ('x')), 'y':xr.DataArray(y_reg, dims = ('y'))})
            
            data = xr.merge([data, xr.DataArray(xr.zeros_like(data['10u'])).to_dataset(name = 'mask')]) # Initialize mask
            mask = np.logical_and(np.logical_and( x2d > x_slice.start, x2d < x_slice.stop), np.logical_and(y2d > y_slice.start, y2d < y_slice.stop))
            data.mask.values = mask[:, :]
            print('Applied mask')

            ds_final = wake(data, wt_x, wt_y, windTurbines, x_reg, y_reg, height = 100, k = k)
            ds_final = ds_final.drop_vars(['mask'])

            # REGGRIDING SHOULD BE DONE WITH U AND V
            v = -ds_final['ws'] * np.cos(ds_final['wd'] * np.pi/180) #
            u = -ds_final['ws'] * np.sin(ds_final['wd'] * np.pi/180) #

            ds_final['100u'] = (('x','y'), u.data) #
            ds_final['100v'] = (('x','y'), v.data) #
            
            ds_final_reg = regridder_xy2latlon(ds_final).assign_coords({'longitude':ds_regular.longitude, 'latitude':ds_regular.latitude}).transpose('x', 'y')
            
            """WIND DIRECTION"""
            ds_final_reg['wd'] = np.arctan2(ds_final_reg['100u'], ds_final_reg['100v']) * 180/np.pi + 180 #

            """WIND SPEED"""
            ds_final_reg['ws'] = np.sqrt(ds_final_reg['100u']**2 + ds_final_reg['100v']**2) #
            #print(ds_final_reg)
            
            # CORRELATION FOR 10-METER WINDS
            mask = 3 * np.abs(np.sqrt(h_noWF['100v']**2 + h_noWF['100u']**2) - ds_final_reg['ws'])/np.sqrt(h_noWF['100v']**2 + h_noWF['100u']**2)
            ds_final_reg['10ws'] = (1 - mask) * np.sqrt(h_noWF['10v']**2 + h_noWF['10u']**2) + mask * np.sqrt(h_noWF['10v']**2 + h_noWF['10u']**2) * \
                            np.clip((3.266 * np.exp(-5.473 * (mean_ratio + 1.503)) * np.exp(5.341 * (ds_final_reg['ws']/np.sqrt(h_noWF['100v']**2 + h_noWF['100u']**2) + 0.7554)) + 0.8849), 0, 3)
            
            v = -ds_final_reg['10ws'] * np.cos((np.arctan2(h_noWF['10u'], h_noWF['10v']) * 180/np.pi + 180)*np.pi/180) # Suppose same direction as model without WF
            u = -ds_final_reg['10ws'] * np.sin((np.arctan2(h_noWF['10u'], h_noWF['10v']) * 180/np.pi + 180)*np.pi/180)
            
            ds_final_reg['10u'] = (('x','y'), u.data) #
            ds_final_reg['10v'] = (('x','y'), v.data) #
            
            file = output_dir + region + '_reg_area_' + fname.split('/')[-1] + '.nc' #

            ds_final_reg.to_netcdf(file)
            print(f'regridding and netcdf done to : {file}')
        print(f"Done with day : {j}/{len(list_j)}")