import os
import sys
import xarray as xr

p = sys.argv[1] if len(sys.argv)>1 else r"C:\Users\sgddsf\Desktop\minimum\ArcticRoute\data\raw\cmems_arc\cmems_mod_arc_phy_my_topaz4_P1M_1762143591239.nc"
print('path:', p)
print('exists:', os.path.exists(p))

ds = xr.open_dataset(p)
print('dims:', dict(ds.sizes))
print('coords:', list(ds.coords))
print('vars:', list(ds.data_vars))
for k in list(ds.data_vars)[:20]:
    da = ds[k]
    print('var', k, 'dims', da.dims, 'units', da.attrs.get('units',''), 'long_name', da.attrs.get('long_name',''))

