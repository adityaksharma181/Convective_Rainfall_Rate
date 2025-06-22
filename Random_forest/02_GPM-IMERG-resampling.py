import h5py
import xarray as xr
import numpy as np

# Input and Output paths for file
input_path = "3B-HHR-L.MS.MRG.3IMERG.20250528-S130000-E132959.0780.V07B.HDF5"  #  <------------------ File Path
output_path = "gpm-imerg-resampled.nc"         #  <------------------ File Path

# Spatial Boundary Condition
lat_min, lat_max = -10.0, 45.5
lon_min, lon_max = 44.5, 110.0

# Extraction of Variables from file
with h5py.File(input_path, 'r') as f:
    lon = f['Grid']['lon'][:] 
    lat = f['Grid']['lat'][:]  
    precip = f['Grid']['precipitation'][0, :, :]  

  # Replace fill values with NaN
    precip = np.where(precip == -9999.9, np.nan, precip)

# Transpose precip from (lon, lat) to (lat, lon) to match the grid
precip = precip.T  

# Xarray Dataset to store necessory variables
ds = xr.Dataset(
    {
        "precipitation": (["lat", "lon"], precip)
    },
    coords={
        "lat": lat,
        "lon": lon
    }
)

# Subset spatial bounds
ds_subset = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

# Assign attributes for better management
ds_subset["precipitation"].attrs["units"] = "mm/hr"
ds_subset["precipitation"].attrs["long_name"] = "IMERG Precipitation Rate"
ds_subset["lat"].attrs["units"] = "degrees_north"
ds_subset["lon"].attrs["units"] = "degrees_east"

# Save to NetCDF
ds_subset.to_netcdf(output_path)
print(f"Subset data saved to: {output_path}")
