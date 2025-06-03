import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pyresample import geometry, kd_tree
from satpy import Scene
import xarray as xr
from datetime import datetime


# Load MSG and GPM IMERG data
msg_file = r"C:\Users\adity\Desktop\Data_Set\data_eumetsat\Level1B\MSG2-SEVI-MSG15-0100-NA-20250528201241.343000000Z-NA\MSG2-SEVI-MSG15-0100-NA-20250528201241.343000000Z-NA.nat"
imerg_file = r"C:\Users\adity\Desktop\Data_Set\GPM_IMERG_late_run\3B-HHR-L.MS.MRG.3IMERG.20250528-S200000-E202959.1200.V07B.HDF5.nc4"


#extraction of data from MSG 
channel = Scene(reader='seviri_l1b_native', filenames=[msg_file])
channel.load(['IR_108', 'IR_039', 'IR_120'], generate=True)
ir10_8 = channel['IR_108'].values
ir3_9 = channel['IR_039'].values
ir12_0 = channel['IR_120'].values
msg_lon, msg_lat = channel['IR_108'].attrs['area'].get_lonlats()
msg_time = channel.start_time
print(f"MSG timestamp: {msg_time}")


# extraxtion of data from GPM IMERG
imerg_data = xr.open_dataset(imerg_file)
rain = imerg_data['precipitation']
imerg_time = imerg_data['time'].values.astype('datetime64[ns]')
target_time64 = np.datetime64(msg_time)
closest_idx = np.abs(imerg_time - target_time64).argmin()
imerg_time = imerg_time[closest_idx]
rain_data = rain[closest_idx].values
lat = imerg_data['lat'].values
lon = imerg_data['lon'].values
imerg_lon_grid, imerg_lat_grid = np.meshgrid(lon, lat, indexing='ij')
print(f"GPM timestamp: {imerg_time}")


# Temporal correction
time_diff = abs((np.datetime64(msg_time) - np.datetime64(imerg_time)) / np.timedelta64(1, 'm'))
if time_diff > 30:
    print(f"Time difference too large: {time_diff} minutes.")
    exit()


# Bounding to Meghalaya region
lat_min, lat_max, lon_min, lon_max = 25.10, 26.7, 89.5, 92.48
mask_meghalaya = (imerg_lat_grid >= lat_min) & (imerg_lat_grid <= lat_max) & \
                 (imerg_lon_grid >= lon_min) & (imerg_lon_grid <= lon_max)
gpm_rain_sub = np.where(mask_meghalaya, rain_data, np.nan)
gpm_lats_sub = imerg_lat_grid
gpm_lons_sub = imerg_lon_grid


# locating GPM IMERG data on MSG SEVIRI over tha span of 100 KM
msg_def = geometry.SwathDefinition(lons=msg_lon, lats=msg_lat)
imerg_def = geometry.SwathDefinition(lons=gpm_lons_sub, lats=gpm_lats_sub)
imerg_on_msg = kd_tree.resample_nearest(
    imerg_def, gpm_rain_sub, msg_def,
    radius_of_influence=100000,
    fill_value=np.nan
)


# convective system conditions according to HEM MOSDAC ATBD
convective_mask = (ir10_8 < 240) & (imerg_on_msg > 0)
convective_tb = ir10_8[convective_mask]
convective_rain = imerg_on_msg[convective_mask]

if len(convective_tb) == 0:
    print("No convective pixels found.")
    exit()
mask_fit = (convective_tb > 0) & (convective_rain >2)


# Fit empirical constants in the empirical formula
# R= a.(Tb)^b
#taking log which becomes
# ln_rain = ln_a + b*ln_tb
convective_tb_fit = convective_tb[mask_fit]
convective_rain_fit = convective_rain[mask_fit]
ln_tb = np.log(convective_tb_fit)
ln_rain = np.log(convective_rain_fit)

def model(x, ln_a, b):
    return ln_a + b * x

popt, _ = curve_fit(model, ln_tb, ln_rain)
ln_a, b = popt
a = np.exp(ln_a)

print(f"Estimated constants: a = {a:.4e}, b = {b:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(convective_tb, convective_rain, s=5, alpha=0.3, label="Convective Pixels")
Tb_model = np.linspace(np.min(convective_tb), np.max(convective_tb), 100)
R_model = a * Tb_model ** b
plt.plot(Tb_model, R_model, 'r-', label=f"Fit: R={a:.2e}*Tb^{b:.2f}")
plt.xlabel('Brightness Temperature (K)')
plt.ylabel('Rain Rate (mm/h)')
plt.title('Empirical Convective Rain Rate ')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
