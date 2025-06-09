# Import necessary Python libraries
import h5py
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import xarray as xr
from pyresample import geometry, kd_tree
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from skimage.transform import resize


# Filepaths
imerg_file = "C:/Users/adity/Desktop/Data_Set/GPM_IMERG_late_run/3B-HHR-L.MS.MRG.3IMERG.20250528-S000000-E002959.0000.V07B.HDF5"
insat_file = "C:/Users/adity/Desktop/Data_Set/INSAT_level1C/3RIMG_28MAY2025_0015_L1C_ASIA_MER_V01R00.h5"


# INSAT lat and lon bounds
insat_lat_min, insat_lat_max = -10.0, 45.5
insat_lon_min, insat_lon_max = 44.5, 110.0 


#Read Attributes and data path of both INSAT LEVEL1C and IMERG LATE RUN LEVEL 3
def insat_l1c(filepath):
    with h5py.File(filepath, 'r') as f:
        print("Attributes of INSAT")
        for key, val in f.attrs.items():
            print(f"{key}: {val}")

        print("\n Data Path")
        def print_name(name):
            print(name)
        f.visit(print_name)
#insat_l1c(insat_file)

def imerg_l3(filepath):
    with h5py.File(filepath, 'r') as f:
        print("Attributes of IMERG")
        for key, val in f.attrs.items():
            print(f"{key}: {val}")

        print("\n Data Path")
        def print_name(name):
            print(name)
        f.visit(print_name)
#imerg_l3(imerg_file)


#data extraction from INSAT
insat_data = xr.open_dataset(insat_file)
BT_ir1 = insat_data['IMG_TIR1_TEMP']
BT_wv = insat_data['IMG_WV_TEMP']
insat_time = insat_data['time']
insat_lon, insat_lat = insat_data['IMG_TIR1_TEMP','IMG_WV_TEMP'].attrs['area'].get_lonlats()


#temporal correction
#since it have different time epoches
#for insat minutes since 2000-01-01 00:00:00 =        13361775.0
#converting to imerg time by adding 10,512,000 min and converting to seconds
#this comes out to be 1432426500
#for IMERG seconds since 1980-01-06 00:00:00 UTC" = 1432425600
insat_ref = datetime(2000, 1, 1, 0, 0, 0)
imerg_ref = datetime(1980, 1, 6, 0, 0, 0)
with h5py.File(imerg_file, 'r') as f:
    gtime_data = f['Grid/time'][:] 
    #print(f"IMERG time raw data (seconds since {imerg_ref.date()}):\n", gtime_data)

with h5py.File(insat_file, 'r') as f:
    time_data = f['time'][:] 
    #print(f"INSAT time raw data (minutes since {insat_ref.date()}):\n", time_data)
    #Convert INSAT time to seconds since IMERG reference
    insat_datetimes = [insat_ref + timedelta(minutes=m) for m in time_data]
    # Convert INSAT datetimes to seconds since IMERG reference
    insat_seconds_aligned = np.array([(dt - imerg_ref).total_seconds() for dt in insat_datetimes])
    #print(f"INSAT time converted to seconds since {imerg_ref.date()} (IMERG reference):\n", insat_seconds_aligned)


#data extraction from IMERG
imerg_data = xr.open_dataset(imerg_file)
imerg_rain = imerg_data['Grid/precipitation'].values
imerg_time = imerg_data['time'].values.astype('datetime64[ns]')
target_time64 = np.datetime64(insat_seconds_aligned)
closest_idx = np.abs(imerg_time - target_time64).argmin()
imerg_time = imerg_time[closest_idx]
imerg_time_rain = imerg_rain[closest_idx].values
imerg_lat = imerg_data['lat'].values
imerg_lon = imerg_data['lon'].values
imerg_lon_grid, imerg_lat_grid = np.meshgrid(imerg_lon, imerg_lat, indexing='ij')

# Bound IMERG to INSAT region
imerg_mask = (imerg_lat_grid >= insat_lat_min) & (imerg_lat_grid <= insat_lat_max) & \
             (imerg_lon_grid >= insat_lon_min) & (imerg_lon_grid <= insat_lon_max)
imerg_rain_mask = np.where(imerg_mask, imerg_time_rain, np.nan)

#INSAT data on GPM INSET data
insat_def = geometry.SwathDefinition(lons=insat_lon, lats=insat_lat)
imerg_def = geometry.SwathDefinition(lons=imerg_lon_grid, lats=imerg_lat_grid)
insat_on_imerg = kd_tree.resample_nearest(
    imerg_def, imerg_rain_mask, insat_def,
    radius_of_influence=10000,
    fill_value=np.nan
)

# HEM model
def hem_model(X, x0, y0, x1, y1):
    BTD, Tb = X
    return x0 + x1 * BTD + y1 * Tb + y0 * BTD * Tb

# Main analysis function
def main():
    try:
        with h5py.File(insat_file, 'r') as f:
            Tb = f['IMG_TIR1_TEMP'][:]       
            T_wv = f['IMG_WV_TEMP'][:]     

        BTD = T_wv - Tb

        with h5py.File(imerg_file, 'r') as f:
            R = f['Grid/precipitation'][:]

        Tb_flat = Tb.flatten()
        BTD_flat = BTD.flatten()
        R_flat = insat_on_imerg.flatten()  

        if R_flat.size != Tb_flat.size:
            R_resized = resize(insat_on_imerg, Tb.shape, preserve_range=True).flatten()
        else:
            R_resized = R_flat

        valid_mask = (R_resized > 0) & (Tb_flat > 200) & (BTD_flat > 0) & \
                     np.isfinite(R_resized) & np.isfinite(Tb_flat) & np.isfinite(BTD_flat)

        if np.count_nonzero(valid_mask) == 0:
            print("[EXIT] No valid data points found.")
            return

        Tb_all = Tb_flat[valid_mask]
        BTD_all = BTD_flat[valid_mask]
        R_all = R_resized[valid_mask]

        print(f"[INFO] Valid data points: {len(R_all)}")

        initial_guess = [1.0, 1.0, -0.05, -0.01]
        popt, _ = curve_fit(hem_model, (BTD_all, Tb_all), R_all, p0=initial_guess, maxfev=10000)
        x0, y0, x1, y1 = popt

        print("\n[FITTED PARAMETERS]")
        print(f"x0 = {x0:.4f}")
        print(f"y0 = {y0:.4f}")
        print(f"x1 = {x1:.4f}")
        print(f"y1 = {y1:.4f}")

        R_pred = hem_model((BTD_all, Tb_all), *popt)
        r2 = r2_score(R_all, R_pred)
        print(f"Model R² score: {r2:.3f}")

    except Exception as e:
        print(f"[ERROR] Processing failed: {e}")

# Run main
main()
