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
    return ln_a - b * x

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










#HEM files
import os
import h5py
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm
from skimage.transform import resize

# ----------------- CONFIGURATION -----------------
INSAT_DIR = "C:\\Users\\adity\\Desktop\\Data_Set\\INSAT_level1C\\"   # Directory for INSAT data files
IMERG_DIR = "C:\\Users\\adity\\Desktop\\Data_Set\\GPM_IMERG_late_run\\"          # Directory for IMERG data files

# Define bounding box for the Indian subcontinent
LAT_MIN, LAT_MAX = 5, 38
LON_MIN, LON_MAX = 68, 98

# ----------------- HEM MODEL FUNCTION -----------------
def hem_model(inputs, x0, y0, x1, y1):
    """HEM model: Relates BTD and Tb to precipitation rate."""
    BTD, Tb = inputs
    A = x0 + y0 * BTD
    B = x1 + y1 * BTD
    return A * (Tb ** B)

# ----------------- PARSE DATETIME FROM FILENAME -----------------
def parse_datetime_from_filename(filename, is_insat=True):
    """Extract datetime from INSAT or IMERG filenames."""
    if is_insat:
        # Expected INSAT filename format: '3RIMG_28MAY2025_0015_L1C_ASIA_MER_V01R00.h5'
        parts = filename.split('_')
        if len(parts) < 3:
            raise ValueError(f"Unexpected INSAT filename format: {filename}")
        date_str = parts[1]  # e.g. '28MAY2025'
        time_str = parts[2]  # e.g. '0015'
        datetime_str = date_str + time_str  # e.g. '28MAY20250015'
        return datetime.strptime(datetime_str, "%d%b%Y%H%M")
    else:
        # IMERG filename format assumed like '3B-HHR.MS.MRG.3IMERG.20250605-S003000-E005959.V07.HDF5'
        parts = filename.split('.')
        if len(parts) < 5:
            raise ValueError(f"Unexpected IMERG filename format: {filename}")
        time_str = parts[4][0:13]  # e.g. '20250605-S0030'
        # Parse with format including 'S' and 'E' - safer to just strip the 'S' and parse manually
        date_part, time_part = time_str.split('-S')
        dt_str = date_part + time_part  # e.g. '202506050030'
        return datetime.strptime(dt_str, "%Y%m%d%H%M")

# ----------------- FILE LISTING -----------------
insat_files = sorted([f for f in os.listdir(INSAT_DIR) if f.endswith('.h5')])
imerg_files = sorted([f for f in os.listdir(IMERG_DIR) if f.endswith('.HDF5')])

# ----------------- DATA STORAGE -----------------
BTD_all, Tb_all, R_all = [], [], []

# ----------------- MAIN PROCESSING LOOP -----------------
for insat_filename in tqdm(insat_files, desc="Processing files"):
    try:
        insat_path = os.path.join(INSAT_DIR, insat_filename)
        insat_time = parse_datetime_from_filename(insat_filename, is_insat=True)

        # Find matched IMERG file within ±30 minutes
        matched_imerg = None
        for imerg_filename in imerg_files:
            try:
                imerg_time = parse_datetime_from_filename(imerg_filename, is_insat=False)
            except Exception:
                continue
            if abs((insat_time - imerg_time).total_seconds()) <= 1800:
                matched_imerg = os.path.join(IMERG_DIR, imerg_filename)
                break
        if not matched_imerg:
            continue  # Skip if no matching IMERG file found

        # ----------- READ INSAT DATA -----------
        with h5py.File(insat_path, 'r') as f:
            Tb = f['IMG_TIR1_TEMP'][:]      # Infrared brightness temperature
            T_wv = f['IMG_WV_TEMP'][:]     # Water vapor brightness temperature
            lat = f['Latitude'][:]
            lon = f['Longitude'][:]

        # Apply geographical mask to limit data to Indian region
        geo_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX) & (lon >= LON_MIN) & (lon <= LON_MAX)
        if not np.any(geo_mask):
            continue

        Tb = Tb[geo_mask]
        T_wv = T_wv[geo_mask]
        BTD = T_wv - Tb  # Brightness temperature difference

        # ----------- READ IMERG DATA -----------
        with h5py.File(matched_imerg, 'r') as f:
            R = f['Grid/precipitationCal'][:]  # Precipitation rate
            lat_imerg = f['Grid/lat'][:]
            lon_imerg = f['Grid/lon'][:]

        # Create meshgrid for lat/lon and mask it to Indian region
        lat_grid, lon_grid = np.meshgrid(lat_imerg, lon_imerg, indexing='ij')
        imerg_mask = (lat_grid >= LAT_MIN) & (lat_grid <= LAT_MAX) & (lon_grid >= LON_MIN) & (lon_grid <= LON_MAX)
        if not np.any(imerg_mask):
            continue

        R = R[imerg_mask]

        # Resize IMERG data to match INSAT cropped shape if sizes differ
        if R.size != Tb.size:
            R = resize(R.reshape(-1, 1), Tb.shape, preserve_range=True).flatten()

        # Keep only valid data points (finite, positive, realistic values)
        valid = (R > 0) & (Tb > 200) & (BTD > 0) & np.isfinite(R) & np.isfinite(Tb) & np.isfinite(BTD)
        if np.count_nonzero(valid) == 0:
            continue

        BTD_all.append(BTD[valid])
        Tb_all.append(Tb[valid])
        R_all.append(R[valid])

    except Exception as e:
        print(f"Error with file {insat_filename}: {e}")
        continue

# ----------------- MODEL FITTING -----------------
if len(BTD_all) == 0 or len(Tb_all) == 0 or len(R_all) == 0:
    print("No valid data found after processing all files. Exiting.")
    exit(1)

BTD_all = np.concatenate(BTD_all)
Tb_all = np.concatenate(Tb_all)
R_all = np.concatenate(R_all)

print(f"\nTotal data points in Indian region: {len(R_all)}")

# Initial guess for model parameters
initial_guess = [1.0, 1.0, -0.05, -0.01]

# Fit the HEM model using nonlinear curve fitting
popt, _ = curve_fit(hem_model, (BTD_all, Tb_all), R_all, p0=initial_guess, maxfev=10000)
x0, y0, x1, y1 = popt

# Print the fitted parameters
print("\nFitted HEM Coefficients (Indian Region):")
print(f"x0 = {x0:.4f}")
print(f"y0 = {y0:.4f}")
print(f"x1 = {x1:.4f}")
print(f"y1 = {y1:.4f}")

# Compute predicted rainfall and R² score
R_pred = hem_model((BTD_all, Tb_all), *popt)
print(f"Model R²: {r2_score(R_all, R_pred):.3f}")
