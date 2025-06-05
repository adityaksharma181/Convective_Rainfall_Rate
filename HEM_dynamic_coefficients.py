import os
import h5py
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm
from skimage.transform import resize

# ----------------- CONFIGURATION -----------------
INSAT_DIR = "C:\\Users\\adity\\Desktop\\Data_Set\\INSAT_level1C\\"
IMERG_DIR = "C:\\Users\\adity\\Desktop\\Data_Set\\GPM_IMERG_late_run\\"

# Define bounding box for the Indian subcontinent
LAT_MIN, LAT_MAX = 5, 38
LON_MIN, LON_MAX = 68, 98

# IST offset from UTC
IST_OFFSET = timedelta(hours=5, minutes=30)

# Reference datetimes for internal time conversions
INSAT_REF = datetime(2000, 1, 1)
IMERG_REF = datetime(1980, 1, 6)

# ----------------- HEM MODEL FUNCTION -----------------
def hem_model(inputs, x0, y0, x1, y1):
    BTD, Tb = inputs
    A = x0 + y0 * BTD
    B = x1 + y1 * BTD
    return A * (Tb ** B)

# ----------------- DATETIME PARSING FROM FILENAMES -----------------
def parse_insat_datetime(filename):
    # Example INSAT filename format: 'INSAT_202505282115_L1C.h5' or '3RIMG_28MAY2025_2115_L1C_ASIA_MER_V01R00.h5'
    try:
        parts = filename.split('_')
        # Your filename might be like '3RIMG_28MAY2025_2115_L1C_ASIA_MER_V01R00.h5' for INSAT
        # or if different, adjust accordingly
        # Here we try both:
        if parts[0].startswith("INSAT"):
            date_str = parts[1]  # '202505282115' YYYYMMDDHHMM
            dt_ist = datetime.strptime(date_str, '%Y%m%d%H%M')
        else:
            # Example: '3RIMG_28MAY2025_2115_L1C_ASIA_MER_V01R00.h5'
            date_str = parts[1]  # '28MAY2025'
            time_str = parts[2]  # '2115'
            dt_ist = datetime.strptime(date_str + time_str, '%d%b%Y%H%M')
        return dt_ist
    except Exception as e:
        print(f"Error parsing INSAT datetime from {filename}: {e}")
        return None

def convert_ist_to_utc(dt_ist):
    return dt_ist - IST_OFFSET

def parse_imerg_datetime(filename):
    # Example IMERG filename: '3B-HHR-L.MS.MRG.3IMERG.20250528-S000000-E002959.0000.V07B.HDF5'
    # Extract date string after the 4th dot (index 4) e.g. '20250528-S000000-E002959'
    try:
        parts = filename.split('.')
        date_part = parts[4]  # '20250528-S000000-E002959'
        date_str = date_part.split('-')[0]  # '20250528'
        # Since time is S000000-E002959 (start and end time), take start time:
        time_str = date_part.split('-')[1][1:]  # '000000'
        datetime_str = date_str + time_str  # e.g. '20250528000000'
        dt_utc = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
        return dt_utc
    except Exception as e:
        print(f"Error parsing IMERG datetime from {filename}: {e}")
        return None

# ----------------- INTERNAL TIME READING -----------------
def get_imerg_file_times(imerg_path):
    with h5py.File(imerg_path, 'r') as f:
        gtime_data = f['Grid/time'][:]  # seconds since 1980-01-06
    return [IMERG_REF + timedelta(seconds=int(s)) for s in gtime_data]

def get_insat_file_times(insat_path):
    with h5py.File(insat_path, 'r') as f:
        time_data = f['time'][:]  # minutes since 2000-01-01
    return [INSAT_REF + timedelta(minutes=m) for m in time_data]

# ----------------- CACHE IMERG TIMES -----------------
print("Caching IMERG file internal times...")
imerg_files = sorted([f for f in os.listdir(IMERG_DIR) if f.endswith('.h5')])
imerg_time_cache = {}
for imerg_file in tqdm(imerg_files, desc="Loading IMERG times"):
    path = os.path.join(IMERG_DIR, imerg_file)
    try:
        times = get_imerg_file_times(path)
        imerg_time_cache[imerg_file] = times
    except Exception as e:
        print(f"Failed to load times from {imerg_file}: {e}")
        imerg_time_cache[imerg_file] = []

# ----------------- FIND CLOSEST IMERG FILE USING INTERNAL TIMES -----------------
def find_closest_imerg_file_using_internal_time(insat_path, imerg_files):
    insat_times = get_insat_file_times(insat_path)
    if not insat_times:
        return None
    # Assume INSAT times are in IST, convert first time to UTC
    insat_time_utc = insat_times[0] - IST_OFFSET

    closest_file = None
    min_time_diff = None

    for imerg_file in imerg_files:
        imerg_times = imerg_time_cache.get(imerg_file, [])
        if not imerg_times:
            continue

        # Find closest IMERG time in this file to INSAT time
        imerg_closest_time = min(imerg_times, key=lambda x: abs(x - insat_time_utc))
        time_diff = abs((imerg_closest_time - insat_time_utc).total_seconds())

        if min_time_diff is None or time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_file = imerg_file

    if closest_file:
        print(f"Matched INSAT {os.path.basename(insat_path)} with IMERG {closest_file}, diff {min_time_diff} sec")
    else:
        print(f"No IMERG match found for INSAT {os.path.basename(insat_path)}")

    return closest_file

# ----------------- FILE LISTING -----------------
insat_files = sorted([f for f in os.listdir(INSAT_DIR) if f.endswith('.h5')])

# ----------------- DATA STORAGE -----------------
BTD_all, Tb_all, R_all = [], [], []

# ----------------- MAIN PROCESSING LOOP -----------------
for insat_filename in tqdm(insat_files, desc="Processing INSAT files"):
    try:
        insat_path = os.path.join(INSAT_DIR, insat_filename)

        # Find closest IMERG file based on internal times
        closest_imerg_filename = find_closest_imerg_file_using_internal_time(insat_path, imerg_files)
        if closest_imerg_filename is None:
            continue  # Skip if no matching IMERG file found

        matched_imerg_path = os.path.join(IMERG_DIR, closest_imerg_filename)

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
        with h5py.File(matched_imerg_path, 'r') as f:
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
