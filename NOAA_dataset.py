# time from epoch and dataset conataining TPW
import os
from datetime import datetime, timezone
import xarray as xr

# Define the GPS epoch
gps_epoch = datetime(1980, 1, 6, 0, 0, 0, tzinfo=timezone.utc)

# Folder containing your .nc files
folder_path = "C:/Users/adity/Desktop/Data_Set/noaa tpw/Blended-Hydro_TPW_MAP_d20250528"

# Get all .nc files and sort them alphabetically
nc_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".nc")])

# List to store TPW datasets
tpw_datasets = []

# Process each sorted file
for filename in nc_files:
    nc_file_path = os.path.join(folder_path, filename)

    # Extract the timestamp part starting with 's'
    parts = filename.split('_')
    try:
        start_str = next(part for part in parts if part.startswith('s'))[1:]  # remove 's'
    except StopIteration:
        print(f"Skipping file (no timestamp): {filename}")
        continue

    # Parse the date and time
    date_part = start_str[:8]       # YYYYMMDD
    time_part = start_str[8:]       # HHMMSS + 1 digit tenth of second

    # Extract time components
    hour = int(time_part[:2])
    minute = int(time_part[2:4])
    second = int(time_part[4:6])
    tenth_second = int(time_part[6]) / 10.0

    # Construct the datetime object (UTC)
    timestamp = datetime(
        year=int(date_part[:4]),
        month=int(date_part[4:6]),
        day=int(date_part[6:8]),
        hour=hour,
        minute=minute,
        second=second,
        microsecond=int(tenth_second * 1e6),
        tzinfo=timezone.utc
    )

    # Calculate seconds since GPS epoch
    seconds_since_gps_epoch = int((timestamp - gps_epoch).total_seconds())

    print(f"{filename} → Seconds since GPS epoch: {seconds_since_gps_epoch}")

    # Load dataset and extract TPW variable
    try:
        ds = xr.open_dataset(nc_file_path, decode_timedelta= True)
        if 'TPW' in ds.variables:
            tpw_datasets.append(ds['TPW'])  # Keep only the TPW variable
        else:
            print(f"'TPW' variable not found in {filename}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Optionally combine TPW variables into one xarray DataArray or Dataset
# combined_tpw = xr.concat(tpw_datasets, dim='time')  # Requires time dim alignment

# Uncomment to inspect the first TPW variable
print(tpw_datasets[0])
