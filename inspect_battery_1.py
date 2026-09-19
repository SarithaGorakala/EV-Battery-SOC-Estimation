from scipy.io import loadmat
import numpy as np
import pandas as pd
import os

# ============================================================
# 1. Load NASA battery file
# ============================================================

file_path = r"C:\Users\User\Desktop\SOC_Estimation\B0005.mat"

data = loadmat(file_path)

battery = data["B0005"]
cycles = battery["cycle"][0, 0]

print("Battery file loaded successfully.")

# ============================================================
# 2. Find discharge cycles
# ============================================================

discharge_cycles = []

for i in range(cycles.shape[1]):

    cycle_type = cycles["type"][0, i]

    if hasattr(cycle_type, "item"):
        cycle_type = cycle_type.item()

    if isinstance(cycle_type, bytes):
        cycle_type = cycle_type.decode()

    cycle_type = str(cycle_type)

    if cycle_type == "discharge":
        discharge_cycles.append(i)

print("Number of discharge cycles:", len(discharge_cycles))

# ============================================================
# 3. Extract measurements
# ============================================================

all_data = []

for cycle_number, i in enumerate(discharge_cycles, start=1):

    discharge = cycles["data"][0, i]

    # Extract each measurement
    voltage = discharge["Voltage_measured"][0, 0].flatten()
    current = discharge["Current_measured"][0, 0].flatten()
    temperature = discharge["Temperature_measured"][0, 0].flatten()
    time = discharge["Time"][0, 0].flatten()

    # Capacity is usually a single value for the whole discharge cycle
    capacity = discharge["Capacity"][0, 0].flatten()

    # Number of measurements
    n = min(
        len(voltage),
        len(current),
        len(temperature),
        len(time)
    )

    # Make sure all arrays have same length
    voltage = voltage[:n]
    current = current[:n]
    temperature = temperature[:n]
    time = time[:n]

    # Capacity is one value for the cycle
    if len(capacity) > 0:
        cycle_capacity = float(capacity[0])
    else:
        cycle_capacity = np.nan

    # Create rows
    for j in range(n):

        all_data.append([
            cycle_number,
            time[j],
            voltage[j],
            current[j],
            temperature[j],
            cycle_capacity
        ])

# ============================================================
# 4. Create DataFrame
# ============================================================

df = pd.DataFrame(
    all_data,
    columns=[
        "Cycle",
        "Time",
        "Voltage",
        "Current",
        "Temperature",
        "Capacity"
    ]
)

# ============================================================
# 5. Display information
# ============================================================

print("\nDataset created successfully!")

print("\nShape:")
print(df.shape)

print("\nFirst 10 rows:")
print(df.head(10))

print("\nColumn information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# ============================================================
# 6. Save CSV
# ============================================================

output_path = r"C:\Users\User\Desktop\SOC_Estimation\B0005_discharge_data_new.csv"

df.to_csv(output_path, index=False)

print("\nCSV saved successfully:")
print(output_path)