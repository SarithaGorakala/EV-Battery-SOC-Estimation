
import scipy.io
import pandas as pd
import numpy as np


# ============================================================
# 1. PATH
# ============================================================

mat_path = r"C:\Users\User\Desktop\SOC_Project\B0006.mat"

output_path = (
    r"C:\Users\User\Desktop\SOC_Project\B0006_discharge_data.csv"
)


# ============================================================
# 2. LOAD MAT FILE
# ============================================================

data = scipy.io.loadmat(
    mat_path,
    squeeze_me=False,
    struct_as_record=True
)

print("MAT file loaded.")

print("\nKeys:")
print(data.keys())


# ============================================================
# 3. GET BATTERY
# ============================================================

battery = data["B0006"]

print("\nBattery shape:")
print(battery.shape)

print("\nBattery dtype:")
print(battery.dtype)


# ============================================================
# 4. GET CYCLES
# ============================================================

cycles = battery["cycle"][0, 0]

print("\nNumber of cycles:")
print(cycles.shape[1])

print("\nCycle fields:")
print(cycles.dtype.names)


# ============================================================
# 5. EXTRACT DISCHARGE CYCLES
# ============================================================

rows = []

discharge_cycle_number = 0


for i in range(cycles.shape[1]):

    cycle = cycles[0, i]

    cycle_type = cycle["type"][0]

    if isinstance(cycle_type, np.ndarray):
        cycle_type = cycle_type.item()

    cycle_type = str(cycle_type).lower()


    # Only discharge cycles
    if cycle_type != "discharge":
        continue


    discharge_cycle_number += 1


    # Get cycle data
    cycle_data = cycle["data"][0, 0]


    # Extract variables
    voltage = np.asarray(
        cycle_data["Voltage_measured"][0, 0]
    ).flatten()

    current = np.asarray(
        cycle_data["Current_measured"][0, 0]
    ).flatten()

    temperature = np.asarray(
        cycle_data["Temperature_measured"][0, 0]
    ).flatten()

    time = np.asarray(
        cycle_data["Time"][0, 0]
    ).flatten()

    capacity = np.asarray(
        cycle_data["Capacity"][0, 0]
    ).flatten()


    # Capacity is normally a single value
    if len(capacity) > 0:
        capacity_value = float(capacity[0])
    else:
        capacity_value = np.nan


    # Store rows
    for j in range(len(time)):

        rows.append({

            "Cycle": discharge_cycle_number,

            "Time": time[j],

            "Voltage": voltage[j],

            "Current": current[j],

            "Temperature": temperature[j],

            "Capacity": capacity_value

        })


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(rows)


# ============================================================
# 7. BASIC CLEANING
# ============================================================

df = df.dropna()

df = df.sort_values(
    ["Cycle", "Time"]
)

df = df.reset_index(drop=True)


# ============================================================
# 8. SAVE
# ============================================================

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 9. PRINT INFORMATION
# ============================================================

print("\n======================================")
print("B0006 EXTRACTION COMPLETED")
print("======================================")

print("Rows:", len(df))

print("Columns:")
print(df.columns.tolist())

print("\nNumber of discharge cycles:")
print(df["Cycle"].nunique())

print("\nFirst 5 rows:")
print(df.head())

print("\nSaved to:")
print(output_path)