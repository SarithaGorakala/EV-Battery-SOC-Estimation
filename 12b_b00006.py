# ============================================================
# STEP 12B: CONSTRUCT SOC FOR B0006
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS
# ============================================================

input_path = (
    r"C:\Users\User\Desktop\SOC_Project\B0006_discharge_data.csv"
)

output_path = (
    r"C:\Users\User\Desktop\SOC_Project\B0006_SOC_dataset.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_csv(input_path)

df = df.sort_values(
    ["Cycle", "Time"]
).reset_index(drop=True)


# ============================================================
# 3. CREATE SOC COLUMN
# ============================================================

df["SOC"] = np.nan


# ============================================================
# 4. CALCULATE SOC CYCLE BY CYCLE
# ============================================================

for cycle in df["Cycle"].unique():

    mask = df["Cycle"] == cycle

    cycle_data = df.loc[mask].copy()

    time = cycle_data["Time"].values

    current = cycle_data["Current"].values

    capacity = cycle_data["Capacity"].iloc[0]


    # --------------------------------------------------------
    # Convert discharge current to positive current
    # NASA discharge current is generally negative
    # --------------------------------------------------------

    discharge_current = -current


    # Avoid negative discharge current
    discharge_current = np.maximum(
        discharge_current,
        0
    )


    # --------------------------------------------------------
    # Time difference in hours
    # --------------------------------------------------------

    dt = np.diff(time) / 3600.0


    # --------------------------------------------------------
    # Coulomb counting using trapezoidal integration
    # --------------------------------------------------------

    charge_used = np.zeros(
        len(time)
    )


    if len(time) > 1:

        charge_increment = (
            (discharge_current[:-1] +
             discharge_current[1:]) / 2
        ) * dt


        charge_used[1:] = np.cumsum(
            charge_increment
        )


    # --------------------------------------------------------
    # SOC calculation
    # --------------------------------------------------------

    if pd.notna(capacity) and capacity > 0:

        soc = (
            1 -
            charge_used / capacity
        ) * 100

    else:

        soc = np.full(
            len(time),
            np.nan
        )


    # Keep SOC between 0 and 100
    soc = np.clip(
        soc,
        0,
        100
    )


    df.loc[mask, "SOC"] = soc


# ============================================================
# 5. REMOVE INVALID ROWS
# ============================================================

df = df.dropna(
    subset=[
        "Voltage",
        "Current",
        "Temperature",
        "SOC"
    ]
)


# ============================================================
# 6. SAVE
# ============================================================

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 7. PRINT INFORMATION
# ============================================================

print("\n======================================")
print("B0006 SOC CALCULATION COMPLETED")
print("======================================")

print("Rows:", len(df))

print("Cycles:", df["Cycle"].nunique())

print(
    f"SOC minimum: {df['SOC'].min():.2f}%"
)

print(
    f"SOC maximum: {df['SOC'].max():.2f}%"
)

print("\nSaved to:")
print(output_path)


# ============================================================
# 8. PLOT SOC
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["SOC"].values,
    linewidth=1
)

plt.xlabel(
    "Samples",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "SOC (%)",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "B0006 Constructed SOC",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.show()