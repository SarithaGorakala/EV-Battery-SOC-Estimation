import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = r"C:\Users\User\Desktop\SOC_Estimation\B0005_discharge_data.csv"

df = pd.read_csv(file_path)

print("Data loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

output_folder = r"C:\Users\User\Desktop\SOC_Project"

plot_folder = os.path.join(output_folder, "plots")

os.makedirs(plot_folder, exist_ok=True)


# ============================================================
# 3. CALCULATE SOC USING COULOMB COUNTING
# ============================================================

df["SOC"] = np.nan

cycles = df["Cycle"].unique()

print("\nCalculating SOC...")

for cycle in cycles:

    # Get one discharge cycle
    cycle_data = df[df["Cycle"] == cycle].copy()

    # Sort according to time
    cycle_data = cycle_data.sort_values("Time")

    # Original index
    original_index = cycle_data.index

    # Time in hours
    time_hours = cycle_data["Time"].values / 3600.0

    # NASA discharge current is negative.
    # Convert it into positive discharge current.
    current = -cycle_data["Current"].values

    # Cycle capacity
    capacity = cycle_data["Capacity"].iloc[0]

    # --------------------------------------------------------
    # Coulomb counting
    # --------------------------------------------------------

    # Calculate time difference
    dt = np.diff(time_hours)

    # Trapezoidal integration
    charge_used = np.zeros(len(cycle_data))

    for i in range(1, len(cycle_data)):

        charge_used[i] = (
            charge_used[i - 1]
            + 0.5
            * (current[i] + current[i - 1])
            * dt[i - 1]
        )

    # --------------------------------------------------------
    # SOC calculation
    # --------------------------------------------------------

    soc = 100 * (1 - charge_used / capacity)

    # Keep SOC between 0 and 100
    soc = np.clip(soc, 0, 100)

    # Put SOC back into original dataframe
    df.loc[original_index, "SOC"] = soc


print("SOC calculation completed!")


# ============================================================
# 4. CHECK SOC VALUES
# ============================================================

print("\nSOC information:")

print(df["SOC"].describe())


# ============================================================
# 5. DISPLAY FIRST ROWS
# ============================================================

print("\nFirst 20 rows:")

print(
    df[
        [
            "Cycle",
            "Time",
            "Voltage",
            "Current",
            "Temperature",
            "Capacity",
            "SOC"
        ]
    ].head(20)
)


# ============================================================
# 6. PLOT SOC FOR CYCLE 1
# ============================================================

cycle_number = 1

cycle1 = df[df["Cycle"] == cycle_number]

plt.figure(figsize=(10, 6))

plt.plot(
    cycle1["Time"],
    cycle1["SOC"],
    linewidth=2
)

plt.xlabel(
    "Time (seconds)",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "SOC (%)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Battery SOC vs Time - Cycle 1",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(
    fontsize=11,
    fontweight="bold"
)

plt.yticks(
    fontsize=11,
    fontweight="bold"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        plot_folder,
        "05_SOC_vs_Time_Cycle1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 7. PLOT VOLTAGE AND SOC TOGETHER
# ============================================================

fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(
    cycle1["Time"],
    cycle1["Voltage"],
    linewidth=2
)

ax1.set_xlabel(
    "Time (seconds)",
    fontsize=14,
    fontweight="bold"
)

ax1.set_ylabel(
    "Voltage (V)",
    fontsize=14,
    fontweight="bold"
)

ax1.tick_params(axis="both", labelsize=11)

ax2 = ax1.twinx()

ax2.plot(
    cycle1["Time"],
    cycle1["SOC"],
    linewidth=2
)

ax2.set_ylabel(
    "SOC (%)",
    fontsize=14,
    fontweight="bold"
)

ax2.tick_params(axis="both", labelsize=11)

plt.title(
    "Voltage and SOC vs Time - Cycle 1",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        plot_folder,
        "06_Voltage_and_SOC_Cycle1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 8. PLOT SOC FOR MULTIPLE CYCLES
# ============================================================

selected_cycles = [1, 50, 100, 150]

plt.figure(figsize=(10, 6))

for cycle_number in selected_cycles:

    cycle_data = df[df["Cycle"] == cycle_number]

    if len(cycle_data) > 0:

        plt.plot(
            cycle_data["Time"],
            cycle_data["SOC"],
            linewidth=2,
            label=f"Cycle {cycle_number}"
        )

plt.xlabel(
    "Time (seconds)",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "SOC (%)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "SOC vs Time for Different Battery Cycles",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.xticks(
    fontsize=11,
    fontweight="bold"
)

plt.yticks(
    fontsize=11,
    fontweight="bold"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        plot_folder,
        "07_SOC_Multiple_Cycles.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 9. SAVE NEW DATASET
# ============================================================

output_csv = os.path.join(
    output_folder,
    "B0005_SOC_dataset.csv"
)

df.to_csv(
    output_csv,
    index=False
)


# ============================================================
# 10. SHOW ALL FIGURES AT ONCE
# ============================================================

plt.show()


# ============================================================
# 11. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("STEP 3 COMPLETED")
print("==========================================")

print("\nNew dataset:")
print(output_csv)

print("\nSaved plots:")

print("05_SOC_vs_Time_Cycle1.png")
print("06_Voltage_and_SOC_Cycle1.png")
print("07_SOC_Multiple_Cycles.png")

print("\nPlot folder:")
print(plot_folder)