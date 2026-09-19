import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. LOAD SOC DATASET
# ============================================================

file_path = r"C:\Users\User\Desktop\SOC_Project\B0005_SOC_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC DATA CHECK
# ============================================================

print("\n==============================")
print("DATA CHECK")
print("==============================")

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nSOC statistics:")
print(df["SOC"].describe())


# ============================================================
# 3. CHECK SOC RANGE
# ============================================================

print("\n==============================")
print("SOC RANGE CHECK")
print("==============================")

print("Minimum SOC:", df["SOC"].min())
print("Maximum SOC:", df["SOC"].max())

if df["SOC"].min() >= 0 and df["SOC"].max() <= 100:
    print("SOC is within 0-100%.")
else:
    print("WARNING: SOC outside 0-100%.")


# ============================================================
# 4. CHECK SOC AT START AND END OF CYCLES
# ============================================================

cycle_summary = df.groupby("Cycle").agg(
    Start_SOC=("SOC", "first"),
    End_SOC=("SOC", "last"),
    Capacity=("Capacity", "first"),
    Voltage_Start=("Voltage", "first"),
    Voltage_End=("Voltage", "last"),
    Temperature_Start=("Temperature", "first"),
    Temperature_End=("Temperature", "last")
).reset_index()

print("\nFirst 10 cycles:")
print(cycle_summary.head(10))


# ============================================================
# 5. SAVE CYCLE SUMMARY
# ============================================================

output_folder = r"C:\Users\User\Desktop\SOC_Project"

os.makedirs(output_folder, exist_ok=True)

cycle_summary_file = os.path.join(
    output_folder,
    "cycle_summary.csv"
)

cycle_summary.to_csv(
    cycle_summary_file,
    index=False
)

print("\nCycle summary saved:")
print(cycle_summary_file)


# ============================================================
# 6. PLOT 1 - SOC VS TIME
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
    "SOC vs Time - Cycle 1",
    fontsize=16,
    fontweight="bold"
)

plt.grid(True, alpha=0.3)

plt.xticks(
    fontsize=11,
    fontweight="bold"
)

plt.yticks(
    fontsize=11,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "08_SOC_Validation_Cycle1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 7. PLOT 2 - START SOC AND END SOC
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    cycle_summary["Cycle"],
    cycle_summary["Start_SOC"],
    linewidth=2,
    label="Start SOC"
)

plt.plot(
    cycle_summary["Cycle"],
    cycle_summary["End_SOC"],
    linewidth=2,
    label="End SOC"
)

plt.xlabel(
    "Cycle Number",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "SOC (%)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "SOC at Start and End of Each Cycle",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.grid(True, alpha=0.3)

plt.xticks(
    fontsize=11,
    fontweight="bold"
)

plt.yticks(
    fontsize=11,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "09_Start_End_SOC.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 8. PLOT 3 - CAPACITY DEGRADATION
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    cycle_summary["Cycle"],
    cycle_summary["Capacity"],
    linewidth=2
)

plt.xlabel(
    "Cycle Number",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Capacity (Ah)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Battery Capacity Degradation",
    fontsize=16,
    fontweight="bold"
)

plt.grid(True, alpha=0.3)

plt.xticks(
    fontsize=11,
    fontweight="bold"
)

plt.yticks(
    fontsize=11,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "10_Capacity_Degradation.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 9. PLOT 4 - VOLTAGE VS SOC
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["SOC"],
    df["Voltage"],
    s=3,
    alpha=0.3
)

plt.xlabel(
    "SOC (%)",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Voltage (V)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Battery Voltage vs SOC",
    fontsize=16,
    fontweight="bold"
)

plt.grid(True, alpha=0.3)

plt.xticks(
    fontsize=11,
    fontweight="bold"
)

plt.yticks(
    fontsize=11,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "11_Voltage_vs_SOC.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 10. PLOT 5 - TEMPERATURE VS SOC
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["SOC"],
    df["Temperature"],
    s=3,
    alpha=0.3
)

plt.xlabel(
    "SOC (%)",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Temperature (°C)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Battery Temperature vs SOC",
    fontsize=16,
    fontweight="bold"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "12_Temperature_vs_SOC.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 11. PLOT 6 - CURRENT VS SOC
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["SOC"],
    df["Current"],
    s=3,
    alpha=0.3
)

plt.xlabel(
    "SOC (%)",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Current (A)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Battery Current vs SOC",
    fontsize=16,
    fontweight="bold"
)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_folder,
        "13_Current_vs_SOC.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 12. CREATE ML DATASET
# ============================================================

ml_df = df[
    [
        "Cycle",
        "Time",
        "Voltage",
        "Current",
        "Temperature",
        "SOC"
    ]
].copy()


# ============================================================
# 13. REMOVE INVALID VALUES
# ============================================================

ml_df = ml_df.replace(
    [np.inf, -np.inf],
    np.nan
)

ml_df = ml_df.dropna()


# ============================================================
# 14. SAVE ML DATASET
# ============================================================

ml_file = os.path.join(
    output_folder,
    "B0005_ML_dataset.csv"
)

ml_df.to_csv(
    ml_file,
    index=False
)

print("\n==========================================")
print("ML DATASET CREATED")
print("==========================================")

print("Shape:", ml_df.shape)

print("\nFeatures:")
print([
    "Voltage",
    "Current",
    "Temperature"
])

print("\nTarget:")
print("SOC")

print("\nSaved here:")
print(ml_file)


# ============================================================
# 15. SHOW ALL PLOTS AT ONCE
# ============================================================

plt.show()


# ============================================================
# 16. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("STEP 4 COMPLETED")
print("==========================================")

print("\nSaved plots:")

print("08_SOC_Validation_Cycle1.png")
print("09_Start_End_SOC.png")
print("10_Capacity_Degradation.png")
print("11_Voltage_vs_SOC.png")
print("12_Temperature_vs_SOC.png")
print("13_Current_vs_SOC.png")

print("\nML dataset:")
print("B0005_ML_dataset.csv")