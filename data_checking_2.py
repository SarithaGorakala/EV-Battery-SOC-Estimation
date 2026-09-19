import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

file_path = r"C:\Users\User\Desktop\SOC_Estimation\B0005_discharge_data.csv"

df = pd.read_csv(file_path)

print("Data loaded successfully!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# 2. CREATE FOLDER FOR SAVING PLOTS
# ============================================================

plot_folder = r"C:\Users\User\Desktop\SOC_Project_Plots"

os.makedirs(plot_folder, exist_ok=True)

print("\nPlots will be saved in:")
print(plot_folder)


# ============================================================
# 3. PLOT 1 - BATTERY CAPACITY VS CYCLE
# ============================================================

capacity_data = df.groupby("Cycle")["Capacity"].first()

plt.figure(figsize=(10, 6))

plt.plot(
    capacity_data.index,
    capacity_data.values,
    linewidth=2
)

plt.xlabel("Cycle Number", fontsize=14, fontweight="bold")
plt.ylabel("Capacity (Ah)", fontsize=14, fontweight="bold")
plt.title("Battery Capacity vs Cycle Number", fontsize=16, fontweight="bold")

plt.xticks(fontsize=11, fontweight="bold")
plt.yticks(fontsize=11, fontweight="bold")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(plot_folder, "01_Capacity_vs_Cycle.png"),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 4. PLOT 2 - VOLTAGE VS TIME
# ============================================================

cycle_number = 1

cycle_data = df[df["Cycle"] == cycle_number]

plt.figure(figsize=(10, 6))

plt.plot(
    cycle_data["Time"],
    cycle_data["Voltage"],
    linewidth=2
)

plt.xlabel("Time (seconds)", fontsize=14, fontweight="bold")
plt.ylabel("Voltage (V)", fontsize=14, fontweight="bold")
plt.title(
    f"Battery Voltage vs Time - Cycle {cycle_number}",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(fontsize=11, fontweight="bold")
plt.yticks(fontsize=11, fontweight="bold")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(plot_folder, "02_Voltage_vs_Time_Cycle1.png"),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 5. PLOT 3 - CURRENT VS TIME
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    cycle_data["Time"],
    cycle_data["Current"],
    linewidth=2
)

plt.xlabel("Time (seconds)", fontsize=14, fontweight="bold")
plt.ylabel("Current (A)", fontsize=14, fontweight="bold")
plt.title(
    f"Battery Current vs Time - Cycle {cycle_number}",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(fontsize=11, fontweight="bold")
plt.yticks(fontsize=11, fontweight="bold")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(plot_folder, "03_Current_vs_Time_Cycle1.png"),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 6. PLOT 4 - TEMPERATURE VS TIME
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    cycle_data["Time"],
    cycle_data["Temperature"],
    linewidth=2
)

plt.xlabel("Time (seconds)", fontsize=14, fontweight="bold")
plt.ylabel("Temperature (°C)", fontsize=14, fontweight="bold")
plt.title(
    f"Battery Temperature vs Time - Cycle {cycle_number}",
    fontsize=16,
    fontweight="bold"
)

plt.xticks(fontsize=11, fontweight="bold")
plt.yticks(fontsize=11, fontweight="bold")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    os.path.join(plot_folder, "04_Temperature_vs_Time_Cycle1.png"),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 7. SHOW ALL PLOTS AT THE SAME TIME
# ============================================================

plt.show()


# ============================================================
# 8. FINISHED
# ============================================================

print("\n==========================================")
print("ALL PLOTS CREATED SUCCESSFULLY")
print("==========================================")

print("\nSaved files:")

print("1. 01_Capacity_vs_Cycle.png")
print("2. 02_Voltage_vs_Time_Cycle1.png")
print("3. 03_Current_vs_Time_Cycle1.png")
print("4. 04_Temperature_vs_Time_Cycle1.png")

print("\nLocation:")
print(plot_folder)