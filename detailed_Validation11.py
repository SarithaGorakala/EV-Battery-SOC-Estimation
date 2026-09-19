import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

project_path = r"C:\Users\User\Desktop\SOC_Project"

test_path = project_path + r"\test_data.csv"

rf_results_path = (
    project_path +
    r"\Random_Forest_Results\Random_Forest_SOC_Results.csv"
)

output_folder = (
    project_path +
    r"\Cycle_Wise_Validation"
)

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

test_df = pd.read_csv(test_path)

rf_results = pd.read_csv(rf_results_path)


print("Test data shape:")
print(test_df.shape)

print("\nRandom Forest results shape:")
print(rf_results.shape)


# ============================================================
# 3. CHECK COLUMNS
# ============================================================

print("\nRandom Forest result columns:")
print(rf_results.columns)


# ============================================================
# 4. CREATE CYCLE-WISE RESULTS
# ============================================================

cycle_results = []


unique_cycles = sorted(
    rf_results["Cycle"].unique()
)


print("\nNumber of test cycles:")
print(len(unique_cycles))


# ============================================================
# 5. CALCULATE METRICS FOR EACH CYCLE
# ============================================================

for cycle in unique_cycles:

    cycle_data = rf_results[
        rf_results["Cycle"] == cycle
    ]

    actual = cycle_data["Actual_SOC"].values

    predicted = cycle_data["Predicted_SOC"].values

    # MAE
    cycle_mae = mean_absolute_error(
        actual,
        predicted
    )

    # RMSE
    cycle_rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # R2
    # R2 cannot be calculated if actual SOC
    # has only one unique value

    if len(np.unique(actual)) > 1:

        cycle_r2 = r2_score(
            actual,
            predicted
        )

    else:

        cycle_r2 = np.nan


    cycle_results.append({

        "Cycle": cycle,

        "MAE": cycle_mae,

        "RMSE": cycle_rmse,

        "R2": cycle_r2,

        "Number_of_Samples": len(cycle_data)

    })


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

cycle_results_df = pd.DataFrame(
    cycle_results
)


print("\n================================================")
print("CYCLE-WISE PERFORMANCE")
print("================================================")

print(
    cycle_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 7. SAVE CYCLE-WISE RESULTS
# ============================================================

cycle_results_path = (
    output_folder +
    r"\Random_Forest_Cycle_Wise_Metrics.csv"
)

cycle_results_df.to_csv(
    cycle_results_path,
    index=False
)

print("\nCycle-wise results saved:")
print(cycle_results_path)


# ============================================================
# 8. OVERALL STATISTICS OF CYCLE-WISE METRICS
# ============================================================

mean_cycle_mae = cycle_results_df["MAE"].mean()

mean_cycle_rmse = cycle_results_df["RMSE"].mean()

mean_cycle_r2 = cycle_results_df["R2"].mean()


print("\n================================================")
print("AVERAGE OF CYCLE-WISE METRICS")
print("================================================")

print(
    f"Average Cycle MAE  : {mean_cycle_mae:.4f}"
)

print(
    f"Average Cycle RMSE : {mean_cycle_rmse:.4f}"
)

print(
    f"Average Cycle R²   : {mean_cycle_r2:.4f}"
)


# ============================================================
# 9. BEST AND WORST CYCLES
# ============================================================

best_mae_cycle = cycle_results_df.loc[
    cycle_results_df["MAE"].idxmin()
]

worst_mae_cycle = cycle_results_df.loc[
    cycle_results_df["MAE"].idxmax()
]


print("\n================================================")
print("CYCLE-WISE MAE RANGE")
print("================================================")

print(
    f"Lowest MAE cycle : "
    f"{int(best_mae_cycle['Cycle'])}"
)

print(
    f"Lowest MAE       : "
    f"{best_mae_cycle['MAE']:.4f}"
)

print(
    f"Highest MAE cycle: "
    f"{int(worst_mae_cycle['Cycle'])}"
)

print(
    f"Highest MAE      : "
    f"{worst_mae_cycle['MAE']:.4f}"
)


# ============================================================
# 10. FIGURE 34
# CYCLE-WISE MAE
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    cycle_results_df["Cycle"],
    cycle_results_df["MAE"],
    marker="o",
    markersize=3,
    linewidth=1.5
)

plt.xlabel(
    "Battery Discharge Cycle",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "MAE (%)",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Random Forest: Cycle-Wise MAE",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plot1 = (
    output_folder +
    r"\34_Cycle_Wise_MAE.png"
)

plt.savefig(
    plot1,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 11. FIGURE 35
# CYCLE-WISE RMSE
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    cycle_results_df["Cycle"],
    cycle_results_df["RMSE"],
    marker="o",
    markersize=3,
    linewidth=1.5
)

plt.xlabel(
    "Battery Discharge Cycle",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "RMSE (%)",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Random Forest: Cycle-Wise RMSE",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plot2 = (
    output_folder +
    r"\35_Cycle_Wise_RMSE.png"
)

plt.savefig(
    plot2,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 12. FIGURE 36
# CYCLE-WISE R2
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    cycle_results_df["Cycle"],
    cycle_results_df["R2"],
    marker="o",
    markersize=3,
    linewidth=1.5
)

plt.xlabel(
    "Battery Discharge Cycle",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "R²",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Random Forest: Cycle-Wise R²",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plot3 = (
    output_folder +
    r"\36_Cycle_Wise_R2.png"
)

plt.savefig(
    plot3,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 13. FIGURE 37
# ACTUAL VS PREDICTED SOC FOR TEST CYCLES
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    rf_results["Actual_SOC"].values,
    label="Actual SOC",
    linewidth=1.5
)

plt.plot(
    rf_results["Predicted_SOC"].values,
    label="Predicted SOC",
    linewidth=1.5
)

plt.xlabel(
    "Test Samples",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "SOC (%)",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Random Forest: SOC Validation on Unseen Test Cycles",
    fontsize=15,
    fontweight="bold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plot4 = (
    output_folder +
    r"\37_Test_Cycles_Actual_vs_Predicted_SOC.png"
)

plt.savefig(
    plot4,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 14. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame({

    "Metric": [
        "Average Cycle MAE",
        "Average Cycle RMSE",
        "Average Cycle R2",
        "Minimum Cycle MAE",
        "Maximum Cycle MAE"
    ],

    "Value": [
        mean_cycle_mae,
        mean_cycle_rmse,
        mean_cycle_r2,
        cycle_results_df["MAE"].min(),
        cycle_results_df["MAE"].max()
    ]

})


summary_path = (
    output_folder +
    r"\Cycle_Wise_Validation_Summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# 15. PRINT OUTPUT FILES
# ============================================================

print("\n================================================")
print("STEP 11 COMPLETED")
print("================================================")

print(cycle_results_path)
print(summary_path)

print(plot1)
print(plot2)
print(plot3)
print(plot4)


# ============================================================
# 16. SHOW ALL FIGURES
# ============================================================

plt.show()