# ============================================================
# STEP 10: COMPARISON OF ALL ML MODELS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# ============================================================
# 1. PROJECT PATH
# ============================================================

project_path = r"C:\Users\User\Desktop\SOC_Project"

output_folder = project_path + r"\Model_Comparison"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 2. LOAD METRICS FROM EACH MODEL
# ============================================================

# Linear Regression
linear_path = project_path + r"\model_results.csv"

# Decision Tree
decision_tree_path = (
    r"C:\Users\User\Desktop\SOC_Project\decision_tree_results.csv"
)

# Random Forest
random_forest_path = (
    project_path +
    r"\Random_Forest_Results\Random_Forest_Metrics.csv"
)

# Gradient Boosting
gradient_boosting_path = (
    project_path +
    r"\Gradient_Boosting_Results\Gradient_Boosting_Metrics.csv"
)


# ============================================================
# 3. READ FILES
# ============================================================

print("Loading model results...")


# Linear Regression
linear_df = pd.read_csv(linear_path)

print("\nLinear Regression:")
print(linear_df)


# Decision Tree
decision_tree_df = pd.read_csv(decision_tree_path)

print("\nDecision Tree:")
print(decision_tree_df)


# Random Forest
random_forest_df = pd.read_csv(random_forest_path)

print("\nRandom Forest:")
print(random_forest_df)


# Gradient Boosting
gradient_boosting_df = pd.read_csv(gradient_boosting_path)

print("\nGradient Boosting:")
print(gradient_boosting_df)


# ============================================================
# 4. EXTRACT METRICS
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting"
    ],

    "MAE": [
        linear_df["MAE"].iloc[0],
        decision_tree_df["MAE"].iloc[0],
        random_forest_df["MAE"].iloc[0],
        gradient_boosting_df["MAE"].iloc[0]
    ],

    "RMSE": [
        linear_df["RMSE"].iloc[0],
        decision_tree_df["RMSE"].iloc[0],
        random_forest_df["RMSE"].iloc[0],
        gradient_boosting_df["RMSE"].iloc[0]
    ],

    "R2": [
        linear_df["R2"].iloc[0],
        decision_tree_df["R2"].iloc[0],
        random_forest_df["R2"].iloc[0],
        gradient_boosting_df["R2"].iloc[0]
    ]
})


# ============================================================
# 5. DISPLAY COMPARISON TABLE
# ============================================================

print("\n")
print("================================================")
print("MODEL PERFORMANCE COMPARISON")
print("================================================")

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("================================================")


# ============================================================
# 6. SAVE COMPARISON TABLE
# ============================================================

comparison_path = (
    output_folder +
    r"\All_Model_Comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)

print("\nComparison table saved:")
print(comparison_path)


# ============================================================
# 7. FIND BEST MODEL FOR EACH METRIC
# ============================================================

# Lower MAE is better
best_mae_model = comparison.loc[
    comparison["MAE"].idxmin(),
    "Model"
]

# Lower RMSE is better
best_rmse_model = comparison.loc[
    comparison["RMSE"].idxmin(),
    "Model"
]

# Higher R2 is better
best_r2_model = comparison.loc[
    comparison["R2"].idxmax(),
    "Model"
]


print("\n================================================")
print("METRIC SUMMARY")
print("================================================")

print("Lowest MAE  :", best_mae_model)
print("Lowest RMSE :", best_rmse_model)
print("Highest R²  :", best_r2_model)

print("================================================")


# ============================================================
# 8. FIGURE 30 — MAE COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["MAE"]
)

plt.xlabel(
    "Machine Learning Model",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "MAE",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Comparison of MAE for SOC Estimation",
    fontsize=15,
    fontweight="bold"
)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


mae_plot = (
    output_folder +
    r"\30_MAE_Comparison.png"
)

plt.savefig(
    mae_plot,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 9. FIGURE 31 — RMSE COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["RMSE"]
)

plt.xlabel(
    "Machine Learning Model",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "RMSE",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Comparison of RMSE for SOC Estimation",
    fontsize=15,
    fontweight="bold"
)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


rmse_plot = (
    output_folder +
    r"\31_RMSE_Comparison.png"
)

plt.savefig(
    rmse_plot,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 10. FIGURE 32 — R² COMPARISON
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(
    comparison["Model"],
    comparison["R2"]
)

plt.xlabel(
    "Machine Learning Model",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "R²",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Comparison of R² for SOC Estimation",
    fontsize=15,
    fontweight="bold"
)

plt.xticks(
    rotation=15
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


r2_plot = (
    output_folder +
    r"\32_R2_Comparison.png"
)

plt.savefig(
    r2_plot,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 11. FIGURE 33 — ALL METRICS
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5)
)


# MAE

axes[0].bar(
    comparison["Model"],
    comparison["MAE"]
)

axes[0].set_title(
    "MAE",
    fontsize=14,
    fontweight="bold"
)

axes[0].set_ylabel(
    "MAE",
    fontsize=12,
    fontweight="bold"
)

axes[0].tick_params(
    axis="x",
    rotation=20
)

axes[0].grid(
    axis="y",
    alpha=0.3
)


# RMSE

axes[1].bar(
    comparison["Model"],
    comparison["RMSE"]
)

axes[1].set_title(
    "RMSE",
    fontsize=14,
    fontweight="bold"
)

axes[1].set_ylabel(
    "RMSE",
    fontsize=12,
    fontweight="bold"
)

axes[1].tick_params(
    axis="x",
    rotation=20
)

axes[1].grid(
    axis="y",
    alpha=0.3
)


# R2

axes[2].bar(
    comparison["Model"],
    comparison["R2"]
)

axes[2].set_title(
    "R²",
    fontsize=14,
    fontweight="bold"
)

axes[2].set_ylabel(
    "R²",
    fontsize=12,
    fontweight="bold"
)

axes[2].tick_params(
    axis="x",
    rotation=20
)

axes[2].grid(
    axis="y",
    alpha=0.3
)


plt.tight_layout()


all_metrics_plot = (
    output_folder +
    r"\33_All_Metrics_Comparison.png"
)

plt.savefig(
    all_metrics_plot,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 12. PRINT OUTPUT FILES
# ============================================================

print("\n================================================")
print("FILES SAVED")
print("================================================")

print(comparison_path)

print(mae_plot)
print(rmse_plot)
print(r2_plot)
print(all_metrics_plot)


# ============================================================
# 13. SHOW ALL FIGURES
# ============================================================

plt.show()