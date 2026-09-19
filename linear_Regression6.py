import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD TRAINING AND TESTING DATA
# ============================================================

project_folder = r"C:\Users\User\Desktop\SOC_Project"

train_file = os.path.join(
    project_folder,
    "train_data.csv"
)

test_file = os.path.join(
    project_folder,
    "test_data.csv"
)

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)

print("Training data:", train_df.shape)
print("Testing data:", test_df.shape)


# ============================================================
# 2. SELECT INPUT FEATURES
# ============================================================

features = [
    "Voltage",
    "Current",
    "Temperature"
]

X_train = train_df[features]
X_test = test_df[features]


# ============================================================
# 3. SELECT TARGET
# ============================================================

y_train = train_df["SOC"]
y_test = test_df["SOC"]


print("\nInput features:")
print(features)

print("\nTarget:")
print("SOC")


# ============================================================
# 4. CREATE LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining Linear Regression model...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 6. PREDICT SOC
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. CALCULATE PERFORMANCE METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("LINEAR REGRESSION RESULTS")
print("==========================================")

print(f"MAE  : {mae:.4f} %")
print(f"RMSE : {rmse:.4f} %")
print(f"R²   : {r2:.4f}")


# ============================================================
# 9. DISPLAY MODEL EQUATION
# ============================================================

print("\n==========================================")
print("MODEL COEFFICIENTS")
print("==========================================")

print(
    "SOC = "
    f"{model.intercept_:.4f}"
    " + "
    f"({model.coef_[0]:.4f} × Voltage)"
    " + "
    f"({model.coef_[1]:.4f} × Current)"
    " + "
    f"({model.coef_[2]:.4f} × Temperature)"
)


# ============================================================
# 10. CREATE RESULTS DATAFRAME
# ============================================================

results = test_df[
    [
        "Cycle",
        "Time",
        "Voltage",
        "Current",
        "Temperature",
        "SOC"
    ]
].copy()

results["Predicted_SOC"] = y_pred

results["SOC_Error"] = (
    results["SOC"]
    - results["Predicted_SOC"]
)


# ============================================================
# 11. SAVE PREDICTION RESULTS
# ============================================================

results_file = os.path.join(
    project_folder,
    "Linear_Regression_SOC_Results.csv"
)

results.to_csv(
    results_file,
    index=False
)

print("\nPrediction results saved:")
print(results_file)


# ============================================================
# 12. PLOT 1 - ACTUAL VS PREDICTED SOC
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    y_test.values,
    linewidth=1.5,
    label="Actual SOC"
)

plt.plot(
    y_pred,
    linewidth=1.5,
    label="Predicted SOC"
)

plt.xlabel(
    "Test Sample",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "SOC (%)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Actual vs Predicted SOC - Linear Regression",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

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
        project_folder,
        "15_Actual_vs_Predicted_SOC_Linear_Regression.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 13. PLOT 2 - ACTUAL SOC VS PREDICTED SOC
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred,
    s=8,
    alpha=0.4
)

# Ideal prediction line
min_value = min(
    y_test.min(),
    y_pred.min()
)

max_value = max(
    y_test.max(),
    y_pred.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linewidth=2
)

plt.xlabel(
    "Actual SOC (%)",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Predicted SOC (%)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Actual SOC vs Predicted SOC",
    fontsize=16,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.3
)

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
        project_folder,
        "16_Actual_vs_Predicted_Scatter_Linear_Regression.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 14. PLOT 3 - PREDICTION ERROR
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    results["SOC_Error"].values,
    linewidth=1
)

plt.axhline(
    0,
    linewidth=2
)

plt.xlabel(
    "Test Sample",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "SOC Error (%)",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "SOC Prediction Error - Linear Regression",
    fontsize=16,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.3
)

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
        project_folder,
        "17_SOC_Prediction_Error_Linear_Regression.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 15. SAVE MODEL PERFORMANCE
# ============================================================

metrics = pd.DataFrame({
    "Model": ["Linear Regression"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

metrics_file = os.path.join(
    project_folder,
    "model_results.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# 16. SHOW ALL PLOTS
# ============================================================

plt.show()


# ============================================================
# 17. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("STEP 6 COMPLETED")
print("==========================================")

print("\nSaved files:")

print("Linear_Regression_SOC_Results.csv")
print("model_results.csv")

print("\nSaved plots:")

print("15_Actual_vs_Predicted_SOC_Linear_Regression.png")
print("16_Actual_vs_Predicted_Scatter_Linear_Regression.png")
print("17_SOC_Prediction_Error_Linear_Regression.png")

print("\nNext step:")
print("Decision Tree Regression")