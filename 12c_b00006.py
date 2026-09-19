
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

project_path = r"C:\Users\User\Desktop\SOC_Project"

train_path = (
    project_path +
    r"\train_data.csv"
)

b0006_path = (
    project_path +
    r"\B0006_SOC_dataset.csv"
)

output_folder = (
    project_path +
    r"\Cross_Battery_Validation"
)

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# 2. LOAD B0005 TRAINING DATA
# ============================================================

train_df = pd.read_csv(
    train_path
)


# ============================================================
# 3. LOAD B0006 DATA
# ============================================================

b0006_df = pd.read_csv(
    b0006_path
)


print("B0005 training data:")
print(train_df.shape)

print("\nB0006 testing data:")
print(b0006_df.shape)


# ============================================================
# 4. FEATURES
# ============================================================

features = [
    "Voltage",
    "Current",
    "Temperature"
]


X_train = train_df[features]

y_train = train_df["SOC"]


X_b0006 = b0006_df[features]

y_b0006 = b0006_df["SOC"]


# ============================================================
# 5. CREATE RANDOM FOREST
# ============================================================

model = RandomForestRegressor(

    n_estimators=150,

    max_depth=15,

    min_samples_split=10,

    min_samples_leaf=5,

    random_state=42,

    n_jobs=-1
)


# ============================================================
# 6. TRAIN USING B0005
# ============================================================

print("\nTraining Random Forest using B0005...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 7. PREDICT B0006 SOC
# ============================================================

print("\nPredicting B0006 SOC...")

b0006_pred = model.predict(
    X_b0006
)

print("Prediction completed!")


# ============================================================
# 8. CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(
    y_b0006,
    b0006_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_b0006,
        b0006_pred
    )
)

r2 = r2_score(
    y_b0006,
    b0006_pred
)


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n======================================")
print("B0006 CROSS-BATTERY RESULTS")
print("======================================")

print(
    f"MAE  : {mae:.4f}%"
)

print(
    f"RMSE : {rmse:.4f}%"
)

print(
    f"R²   : {r2:.4f}"
)

print("======================================")


# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({

    "Cycle": b0006_df["Cycle"],

    "Time": b0006_df["Time"],

    "Actual_SOC": y_b0006,

    "Predicted_SOC": b0006_pred,

    "Error": y_b0006 - b0006_pred

})


results_path = (
    output_folder +
    r"\B0006_SOC_Predictions.csv"
)

results.to_csv(
    results_path,
    index=False
)


# ============================================================
# 11. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({

    "Training_Battery": ["B0005"],

    "Testing_Battery": ["B0006"],

    "Model": ["Random Forest"],

    "MAE": [mae],

    "RMSE": [rmse],

    "R2": [r2]

})


metrics_path = (
    output_folder +
    r"\B0005_to_B0006_Metrics.csv"
)

metrics.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# 12. FIGURE 38
# ACTUAL VS PREDICTED SOC
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    y_b0006.values,
    label="Actual SOC",
    linewidth=1.5
)

plt.plot(
    b0006_pred,
    label="Predicted SOC",
    linewidth=1.5
)

plt.xlabel(
    "B0006 Test Samples",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "SOC (%)",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Cross-Battery SOC Estimation: B0005 → B0006",
    fontsize=15,
    fontweight="bold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plot1 = (
    output_folder +
    r"\38_B0005_to_B0006_Actual_vs_Predicted.png"
)

plt.savefig(
    plot1,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 13. FIGURE 39
# SCATTER PLOT
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_b0006,
    b0006_pred,
    alpha=0.25,
    s=10
)


minimum = min(
    y_b0006.min(),
    b0006_pred.min()
)

maximum = max(
    y_b0006.max(),
    b0006_pred.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2
)


plt.xlabel(
    "Actual SOC (%)",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "Predicted SOC (%)",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "B0006 Cross-Battery Prediction",
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
    r"\39_B0006_Scatter.png"
)

plt.savefig(
    plot2,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 14. FIGURE 40
# PREDICTION ERROR
# ============================================================

error = (
    y_b0006.values -
    b0006_pred
)


plt.figure(figsize=(12, 5))

plt.plot(
    error,
    linewidth=1
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1.5
)

plt.xlabel(
    "B0006 Test Samples",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "Prediction Error (%)",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "B0006 SOC Prediction Error",
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
    r"\40_B0006_Prediction_Error.png"
)

plt.savefig(
    plot3,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 15. SAVE OUTPUT INFORMATION
# ============================================================

print("\n======================================")
print("STEP 12 COMPLETED")
print("======================================")

print("\nPrediction file:")
print(results_path)

print("\nMetrics file:")
print(metrics_path)

print("\nFigures:")
print(plot1)
print(plot2)
print(plot3)


# ============================================================
# 16. SHOW ALL FIGURES
# ============================================================

plt.show()