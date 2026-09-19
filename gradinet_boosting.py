# ============================================================
# STEP 9: GRADIENT BOOSTING REGRESSION FOR SOC ESTIMATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PATHS
# ============================================================

project_path = r"C:\Users\User\Desktop\SOC_Project"

train_path = project_path + r"\train_data.csv"
test_path = project_path + r"\test_data.csv"

output_folder = project_path + r"\Gradient_Boosting_Results"

os.makedirs(output_folder, exist_ok=True)


# ============================================================
# 2. LOAD TRAIN AND TEST DATA
# ============================================================

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train data shape:", train_df.shape)
print("Test data shape :", test_df.shape)


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "Voltage",
    "Current",
    "Temperature"
]

target = "SOC"


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


print("\nFeatures:")
print(features)

print("\nTarget:")
print(target)


# ============================================================
# 4. CREATE GRADIENT BOOSTING MODEL
# ============================================================

model = GradientBoostingRegressor(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\nTraining Gradient Boosting model...")

model.fit(X_train, y_train)

print("Training completed!")


# ============================================================
# 6. PREDICT SOC
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)


print("\n======================================")
print("GRADIENT BOOSTING PERFORMANCE")
print("======================================")

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

print("======================================")


# ============================================================
# 8. SAVE PREDICTION RESULTS
# ============================================================

results = pd.DataFrame({
    "Cycle": test_df["Cycle"].values,
    "Time": test_df["Time"].values,
    "Actual_SOC": y_test.values,
    "Predicted_SOC": y_pred,
    "Error": y_test.values - y_pred
})


results_path = (
    output_folder +
    r"\Gradient_Boosting_SOC_Results.csv"
)

results.to_csv(
    results_path,
    index=False
)

print("\nPrediction results saved:")
print(results_path)


# ============================================================
# 9. SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Model": ["Gradient Boosting"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})


metrics_path = (
    output_folder +
    r"\Gradient_Boosting_Metrics.csv"
)

metrics_df.to_csv(
    metrics_path,
    index=False
)

print("Metrics saved:")
print(metrics_path)


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_


feature_importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})


feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n======================================")
print("FEATURE IMPORTANCE")
print("======================================")

print(feature_importance_df)


importance_path = (
    output_folder +
    r"\Gradient_Boosting_Feature_Importance.csv"
)

feature_importance_df.to_csv(
    importance_path,
    index=False
)


# ============================================================
# 11. FIGURE 26
# ACTUAL VS PREDICTED SOC
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    y_test.values,
    label="Actual SOC",
    linewidth=1.5
)

plt.plot(
    y_pred,
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
    "Gradient Boosting: Actual vs Predicted SOC",
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
    r"\26_Gradient_Boosting_Actual_vs_Predicted_SOC.png"
)

plt.savefig(
    plot1,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 12. FIGURE 27
# ACTUAL VS PREDICTED SCATTER
# ============================================================

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.25,
    s=10
)


# Perfect prediction line

minimum = min(
    y_test.min(),
    y_pred.min()
)

maximum = max(
    y_test.max(),
    y_pred.max()
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
    "Gradient Boosting: Actual vs Predicted SOC",
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
    r"\27_Gradient_Boosting_Scatter.png"
)

plt.savefig(
    plot2,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 13. FIGURE 28
# PREDICTION ERROR
# ============================================================

error = y_test.values - y_pred


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
    "Test Samples",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "Prediction Error (%)",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Gradient Boosting: SOC Prediction Error",
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
    r"\28_Gradient_Boosting_Error.png"
)

plt.savefig(
    plot3,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 14. FIGURE 29
# FEATURE IMPORTANCE
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
)

plt.xlabel(
    "Input Feature",
    fontsize=13,
    fontweight="bold"
)

plt.ylabel(
    "Importance",
    fontsize=13,
    fontweight="bold"
)

plt.title(
    "Gradient Boosting Feature Importance",
    fontsize=15,
    fontweight="bold"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


plot4 = (
    output_folder +
    r"\29_Gradient_Boosting_Feature_Importance.png"
)

plt.savefig(
    plot4,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 15. PRINT SAVED FILES
# ============================================================

print("\n======================================")
print("ALL FILES SAVED")
print("======================================")

print(results_path)
print(metrics_path)
print(importance_path)

print(plot1)
print(plot2)
print(plot3)
print(plot4)


# ============================================================
# 16. SHOW ALL FIGURES
# ============================================================

plt.show()