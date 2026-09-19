import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. PROJECT FOLDER
# ============================================================

project_folder = r"C:\Users\User\Desktop\SOC_Project"


# ============================================================
# 2. LOAD TRAINING AND TESTING DATA
# ============================================================

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

print("Training data shape:", train_df.shape)
print("Testing data shape :", test_df.shape)


# ============================================================
# 3. SELECT INPUT FEATURES
# ============================================================

features = [
    "Voltage",
    "Current",
    "Temperature"
]

X_train = train_df[features]
X_test = test_df[features]


# ============================================================
# 4. SELECT TARGET
# ============================================================

y_train = train_df["SOC"]
y_test = test_df["SOC"]


print("\nInput features:")
print(features)

print("\nTarget:")
print("SOC")


# ============================================================
# 5. CREATE DECISION TREE MODEL
# ============================================================

model = DecisionTreeRegressor(
    max_depth=12,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining Decision Tree...")

model.fit(
    X_train,
    y_train
)

print("Training completed!")


# ============================================================
# 7. PREDICT SOC
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. CALCULATE METRICS
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
# 9. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("DECISION TREE RESULTS")
print("==========================================")

print(f"MAE  : {mae:.4f} %")
print(f"RMSE : {rmse:.4f} %")
print(f"R²   : {r2:.4f}")


# ============================================================
# 10. FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\n==========================================")
print("FEATURE IMPORTANCE")
print("==========================================")

print(importance_df)


# ============================================================
# 11. SAVE FEATURE IMPORTANCE
# ============================================================

importance_file = os.path.join(
    project_folder,
    "Decision_Tree_Feature_Importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


# ============================================================
# 12. CREATE RESULTS DATAFRAME
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
# 13. SAVE PREDICTION RESULTS
# ============================================================

results_file = os.path.join(
    project_folder,
    "Decision_Tree_SOC_Results.csv"
)

results.to_csv(
    results_file,
    index=False
)

print("\nPrediction results saved:")
print(results_file)


# ============================================================
# 14. PLOT 1
# ACTUAL SOC VS PREDICTED SOC
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
    "Actual vs Predicted SOC - Decision Tree",
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
        "18_Actual_vs_Predicted_SOC_Decision_Tree.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 15. PLOT 2
# ACTUAL VS PREDICTED SCATTER
# ============================================================

plt.figure(figsize=(8, 8))

plt.scatter(
    y_test,
    y_pred,
    s=8,
    alpha=0.4
)

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
    "Actual SOC vs Predicted SOC - Decision Tree",
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
        "19_Decision_Tree_Actual_vs_Predicted.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 16. PLOT 3
# PREDICTION ERROR
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
    "SOC Prediction Error - Decision Tree",
    fontsize=16,
    fontweight="bold"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        project_folder,
        "20_Decision_Tree_Error.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 17. PLOT 4
# FEATURE IMPORTANCE
# ============================================================

plt.figure(figsize=(8, 6))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel(
    "Features",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Importance",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Decision Tree Feature Importance",
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

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        project_folder,
        "21_Decision_Tree_Feature_Importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 18. SAVE MODEL RESULTS
# ============================================================

new_result = pd.DataFrame({
    "Model": ["Decision Tree"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2]
})

model_results_file = os.path.join(
    project_folder,
    "decision_tree_results.csv"
)

new_result.to_csv(
    model_results_file,
    index=False
)


# ============================================================
# 19. SHOW ALL PLOTS
# ============================================================

plt.show()


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("STEP 7 COMPLETED")
print("==========================================")

print("\nModel: Decision Tree Regression")

print(f"MAE  = {mae:.4f} %")
print(f"RMSE = {rmse:.4f} %")
print(f"R²   = {r2:.4f}")

print("\nSaved prediction file:")
print(results_file)

print("\nSaved feature importance:")
print(importance_file)

print("\nSaved plots:")
print("18_Actual_vs_Predicted_SOC_Decision_Tree.png")
print("19_Decision_Tree_Actual_vs_Predicted.png")
print("20_Decision_Tree_Error.png")
print("21_Decision_Tree_Feature_Importance.png")

print("\nNext step:")
print("Random Forest Regression")