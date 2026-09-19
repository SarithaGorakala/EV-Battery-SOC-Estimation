import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. LOAD ML DATASET
# ============================================================

file_path = r"C:\Users\User\Desktop\SOC_Project\B0005_ML_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. SORT DATA BY CYCLE AND TIME
# ============================================================

df = df.sort_values(
    ["Cycle", "Time"]
).reset_index(drop=True)


# ============================================================
# 3. GET UNIQUE CYCLES
# ============================================================

cycles = sorted(df["Cycle"].unique())

print("\nTotal number of cycles:", len(cycles))

print("First cycle:", cycles[0])
print("Last cycle:", cycles[-1])


# ============================================================
# 4. SPLIT CYCLES
# ============================================================

# 80% cycles for training
# 20% cycles for testing

split_index = int(len(cycles) * 0.80)

train_cycles = cycles[:split_index]

test_cycles = cycles[split_index:]


# ============================================================
# 5. CREATE TRAINING AND TESTING DATA
# ============================================================

train_df = df[
    df["Cycle"].isin(train_cycles)
].copy()

test_df = df[
    df["Cycle"].isin(test_cycles)
].copy()


# ============================================================
# 6. DISPLAY SPLIT INFORMATION
# ============================================================

print("\n==========================================")
print("TRAIN / TEST SPLIT")
print("==========================================")

print("\nTraining cycles:")
print(
    train_cycles[0],
    "to",
    train_cycles[-1]
)

print("Number of training cycles:", len(train_cycles))

print("\nTesting cycles:")
print(
    test_cycles[0],
    "to",
    test_cycles[-1]
)

print("Number of testing cycles:", len(test_cycles))


print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))


# ============================================================
# 7. CREATE X AND y
# ============================================================

# Input features
X_train = train_df[
    [
        "Voltage",
        "Current",
        "Temperature"
    ]
]

X_test = test_df[
    [
        "Voltage",
        "Current",
        "Temperature"
    ]
]


# Target
y_train = train_df["SOC"]

y_test = test_df["SOC"]


# ============================================================
# 8. DISPLAY SHAPES
# ============================================================

print("\n==========================================")
print("FEATURE / TARGET SHAPES")
print("==========================================")

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 9. DISPLAY TRAINING FEATURES
# ============================================================

print("\nTraining features:")
print(X_train.head())


print("\nTraining target:")
print(y_train.head())


# ============================================================
# 10. CHECK FOR DATA LEAKAGE
# ============================================================

common_cycles = set(train_cycles).intersection(
    set(test_cycles)
)

print("\n==========================================")
print("DATA LEAKAGE CHECK")
print("==========================================")

print("Common cycles:", common_cycles)

if len(common_cycles) == 0:
    print("SUCCESS: No cycle overlap between train and test.")
else:
    print("WARNING: Cycle overlap detected!")


# ============================================================
# 11. CREATE OUTPUT FOLDER
# ============================================================

output_folder = r"C:\Users\User\Desktop\SOC_Project"

os.makedirs(
    output_folder,
    exist_ok=True
)


# ============================================================
# 12. SAVE TRAINING DATA
# ============================================================

train_file = os.path.join(
    output_folder,
    "train_data.csv"
)

train_df.to_csv(
    train_file,
    index=False
)


# ============================================================
# 13. SAVE TESTING DATA
# ============================================================

test_file = os.path.join(
    output_folder,
    "test_data.csv"
)

test_df.to_csv(
    test_file,
    index=False
)


# ============================================================
# 14. PLOT TRAIN / TEST CYCLES
# ============================================================

plt.figure(figsize=(10, 5))

plt.scatter(
    train_cycles,
    [1] * len(train_cycles),
    s=20,
    label="Training cycles"
)

plt.scatter(
    test_cycles,
    [2] * len(test_cycles),
    s=20,
    label="Testing cycles"
)

plt.yticks(
    [1, 2],
    ["Training", "Testing"],
    fontsize=11,
    fontweight="bold"
)

plt.xlabel(
    "Cycle Number",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel(
    "Dataset",
    fontsize=14,
    fontweight="bold"
)

plt.title(
    "Cycle-Based Training and Testing Split",
    fontsize=16,
    fontweight="bold"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# 15. SAVE PLOT
# ============================================================

plot_file = os.path.join(
    output_folder,
    "14_Train_Test_Cycle_Split.png"
)

plt.savefig(
    plot_file,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 16. SHOW PLOT
# ============================================================

plt.show()


# ============================================================
# 17. FINAL INFORMATION
# ============================================================

print("\n==========================================")
print("STEP 5 COMPLETED")
print("==========================================")

print("\nTraining data:")
print(train_file)

print("\nTesting data:")
print(test_file)

print("\nSplit plot:")
print(plot_file)

print("\nNext step:")
print("Train the first ML model - Linear Regression")