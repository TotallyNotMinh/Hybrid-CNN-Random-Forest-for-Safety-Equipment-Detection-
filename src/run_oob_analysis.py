import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
print("Loading resnet_embeddings_filtered.csv...")
df = pd.read_csv("resnet_embeddings_filtered.csv")

feature_cols = [col for col in df.columns if col.startswith("f")]
X = df[feature_cols].values
y = df["label"].values
class_names = df["class_name"].values

tasks = {
    "Hardhat_vs_NO-Hardhat": [1, 3],
    "Safety_Vest_vs_NO-Safety_Vest": [5, 4],
    "Gloves_vs_NO-Gloves": [0, 2]
}

n_estimators_list = [10, 25, 50, 100, 150, 200, 300]
oob_results = {task: [] for task in tasks.keys()}

for task_name, labels in tasks.items():
    print(f"\nAnalyzing OOB convergence for: {task_name}")
    mask = np.isin(y, labels)
    X_task = X[mask]
    y_task = y[mask]
    
    # 80/20 split (we fit on training set and use its OOB score)
    X_train, _, y_train, _ = train_test_split(
        X_task, y_task, test_size=0.2, random_state=42, stratify=y_task
    )
    
    for n in n_estimators_list:
        print(f"  Training RF with n_estimators={n} (oob_score=True)...")
        rf = RandomForestClassifier(
            n_estimators=n,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        oob_error = 1.0 - rf.oob_score_
        print(f"    OOB Error: {oob_error:.4f}")
        oob_results[task_name].append(oob_error)

# Save to CSV
oob_df = pd.DataFrame(oob_results, index=n_estimators_list)
oob_df.index.name = "n_estimators"
oob_df.reset_index(inplace=True)
oob_df.to_csv("results/oob_data.csv", index=False)
print("\nSaved OOB data to results/oob_data.csv")

# Plot styling
plt.figure(figsize=(9, 5), dpi=300)
colors = {"Hardhat_vs_NO-Hardhat": "#FF6B6B", "Safety_Vest_vs_NO-Safety_Vest": "#4D96FF", "Gloves_vs_NO-Gloves": "#6BCB77"}
markers = {"Hardhat_vs_NO-Hardhat": "o", "Safety_Vest_vs_NO-Safety_Vest": "s", "Gloves_vs_NO-Gloves": "^"}
labels = {"Hardhat_vs_NO-Hardhat": "Hardhat vs. NO-Hardhat", "Safety_Vest_vs_NO-Safety_Vest": "Safety Vest vs. NO-Safety Vest", "Gloves_vs_NO-Gloves": "Gloves vs. NO-Gloves"}

for task_name in tasks.keys():
    plt.plot(
        n_estimators_list, 
        oob_results[task_name], 
        label=labels[task_name], 
        color=colors[task_name], 
        marker=markers[task_name], 
        linewidth=2.5, 
        markersize=7
    )

plt.title("Out-of-Bag (OOB) Error vs. Number of Trees ($n\\_estimators$)", fontsize=13, fontweight='bold', pad=15)
plt.xlabel("Number of Trees ($n\\_estimators$)", fontsize=11, labelpad=10)
plt.ylabel("OOB Error Rate", fontsize=11, labelpad=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.xticks(n_estimators_list)
plt.legend(frameon=True, facecolor="white", edgecolor="#ddd", fontsize=10)
plt.tight_layout()

# Save plot
plt.savefig("results/oob_error_vs_trees.png", bbox_inches='tight')
print("Saved convergence plot to results/oob_error_vs_trees.png")
