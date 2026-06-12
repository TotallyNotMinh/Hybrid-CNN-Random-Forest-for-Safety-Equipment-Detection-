"""
5-Fold Stratified Cross-Validation for Random Forest on PPE Detection Binary Tasks
Uses the FULL dataset (CV creates its own folds).
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

warnings.filterwarnings("ignore")

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv("resnet_embeddings_filtered.csv")
feature_cols = [f"f{i}" for i in range(512)]
X_all = df[feature_cols].values
y_all = df["label"].values

# ── Binary tasks ─────────────────────────────────────────────────────────
tasks = {
    "Gloves vs NO-Gloves": (0, 2),
    "Hardhat vs NO-Hardhat": (1, 3),
    "Safety Vest vs NO-Safety Vest": (5, 4),
}

# ── Cross-validation setup ───────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for task_name, (pos_label, neg_label) in tasks.items():
    print(f"\n{'='*70}")
    print(f"  Task: {task_name}  (positive={pos_label}, negative={neg_label})")
    print(f"{'='*70}")

    # Filter to only the 2 relevant labels (FULL dataset)
    task_mask = np.isin(y_all, [pos_label, neg_label])
    X_task = X_all[task_mask]
    y_task = y_all[task_mask]

    # Remap labels to 0/1
    y_bin = (y_task == pos_label).astype(int)

    print(f"  Total samples: {len(X_task)}")
    print(f"  Class distribution: 0={np.sum(y_bin==0)}, 1={np.sum(y_bin==1)}")

    # Storage for fold metrics
    train_acc_list, val_acc_list = [], []
    train_f1w_list, val_f1w_list = [], []
    train_rec_list, val_rec_list = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_task, y_bin), 1):
        X_train, X_val = X_task[train_idx], X_task[val_idx]
        y_train, y_val = y_bin[train_idx], y_bin[val_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Predictions
        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)

        # Metrics - Train
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1w = f1_score(y_train, y_train_pred, average="weighted")
        train_rec = recall_score(y_train, y_train_pred, average="macro")

        # Metrics - Val
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1w = f1_score(y_val, y_val_pred, average="weighted")
        val_rec = recall_score(y_val, y_val_pred, average="macro")

        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_f1w_list.append(train_f1w)
        val_f1w_list.append(val_f1w)
        train_rec_list.append(train_rec)
        val_rec_list.append(val_rec)

        print(f"  Fold {fold_idx}: train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  "
              f"train_f1w={train_f1w:.4f}  val_f1w={val_f1w:.4f}  "
              f"train_rec={train_rec:.4f}  val_rec={val_rec:.4f}")

    # Summary
    print(f"\n  {'Metric':<25} {'Train (mean±std)':>20} {'Val (mean±std)':>20}")
    print(f"  {'─'*65}")

    metrics = [
        ("Accuracy", train_acc_list, val_acc_list),
        ("F1-weighted", train_f1w_list, val_f1w_list),
        ("Recall-macro", train_rec_list, val_rec_list),
    ]

    for metric_name, train_vals, val_vals in metrics:
        t_mean, t_std = np.mean(train_vals), np.std(train_vals)
        v_mean, v_std = np.mean(val_vals), np.std(val_vals)
        print(f"  {metric_name:<25} {t_mean:.4f} ± {t_std:.4f}      {v_mean:.4f} ± {v_std:.4f}")

        results.append({
            "Task": task_name,
            "Metric": metric_name,
            "Train_mean": round(t_mean, 4),
            "Train_std": round(t_std, 4),
            "Val_mean": round(v_mean, 4),
            "Val_std": round(v_std, 4),
        })

# ── Save results ─────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("results/cross_validation.csv", index=False)
print(f"\n✓ Results saved to results/cross_validation.csv")
print(f"\n{'='*70}")
print("  Summary Table")
print(f"{'='*70}")
print(results_df.to_string(index=False))
