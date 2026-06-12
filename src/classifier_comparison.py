"""
Classifier Comparison for PPE Detection Binary Tasks
Trains 7 classifiers on each of 3 binary tasks using 80/20 split.
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ── Load data ────────────────────────────────────────────────────────────
df = pd.read_csv("resnet_embeddings_filtered.csv")
feature_cols = [f"f{i}" for i in range(512)]
X_all = df[feature_cols].values
y_all = df["label"].values

# ── Global 80/20 split (stratified) ─────────────────────────────────────
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# ── Binary tasks ─────────────────────────────────────────────────────────
tasks = {
    "Gloves vs NO-Gloves": (0, 2),
    "Hardhat vs NO-Hardhat": (1, 3),
    "Safety Vest vs NO-Safety Vest": (5, 4),
}

# ── Classifiers ──────────────────────────────────────────────────────────
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM RBF": SVC(probability=True, random_state=42),
    "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "AdaBoost": AdaBoostClassifier(random_state=42, algorithm="SAMME"),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# ── Results storage ──────────────────────────────────────────────────────
results = []

for task_name, (pos_label, neg_label) in tasks.items():
    print(f"\n{'='*70}")
    print(f"  Task: {task_name}  (positive={pos_label}, negative={neg_label})")
    print(f"{'='*70}")

    # Filter train/val to only the 2 relevant labels
    train_mask = np.isin(y_train_full, [pos_label, neg_label])
    val_mask = np.isin(y_val_full, [pos_label, neg_label])

    X_train_task = X_train_full[train_mask]
    y_train_task = y_train_full[train_mask]
    X_val_task = X_val_full[val_mask]
    y_val_task = y_val_full[val_mask]

    # Remap labels to 0/1 (pos_label -> 1, neg_label -> 0)
    y_train_bin = (y_train_task == pos_label).astype(int)
    y_val_bin = (y_val_task == pos_label).astype(int)

    # Determine minority class in validation set
    minority_class = 1 if y_val_bin.sum() <= len(y_val_bin) / 2 else 0

    print(f"  Train: {len(X_train_task)} samples | Val: {len(X_val_task)} samples")
    print(f"  Train class distribution: 0={np.sum(y_train_bin==0)}, 1={np.sum(y_train_bin==1)}")
    print(f"  Val   class distribution: 0={np.sum(y_val_bin==0)}, 1={np.sum(y_val_bin==1)}")
    print(f"  Minority class: {minority_class}")
    print(f"{'─'*70}")
    print(f"  {'Classifier':<25} {'Accuracy':>10} {'F1-min':>10} {'Time (s)':>10}")
    print(f"{'─'*70}")

    for clf_name, clf_template in classifiers.items():
        # Clone the classifier for each task
        from sklearn.base import clone
        clf = clone(clf_template)

        # Subsample for SVM RBF
        if clf_name == "SVM RBF" and len(X_train_task) > 10000:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
            sub_idx, _ = next(sss.split(X_train_task, y_train_bin))
            X_train_clf = X_train_task[sub_idx]
            y_train_clf = y_train_bin[sub_idx]
            print(f"    [SVM subsampled to {len(X_train_clf)} training samples]")
        else:
            X_train_clf = X_train_task
            y_train_clf = y_train_bin

        # Train
        t0 = time.time()
        clf.fit(X_train_clf, y_train_clf)
        train_time = time.time() - t0

        # Predict on validation
        y_pred = clf.predict(X_val_task)

        acc = accuracy_score(y_val_bin, y_pred)
        f1_min = f1_score(y_val_bin, y_pred, pos_label=minority_class)

        print(f"  {clf_name:<25} {acc:>10.4f} {f1_min:>10.4f} {train_time:>10.2f}")

        results.append({
            "Task": task_name,
            "Classifier": clf_name,
            "Accuracy": round(acc, 4),
            "F1_minority": round(f1_min, 4),
            "Train_time_s": round(train_time, 2),
        })

# ── Save results ─────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv("results/classifier_comparison.csv", index=False)
print(f"\n✓ Results saved to results/classifier_comparison.csv")
print(f"\n{'='*70}")
print("  Summary Table")
print(f"{'='*70}")
print(results_df.to_string(index=False))
