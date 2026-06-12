import pandas as pd
import numpy as np
import os
import time
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# ── data ──────────────────────────────────────────────────────────────
CSV_PATH = "resnet_embeddings_filtered.csv"
df = pd.read_csv(CSV_PATH)

feature_cols = [col for col in df.columns if col.startswith("f")]
X_all = df[feature_cols].values
y_all = df["label"].values

label_to_name = {}
for label, name in zip(df["label"], df["class_name"]):
    label_to_name[label] = name
name_to_label = {v: k for k, v in label_to_name.items()}

print(f"Dataset shape: {df.shape}")
print(f"Label mapping: {label_to_name}\n")

# ── binary tasks ──────────────────────────────────────────────────────
tasks = [
    ("Gloves", "NO-Gloves"),
    ("Hardhat", "NO-Hardhat"),
    ("Safety Vest", "NO-Safety Vest"),
]

# ── hyperparameter grid ──────────────────────────────────────────────
param_dist = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'max_features': [8, 16, 22, 32, 64, 128],
    'min_samples_leaf': [1, 2, 4, 8],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy'],
}

os.makedirs("results", exist_ok=True)

all_best_rows = []   # for hyperparam_search.csv
all_full_dfs = []    # for hyperparam_search_full.csv

for class_a_name, class_b_name in tasks:
    task_label = f"{class_a_name} vs {class_b_name}"
    print("=" * 70)
    print(f"Task: {task_label}")
    print("=" * 70)

    class_a = name_to_label[class_a_name]
    class_b = name_to_label[class_b_name]

    # filter to the two relevant labels
    mask = np.isin(y_all, [class_a, class_b])
    X_task = X_all[mask]
    y_task = y_all[mask]

    # remap to 0/1  (class_a → 0, class_b → 1)
    y_task = np.where(y_task == class_a, 0, 1)
    print(f"  Samples: {len(y_task)}  |  class 0 ({class_a_name}): {(y_task==0).sum()}  |  class 1 ({class_b_name}): {(y_task==1).sum()}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        random_state=42,
        verbose=1,
        return_train_score=False,
    )

    t0 = time.time()
    search.fit(X_task, y_task)
    elapsed = time.time() - t0

    print(f"\n  Best CV F1-macro: {search.best_score_:.5f}")
    print(f"  Best params: {search.best_params_}")
    print(f"  Elapsed: {elapsed/60:.1f} min\n")

    # top-5 configurations
    cv_df = pd.DataFrame(search.cv_results_)
    cv_df = cv_df.sort_values("rank_test_score")
    top5 = cv_df.head(5)

    print("  Top-5 configurations:")
    for i, row in enumerate(top5.itertuples(), 1):
        print(f"    #{i}  F1={row.mean_test_score:.5f} ± {row.std_test_score:.5f}  | {row.params}")
    print()

    # collect best params for summary CSV
    for param, value in search.best_params_.items():
        all_best_rows.append({
            "task": task_label,
            "param": param,
            "value": value,
            "best_score": search.best_score_,
        })

    # collect full CV results
    cv_df["task"] = task_label
    all_full_dfs.append(cv_df)

# ── save results ──────────────────────────────────────────────────────
best_df = pd.DataFrame(all_best_rows)
best_df.to_csv("results/hyperparam_search.csv", index=False)
print(f"Saved best-param summary → results/hyperparam_search.csv  ({len(best_df)} rows)")

full_df = pd.concat(all_full_dfs, ignore_index=True)
full_df.to_csv("results/hyperparam_search_full.csv", index=False)
print(f"Saved full CV results   → results/hyperparam_search_full.csv  ({len(full_df)} rows)")
