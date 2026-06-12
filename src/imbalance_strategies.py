import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print("WARNING: imblearn not found. SMOTE and RandomUnderSampler will be skipped.")
    print("Install with: .venv\\Scripts\\pip.exe install imbalanced-learn\n")

# ── data ──────────────────────────────────────────────────────────────
CSV_PATH = "resnet_embeddings_filtered.csv"
df = pd.read_csv(CSV_PATH)

feature_cols = [col for col in df.columns if col.startswith("f")]

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

os.makedirs("results", exist_ok=True)

all_rows = []

for class_a_name, class_b_name in tasks:
    task_label = f"{class_a_name} vs {class_b_name}"
    print("=" * 70)
    print(f"Task: {task_label}")
    print("=" * 70)

    class_a = name_to_label[class_a_name]
    class_b = name_to_label[class_b_name]

    # filter to 2 relevant labels
    mask = np.isin(df["label"].values, [class_a, class_b])
    df_task = df[mask].copy()

    X = df_task[feature_cols].values
    y_raw = df_task["label"].values

    # remap to 0/1  (class_a → 0, class_b → 1)
    y = np.where(y_raw == class_a, 0, 1)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # determine minority/majority
    unique, counts = np.unique(y_train, return_counts=True)
    count_map = dict(zip(unique, counts))
    minority_label = min(count_map, key=count_map.get)
    majority_label = max(count_map, key=count_map.get)
    minority_name = class_a_name if minority_label == 0 else class_b_name
    majority_name = class_a_name if majority_label == 0 else class_b_name

    print(f"  Train: {len(y_train)} | Val: {len(y_val)}")
    print(f"  Train class 0 ({class_a_name}): {count_map.get(0,0)}  |  class 1 ({class_b_name}): {count_map.get(1,0)}")
    print(f"  Minority class: {minority_label} ({minority_name})  |  Majority class: {majority_label} ({majority_name})\n")

    # ── define strategies ─────────────────────────────────────────────
    strategies = {}

    strategies["Baseline"] = {
        "X_train": X_train,
        "y_train": y_train,
        "model_kwargs": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
    }
    strategies["class_weight='balanced'"] = {
        "X_train": X_train,
        "y_train": y_train,
        "model_kwargs": {"n_estimators": 100, "random_state": 42, "n_jobs": -1, "class_weight": "balanced"},
    }

    if HAS_IMBLEARN:
        smote = SMOTE(random_state=42)
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        strategies["SMOTE"] = {
            "X_train": X_sm,
            "y_train": y_sm,
            "model_kwargs": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        }

        rus = RandomUnderSampler(random_state=42)
        X_ru, y_ru = rus.fit_resample(X_train, y_train)
        strategies["RandomUnderSampler"] = {
            "X_train": X_ru,
            "y_train": y_ru,
            "model_kwargs": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        }

    # ── run strategies ────────────────────────────────────────────────
    for strat_name, cfg in strategies.items():
        model = RandomForestClassifier(**cfg["model_kwargs"])
        model.fit(cfg["X_train"], cfg["y_train"])
        preds = model.predict(X_val)

        acc = accuracy_score(y_val, preds)

        # per-class precision, recall, f1
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_val, preds, labels=[0, 1], zero_division=0
        )

        min_idx = minority_label
        maj_idx = majority_label

        row = {
            "task": task_label,
            "strategy": strat_name,
            "accuracy": acc,
            "minority_class": f"{minority_label} ({minority_name})",
            "minority_precision": prec[min_idx],
            "minority_recall": rec[min_idx],
            "minority_f1": f1[min_idx],
            "majority_class": f"{majority_label} ({majority_name})",
            "majority_precision": prec[maj_idx],
            "majority_recall": rec[maj_idx],
            "majority_f1": f1[maj_idx],
        }
        all_rows.append(row)

        print(f"  [{strat_name}]")
        print(f"    Accuracy:          {acc:.4f}")
        print(f"    Minority  ({minority_name:15s}):  P={prec[min_idx]:.4f}  R={rec[min_idx]:.4f}  F1={f1[min_idx]:.4f}")
        print(f"    Majority  ({majority_name:15s}):  P={prec[maj_idx]:.4f}  R={rec[maj_idx]:.4f}  F1={f1[maj_idx]:.4f}")
        print()

# ── save results ──────────────────────────────────────────────────────
results_df = pd.DataFrame(all_rows)
results_df.to_csv("results/imbalance_strategies.csv", index=False)
print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(results_df.to_string(index=False))
print(f"\nSaved → results/imbalance_strategies.csv  ({len(results_df)} rows)")
