import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 1. Load data
csv_path = "resnet_embeddings_filtered.csv"
df = pd.read_csv(csv_path)

feature_cols = [col for col in df.columns if col.startswith("f")]
X = df[feature_cols].values
y = df["label"].values
class_names = df["class_name"].values

# 80/20 stratified split (identical random state)
X_train, X_val, y_train, y_val, name_train, name_val = train_test_split(
    X, y, class_names, test_size=0.2, random_state=42, stratify=y
)

label_to_name = {
    0: 'Gloves',
    1: 'Hardhat',
    2: 'NO-Gloves',
    3: 'NO-Hardhat',
    4: 'NO-Safety Vest',
    5: 'Safety Vest'
}
name_to_label = {v: k for k, v in label_to_name.items()}

# Load models
binary_models = {
    "hardhat": joblib.load("saved_models/Hardhat_vs_NO-Hardhat.pkl"),
    "vest": joblib.load("saved_models/Safety_Vest_vs_NO-Safety_Vest.pkl"),
    "gloves": joblib.load("saved_models/Gloves_vs_NO-Gloves.pkl")
}

six_class_model = joblib.load("saved_models/six_class_model.pkl")

# We will collect metrics for each class
# Class: name, label ID, task name, and the binary model key
classes_info = [
    {"name": "Hardhat", "label": 1, "task": "hardhat", "opp_name": "NO-Hardhat", "opp_label": 3},
    {"name": "NO-Hardhat", "label": 3, "task": "hardhat", "opp_name": "Hardhat", "opp_label": 1},
    {"name": "Safety Vest", "label": 5, "task": "vest", "opp_name": "NO-Safety Vest", "opp_label": 4},
    {"name": "NO-Safety Vest", "label": 4, "task": "vest", "opp_name": "Safety Vest", "opp_label": 5},
    {"name": "Gloves", "label": 0, "task": "gloves", "opp_name": "NO-Gloves", "opp_label": 2},
    {"name": "NO-Gloves", "label": 2, "task": "gloves", "opp_name": "Gloves", "opp_label": 0}
]

print("Comparing models on the exact same validation split...")
print(f"Validation size: {len(y_val)}\n")

results = []

for c_info in classes_info:
    name = c_info["name"]
    lbl = c_info["label"]
    opp_name = c_info["opp_name"]
    opp_lbl = c_info["opp_label"]
    task = c_info["task"]
    
    # Subset mask for this task
    task_mask = np.isin(y_val, [lbl, opp_lbl])
    X_val_task = X_val[task_mask]
    y_val_task = y_val[task_mask]
    
    # 1. Evaluate Binary Model
    bin_model = binary_models[task]
    bin_preds = bin_model.predict(X_val_task)
    bin_p, bin_r, bin_f, _ = precision_recall_fscore_support(
        y_val_task, bin_preds, labels=[lbl, opp_lbl]
    )
    
    # 2. Evaluate 6-Class Unconstrained (on the task subset)
    six_preds_global = six_class_model.predict(X_val_task)
    # Note: the predictions can be anything (0-5). We compute precision/recall treating any prediction != lbl as negative (if true is lbl)
    # or we can use the classification report metrics of the global 6-class model.
    # Let's get the global 6-class predictions on the entire validation set and extract metrics for this class.
    six_preds_all = six_class_model.predict(X_val)
    six_p_all, six_r_all, six_f_all, _ = precision_recall_fscore_support(
        y_val, six_preds_all, labels=list(label_to_name.keys())
    )
    class_idx = list(label_to_name.keys()).index(lbl)
    uncon_p = six_p_all[class_idx]
    uncon_r = six_r_all[class_idx]
    uncon_f = six_f_all[class_idx]
    
    # 3. Evaluate 6-Class Constrained
    # Get probabilities for all 6 classes
    probs = six_class_model.predict_proba(X_val_task)
    # Restrict to task classes
    classes_list = list(six_class_model.classes_)
    idx_lbl = classes_list.index(lbl)
    idx_opp = classes_list.index(opp_lbl)
    
    task_probs = probs[:, [idx_lbl, idx_opp]]
    preds_constrained_idx = np.argmax(task_probs, axis=1)
    preds_constrained = np.array([lbl, opp_lbl])[preds_constrained_idx]
    
    const_p, const_r, const_f, _ = precision_recall_fscore_support(
        y_val_task, preds_constrained, labels=[lbl, opp_lbl]
    )
    
    results.append({
        "Class": name,
        "Binary Precision": bin_p[0],
        "Binary Recall": bin_r[0],
        "Binary F1": bin_f[0],
        "Unconstrained Precision": uncon_p,
        "Unconstrained Recall": uncon_r,
        "Unconstrained F1": uncon_f,
        "Constrained Precision": const_p[0],
        "Constrained Recall": const_r[0],
        "Constrained F1": const_f[0]
    })

# Format into a beautiful pandas DataFrame
df_res = pd.DataFrame(results)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("SUMMARY OF METRICS COMPARISON ON STRATIFIED VALIDATION SPLIT")
print("=" * 100)
for idx, row in df_res.iterrows():
    print(f"Class: {row['Class']}")
    print(f"  Binary:        Precision={row['Binary Precision']:.4f} | Recall={row['Binary Recall']:.4f} | F1={row['Binary F1']:.4f}")
    print(f"  Unconstrained: Precision={row['Unconstrained Precision']:.4f} | Recall={row['Unconstrained Recall']:.4f} | F1={row['Unconstrained F1']:.4f}")
    print(f"  Constrained:   Precision={row['Constrained Precision']:.4f} | Recall={row['Constrained Recall']:.4f} | F1={row['Constrained F1']:.4f}")
    print("-" * 100)
