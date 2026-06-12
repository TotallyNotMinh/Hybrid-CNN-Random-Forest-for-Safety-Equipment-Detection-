import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

# Load data
print("Loading resnet_embeddings_filtered.csv...")
df = pd.read_csv("resnet_embeddings_filtered.csv")

feature_cols = [col for col in df.columns if col.startswith("f")]
X = df[feature_cols].values
y = df["label"].values
class_names = df["class_name"].values

# Task definitions: (Task Name, Class A ID, Class B ID, minority_label)
# To find the minority label, let's look at value counts in the dataset:
# Class 0: Gloves, Class 2: NO-Gloves
# Class 1: Hardhat, Class 3: NO-Hardhat
# Class 4: NO-Safety Vest, Class 5: Safety Vest
# Let's count them:
for label in sorted(np.unique(y)):
    count = np.sum(y == label)
    print(f"Label {label} ({class_names[y == label][0]}): {count}")

tasks = {
    "Hardhat_vs_NO-Hardhat": {
        "labels": [1, 3],
        "class_names": ["Hardhat", "NO-Hardhat"]
    },
    "Safety_Vest_vs_NO-Safety_Vest": {
        "labels": [5, 4],
        "class_names": ["Safety Vest", "NO-Safety Vest"]
    },
    "Gloves_vs_NO-Gloves": {
        "labels": [0, 2],
        "class_names": ["Gloves", "NO-Gloves"]
    }
}

results = []

for task_name, info in tasks.items():
    print(f"\nEvaluating task: {task_name}")
    mask = np.isin(y, info["labels"])
    X_task = X[mask]
    y_task = y[mask]
    class_names_task = class_names[mask]
    
    # 80/20 split
    X_train, X_val, y_train, y_val = train_test_split(
        X_task, y_task, test_size=0.2, random_state=42, stratify=y_task
    )
    
    # Identify minority class label in training set
    unique_lbls, counts = np.unique(y_train, return_counts=True)
    minority_idx = np.argmin(counts)
    minority_label = unique_lbls[minority_idx]
    minority_name = info["class_names"][0] if minority_label == info["labels"][0] else info["class_names"][1]
    print(f"Minority class: {minority_name} (label {minority_label}), counts: {counts}")
    
    # Define models
    # Subsample training data for SVM to avoid long training times
    max_svm_samples = 10000
    if len(y_train) > max_svm_samples:
        svm_idx, _ = train_test_split(
            np.arange(len(y_train)), train_size=max_svm_samples, 
            stratify=y_train, random_state=42
        )
        X_train_svm = X_train[svm_idx]
        y_train_svm = y_train[svm_idx]
    else:
        X_train_svm = X_train
        y_train_svm = y_train

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM (RBF)*": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    for clf_name, clf in classifiers.items():
        print(f"  Training {clf_name}...")
        
        # Select appropriate training set
        X_tr = X_train_svm if "SVM" in clf_name else X_train
        y_tr = y_train_svm if "SVM" in clf_name else y_train
        
        start_time = time.time()
        clf.fit(X_tr, y_tr)
        train_time = time.time() - start_time
        
        # Predict on validation set
        y_pred = clf.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro")
        f1_weighted = f1_score(y_val, y_pred, average="weighted")
        f1_minority = f1_score(y_val, y_pred, pos_label=minority_label)
        recall_minority = recall_score(y_val, y_pred, pos_label=minority_label)
        
        print(f"    Acc: {acc:.4f} | F1 Macro: {f1_macro:.4f} | F1 Minority ({minority_name}): {f1_minority:.4f} | Time: {train_time:.2f}s")
        
        results.append({
            "Task": task_name,
            "Classifier": clf_name,
            "Accuracy": acc,
            "F1_Macro": f1_macro,
            "F1_Weighted": f1_weighted,
            "Minority_Class": minority_name,
            "Minority_F1": f1_minority,
            "Minority_Recall": recall_minority,
            "Train_Time_Sec": train_time
        })

# Save results
res_df = pd.DataFrame(results)
res_df.to_csv("results/classifier_comparison.csv", index=False)
print("\nSaved comparison results to results/classifier_comparison.csv")

# Print LaTeX tables for easy copy-pasting
for task_name in tasks.keys():
    task_res = res_df[res_df["Task"] == task_name]
    minority_name = task_res["Minority_Class"].iloc[0]
    print(f"\nLaTeX Table for {task_name}:")
    print("\\begin{table}[h]")
    print("  \\centering")
    print("  \\caption{Classifier Comparison for " + task_name.replace("_", " ") + " task (Validation Set)}")
    print("  \\begin{tabular}{lcccc}")
    print("    \\hline")
    print(f"    Classifier & Accuracy (\\%) & Macro F1 (\\%) & {minority_name} F1 (\\%) & Train Time (s) \\\\")
    print("    \\hline")
    for _, row in task_res.iterrows():
        print(f"    {row['Classifier']} & {row['Accuracy']*100:.2f}\\% & {row['F1_Macro']*100:.2f}\\% & {row['Minority_F1']*100:.2f}\\% & {row['Train_Time_Sec']:.2f}s \\\\")
    print("    \\hline")
    print("  \\end{tabular}")
    print("  \\label{tab:comp_" + task_name.lower().replace("_", "") + "}")
    print("\\end{table}")
