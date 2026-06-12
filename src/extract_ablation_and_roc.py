import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def main():
    filt_path = "resnet_embeddings_filtered.csv"
    if not os.path.exists(filt_path):
        print("Missing filtered dataset file.")
        return
        
    print("Loading filtered dataset...")
    df_filt = pd.read_csv(filt_path)
    
    # Extract split identical to train.py
    feature_cols = [col for col in df_filt.columns if col.startswith("f")]
    X = df_filt[feature_cols].values
    y = df_filt["label"].values
    class_names = df_filt["class_name"].values
    
    X_train, X_val, y_train, y_val, name_train, name_val = train_test_split(
        X, y, class_names, test_size=0.2, random_state=42, stratify=y
    )
    
    label_to_name = {}
    for label, name in zip(df_filt["label"], df_filt["class_name"]):
        label_to_name[label] = name
    name_to_label = {v: k for k, v in label_to_name.items()}

    # 1. Feature Importance Concentration
    print("\n=== FEATURE IMPORTANCE CONCENTRATION ===")
    tasks = [
        ("Hardhat", "NO-Hardhat"),
        ("Safety Vest", "NO-Safety Vest"),
        ("Gloves", "NO-Gloves")
    ]
    for class_a_name, class_b_name in tasks:
        model_name = f"{class_a_name}_vs_{class_b_name}.pkl".replace(" ", "_")
        model_path = os.path.join("saved_models", model_name)
        if not os.path.exists(model_path):
            continue
        model = joblib.load(model_path)
        importances = model.feature_importances_
        sorted_importances = np.sort(importances)[::-1]
        cumulative = np.cumsum(sorted_importances)
        
        n_50 = np.argmax(cumulative >= 0.5) + 1
        n_80 = np.argmax(cumulative >= 0.8) + 1
        print(f"Task: {class_a_name} vs {class_b_name}")
        print(f"  Dimensions for 50% importance: {n_50} / 512 ({n_50/512*100:.1f}%)")
        print(f"  Dimensions for 80% importance: {n_80} / 512 ({n_80/512*100:.1f}%)")

    # 2. max_features Ablation (For all three tasks)
    print("\n=== MAX_FEATURES ABLATION FOR ALL THREE TASKS ===")
    for class_a_name, class_b_name in tasks:
        print(f"\nRunning ablation for: {class_a_name} vs {class_b_name}")
        class_a = name_to_label[class_a_name]
        class_b = name_to_label[class_b_name]
        
        train_mask = np.isin(y_train, [class_a, class_b])
        X_train_pair = X_train[train_mask]
        y_train_pair = y_train[train_mask]
        
        val_mask = np.isin(y_val, [class_a, class_b])
        X_val_pair = X_val[val_mask]
        y_val_pair = y_val[val_mask]
        
        # Test different max_features values
        max_features_list = [8, 16, 22, 32, 64, 128]
        for mf in max_features_list:
            rf = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42, n_jobs=-1)
            rf.fit(X_train_pair, y_train_pair)
            preds = rf.predict(X_val_pair)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val_pair, preds, labels=[class_a, class_b]
            )
            
            # class_b is the NO- compliance class
            print(f"  max_features={mf:3d} | {class_b_name} Precision: {precision[1]:.4f} | Recall: {recall[1]:.4f} | F1: {f1[1]:.4f}")

    # 3. Hardhat Threshold Analysis & ROC
    print("\n=== HARDHAT THRESHOLD & ROC ANALYSIS ===")
    class_a_name, class_b_name = "Hardhat", "NO-Hardhat"
    class_a = name_to_label[class_a_name]
    class_b = name_to_label[class_b_name]
    
    val_mask = np.isin(y_val, [class_a, class_b])
    X_val_pair = X_val[val_mask]
    y_val_pair = y_val[val_mask]
    
    model_name = f"{class_a_name}_vs_{class_b_name}.pkl".replace(" ", "_")
    model_path = os.path.join("saved_models", model_name)
    model = joblib.load(model_path)
    
    # predict_proba returns sorted columns: class_a (1) is column 0, class_b (3) is column 1
    # Let's verify classes_ attribute order
    print("Model classes order:", model.classes_)
    # Hardhat = 1, NO-Hardhat = 3
    # Let's get probability for positive class (NO-Hardhat, i.e., index 1)
    probs_no_hardhat = model.predict_proba(X_val_pair)[:, 1]
    
    # Calculate AUC
    # We want to treat NO-Hardhat as 1 (positive) and Hardhat as 0 (negative)
    y_true_binary = (y_val_pair == class_b).astype(int)
    auc = roc_auc_score(y_true_binary, probs_no_hardhat)
    print(f"Hardhat classifier AUC-ROC Score: {auc:.4f}")
    
    # Check metrics at thresholds [0.1, 0.3, 0.5, 0.7, 0.9]
    # Prediction is NO-Hardhat if prob >= threshold
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        preds_t = np.where(probs_no_hardhat >= t, class_b, class_a)
        
        # Confusion matrix:
        # [[TN, FP], [FN, TP]]
        # where negative is Hardhat, positive is NO-Hardhat
        cm = confusion_matrix(y_val_pair, preds_t, labels=[class_a, class_b])
        tn, fp, fn, tp = cm.ravel()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_pair, preds_t, labels=[class_a, class_b]
        )
        
        print(f"Threshold: {t:.1f}")
        print(f"  Confusion Matrix: TN={tn}, FP={fp} (bare head predicted as helmet), FN={fn} (helmet predicted as bare), TP={tp}")
        print(f"  Hardhat: Recall={recall[0]:.4f} | NO-Hardhat: Precision={precision[1]:.4f} | Recall={recall[1]:.4f} | F1={f1[1]:.4f}")

if __name__ == "__main__":
    main()
