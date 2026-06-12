import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier

def main():
    # Load datasets
    raw_path = "resnet_features.csv"
    filt_path = "resnet_embeddings_filtered.csv"
    
    if not os.path.exists(raw_path) or not os.path.exists(filt_path):
        print("Missing dataset files.")
        return
        
    print("Loading datasets...")
    df_raw = pd.read_csv(raw_path)
    df_filt = pd.read_csv(filt_path)
    
    # 1. Quantify outlier filtering per class
    print("\n=== OUTLIER FILTERING QUANTIFICATION ===")
    raw_counts = df_raw["class_name"].value_counts()
    filt_counts = df_filt["class_name"].value_counts()
    
    outlier_df = pd.DataFrame({
        "Raw Count": raw_counts,
        "Filtered Count": filt_counts
    })
    outlier_df["Removed"] = outlier_df["Raw Count"] - outlier_df["Filtered Count"]
    outlier_df["% Removed"] = (outlier_df["Removed"] / outlier_df["Raw Count"]) * 100
    print(outlier_df)

    # 2. Extract split identical to train.py
    feature_cols = [col for col in df_filt.columns if col.startswith("f")]
    X = df_filt[feature_cols].values
    y = df_filt["label"].values
    class_names = df_filt["class_name"].values
    
    X_train, X_val, y_train, y_val, name_train, name_val = train_test_split(
        X, y, class_names, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build a bidirectional lookup
    label_to_name = {}
    for label, name in zip(df_filt["label"], df_filt["class_name"]):
        label_to_name[label] = name
    name_to_label = {v: k for k, v in label_to_name.items()}
    
    tasks = [
        ("Hardhat", "NO-Hardhat"),
        ("Safety Vest", "NO-Safety Vest"),
        ("Gloves", "NO-Gloves")
    ]
    
    # 3. Evaluate models on validation split
    print("\n=== EVALUATING PRE-TRAINED MODELS ON IMBA_VALIDATION ===")
    for class_a_name, class_b_name in tasks:
        class_a = name_to_label[class_a_name]
        class_b = name_to_label[class_b_name]
        
        # Slices validation
        val_mask = np.isin(y_val, [class_a, class_b])
        X_val_pair = X_val[val_mask]
        y_val_pair = y_val[val_mask]
        
        model_name = f"{class_a_name}_vs_{class_b_name}.pkl".replace(" ", "_")
        model_path = os.path.join("saved_models", model_name)
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found.")
            continue
            
        model = joblib.load(model_path)
        preds = model.predict(X_val_pair)
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val_pair, preds, labels=[class_a, class_b], average=None
        )
        cm = confusion_matrix(y_val_pair, preds, labels=[class_a, class_b])
        
        print(f"\nTask: {class_a_name} vs {class_b_name}")
        print(f"Confusion Matrix:\n{cm}")
        print(f"Class: {class_a_name} | Precision: {precision[0]:.4f} | Recall: {recall[0]:.4f} | F1: {f1[0]:.4f}")
        print(f"Class: {class_b_name} | Precision: {precision[1]:.4f} | Recall: {recall[1]:.4f} | F1: {f1[1]:.4f}")

    # 4. Compare class_weight='balanced' vs default
    print("\n=== COMPARING CLASS WEIGHTS ON MINORITY CLASS (NO-Safety Vest) ===")
    class_a_name, class_b_name = "Safety Vest", "NO-Safety Vest"
    class_a = name_to_label[class_a_name]
    class_b = name_to_label[class_b_name]
    
    train_mask = np.isin(y_train, [class_a, class_b])
    X_train_pair = X_train[train_mask]
    y_train_pair = y_train[train_mask]
    
    val_mask = np.isin(y_val, [class_a, class_b])
    X_val_pair = X_val[val_mask]
    y_val_pair = y_val[val_mask]
    
    # Train default model
    print("Training default model (class_weight=None)...")
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_def.fit(X_train_pair, y_train_pair)
    preds_def = rf_def.predict(X_val_pair)
    
    # Train balanced model
    print("Training balanced model (class_weight='balanced')...")
    rf_bal = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_bal.fit(X_train_pair, y_train_pair)
    preds_bal = rf_bal.predict(X_val_pair)
    
    # Compute recall specifically for minority class (NO-Safety Vest = class_b)
    _, recall_def, f1_def, _ = precision_recall_fscore_support(y_val_pair, preds_def, labels=[class_a, class_b])
    _, recall_bal, f1_bal, _ = precision_recall_fscore_support(y_val_pair, preds_bal, labels=[class_a, class_b])
    
    print("\nResults for Safety Vest vs NO-Safety Vest:")
    print(f"Default (None):  {class_b_name} Recall = {recall_def[1]:.4f} | F1 = {f1_def[1]:.4f}")
    print(f"Balanced (bal):  {class_b_name} Recall = {recall_bal[1]:.4f} | F1 = {f1_bal[1]:.4f}")

    # 5. Feature Importance Analysis
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    for class_a_name, class_b_name in tasks:
        model_name = f"{class_a_name}_vs_{class_b_name}.pkl".replace(" ", "_")
        model_path = os.path.join("saved_models", model_name)
        if not os.path.exists(model_path):
            continue
        model = joblib.load(model_path)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print(f"\nTask: {class_a_name} vs {class_b_name} Top 5 Features:")
        for i in range(5):
            print(f"  Feature f{indices[i]}: Weight = {importances[indices[i]]:.4f}")

if __name__ == "__main__":
    main()
