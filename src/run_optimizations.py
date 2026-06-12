import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

# Load data
print("Loading resnet_embeddings_filtered.csv...")
df = pd.read_csv("resnet_embeddings_filtered.csv")
feature_cols = [col for col in df.columns if col.startswith("f")]
X = df[feature_cols].values
y = df["label"].values
class_names = df["class_name"].values

# ═════════════════════════════════════════════════════════
# 1. OPTIMIZATION FOR GLOVES: max_features=64
# ═════════════════════════════════════════════════════════
print("\n" + "="*50)
print("1. OPTIMIZATION FOR GLOVES: max_features=64")
print("="*50)

# Filter for Gloves task: label 0 (Gloves) and label 2 (NO-Gloves)
glove_mask = np.isin(y, [0, 2])
X_glove = X[glove_mask]
y_glove = y[glove_mask]

X_train_g, X_val_g, y_train_g, y_val_g = train_test_split(
    X_glove, y_glove, test_size=0.2, random_state=42, stratify=y_glove
)

# Baseline RF (default max_features='sqrt' which is sqrt(512) ≈ 22.6)
print("Training Baseline Gloves RF (max_features='sqrt')...")
rf_glove_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_glove_base.fit(X_train_g, y_train_g)
y_pred_g_base = rf_glove_base.predict(X_val_g)

# Tuned RF (max_features=64)
print("Training Optimized Gloves RF (max_features=64)...")
rf_glove_tuned = RandomForestClassifier(n_estimators=100, max_features=64, random_state=42, n_jobs=-1)
rf_glove_tuned.fit(X_train_g, y_train_g)
y_pred_g_tuned = rf_glove_tuned.predict(X_val_g)

# Compare validation metrics
acc_g_base = accuracy_score(y_val_g, y_pred_g_base)
f1_g_base = f1_score(y_val_g, y_pred_g_base, pos_label=0) # F1 for Gloves
rec_g_base = recall_score(y_val_g, y_pred_g_base, pos_label=0)

acc_g_tuned = accuracy_score(y_val_g, y_pred_g_tuned)
f1_g_tuned = f1_score(y_val_g, y_pred_g_tuned, pos_label=0)
rec_g_tuned = recall_score(y_val_g, y_pred_g_tuned, pos_label=0)

print(f"Gloves Baseline (max_features=sqrt):")
print(f"  Accuracy: {acc_g_base*100:.2f}% | Gloves F1: {f1_g_base*100:.2f}% | Gloves Recall: {rec_g_base*100:.2f}%")
print(f"Gloves Optimized (max_features=64):")
print(f"  Accuracy: {acc_g_tuned*100:.2f}% | Gloves F1: {f1_g_tuned*100:.2f}% | Gloves Recall: {rec_g_tuned*100:.2f}%")

# Save the tuned model
joblib.dump(rf_glove_tuned, "saved_models/Gloves_vs_NO-Gloves_tuned.pkl")
print("Saved optimized Gloves model to saved_models/Gloves_vs_NO-Gloves_tuned.pkl")


# ═════════════════════════════════════════════════════════
# 2. OPTIMIZATION FOR HARDHAT: Threshold Tuning (t=0.3)
# ═════════════════════════════════════════════════════════
print("\n" + "="*50)
print("2. OPTIMIZATION FOR HARDHAT: Threshold Tuning")
print("="*50)

# Filter for Hardhat task: label 1 (Hardhat) and label 3 (NO-Hardhat)
hardhat_mask = np.isin(y, [1, 3])
X_hh = X[hardhat_mask]
y_hh = y[hardhat_mask]

X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(
    X_hh, y_hh, test_size=0.2, random_state=42, stratify=y_hh
)

# Load baseline model
hh_model = joblib.load("saved_models/Hardhat_vs_NO-Hardhat.pkl")
classes_h = list(hh_model.classes_)
print(f"Hardhat model classes order: {classes_h}")

# Predict probabilities
probs_h = hh_model.predict_proba(X_val_h) # shape (N, 2)
# Under standard threshold t=0.5:
y_pred_h_base = hh_model.predict(X_val_h)

# Under t=0.3 for the minority class (NO-Hardhat, label 3)
# Let's verify index of label 3
idx_no_hardhat = classes_h.index(3) # typically 1
probs_no_hardhat = probs_h[:, idx_no_hardhat]

# If prob(NO-Hardhat) >= 0.3, predict 3, else predict 1
y_pred_h_tuned = np.where(probs_no_hardhat >= 0.3, 3, 1)

# Compare validation metrics
acc_h_base = accuracy_score(y_val_h, y_pred_h_base)
rec_nh_base = recall_score(y_val_h, y_pred_h_base, pos_label=3) # Recall for NO-Hardhat
f1_nh_base = f1_score(y_val_h, y_pred_h_base, pos_label=3)

acc_h_tuned = accuracy_score(y_val_h, y_pred_h_tuned)
rec_nh_tuned = recall_score(y_val_h, y_pred_h_tuned, pos_label=3)
f1_nh_tuned = f1_score(y_val_h, y_pred_h_tuned, pos_label=3)

print(f"Hardhat Baseline (t=0.5):")
print(f"  Accuracy: {acc_h_base*100:.2f}% | NO-Hardhat Recall: {rec_nh_base*100:.2f}% | NO-Hardhat F1: {f1_nh_base*100:.2f}%")
print(f"Hardhat Optimized (t=0.3 for NO-Hardhat):")
print(f"  Accuracy: {acc_h_tuned*100:.2f}% | NO-Hardhat Recall: {rec_nh_tuned*100:.2f}% | NO-Hardhat F1: {f1_nh_tuned*100:.2f}%")
