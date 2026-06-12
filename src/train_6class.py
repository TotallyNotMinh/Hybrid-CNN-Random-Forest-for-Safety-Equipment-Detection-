import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the outlier-filtered embeddings
CSV_PATH = "resnet_embeddings_filtered.csv"
print(f"Loading dataset from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print("Dataset shape:", df.shape)

# Pull out only the feature columns (f0..f511)
feature_cols = [col for col in df.columns if col.startswith("f")]
X = df[feature_cols].values
y = df["label"].values
class_names = df["class_name"].values

# 80/20 split, stratified so each class keeps its proportion
X_train, X_val, y_train, y_val, name_train, name_val = train_test_split(
    X,
    y,
    class_names,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrain size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")

# Build bidirectional lookup
label_to_name = {}
for label, name in zip(df["label"], df["class_name"]):
    label_to_name[label] = name
name_to_label = {v: k for k, v in label_to_name.items()}

# Get unique labels sorted
unique_labels = sorted(list(label_to_name.keys()))
unique_names = [label_to_name[lbl] for lbl in unique_labels]

print("\nClasses in order of label ID:")
for lbl, name in zip(unique_labels, unique_names):
    print(f"  Label {lbl}: {name}")

# Train the single 6-class Random Forest Classifier
print("\nTraining 6-class Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Save the model
os.makedirs("saved_models", exist_ok=True)
save_path = os.path.join("saved_models", "six_class_model.pkl")
joblib.dump(model, save_path)
print(f"Saved 6-class model to: {save_path}")

# Evaluate
preds = model.predict(X_val)
acc = accuracy_score(y_val, preds)
print(f"\nOverall Validation Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        preds,
        labels=unique_labels,
        target_names=unique_names,
        digits=4
    )
)

print("\nConfusion Matrix:")
cm = confusion_matrix(y_val, preds, labels=unique_labels)
print(cm)

# Task-constrained validation evaluation
# Let's see how it compares to binary if we only evaluate on specific task subsets.
print("\n" + "="*50)
print("TASK-SPECIFIC SUBSET EVALUATION (UNCONSTRAINED)")
print("="*50)

# 1. Hardhat vs NO-Hardhat subset
hardhat_labels = [name_to_label["Hardhat"], name_to_label["NO-Hardhat"]]
hardhat_mask = np.isin(y_val, hardhat_labels)
y_val_hh = y_val[hardhat_mask]
preds_hh = preds[hardhat_mask]
print("\nHardhat vs NO-Hardhat Subset:")
print(f"Subset Size: {len(y_val_hh)}")
# Check how many are misclassified as vest/gloves classes (label not in 1, 3)
mis_hh = ~np.isin(preds_hh, hardhat_labels)
print(f"Misclassified into other task categories: {np.sum(mis_hh)} ({np.mean(mis_hh)*100:.2f}%)")
# Calculate metrics within the subset
hh_acc = np.mean(y_val_hh == preds_hh)
print(f"Subset accuracy (unconstrained): {hh_acc:.4f}")
print("Confusion Matrix on Head Crops (labels: Hardhat, NO-Hardhat):")
cm_hh = confusion_matrix(y_val_hh, preds_hh, labels=hardhat_labels)
print(cm_hh)

# 2. Safety Vest vs NO-Safety Vest subset
vest_labels = [name_to_label["Safety Vest"], name_to_label["NO-Safety Vest"]]
vest_mask = np.isin(y_val, vest_labels)
y_val_vest = y_val[vest_mask]
preds_vest = preds[vest_mask]
print("\nSafety Vest vs NO-Safety Vest Subset:")
print(f"Subset Size: {len(y_val_vest)}")
mis_vest = ~np.isin(preds_vest, vest_labels)
print(f"Misclassified into other task categories: {np.sum(mis_vest)} ({np.mean(mis_vest)*100:.2f}%)")
vest_acc = np.mean(y_val_vest == preds_vest)
print(f"Subset accuracy (unconstrained): {vest_acc:.4f}")
print("Confusion Matrix on Torso Crops (labels: Safety Vest, NO-Safety Vest):")
cm_vest = confusion_matrix(y_val_vest, preds_vest, labels=vest_labels)
print(cm_vest)

# 3. Gloves vs NO-Gloves subset
glove_labels = [name_to_label["Gloves"], name_to_label["NO-Gloves"]]
glove_mask = np.isin(y_val, glove_labels)
y_val_glove = y_val[glove_mask]
preds_glove = preds[glove_mask]
print("\nGloves vs NO-Gloves Subset:")
print(f"Subset Size: {len(y_val_glove)}")
mis_glove = ~np.isin(preds_glove, glove_labels)
print(f"Misclassified into other task categories: {np.sum(mis_glove)} ({np.mean(mis_glove)*100:.2f}%)")
glove_acc = np.mean(y_val_glove == preds_glove)
print(f"Subset accuracy (unconstrained): {glove_acc:.4f}")
print("Confusion Matrix on Hand Crops (labels: Gloves, NO-Gloves):")
cm_glove = confusion_matrix(y_val_glove, preds_glove, labels=glove_labels)
print(cm_glove)


print("\n" + "="*50)
print("TASK-CONSTRAINED SUBSET EVALUATION (PROBABILITIES)")
print("="*50)
# Here we take the model's predict_proba outputs and restrict the search space to the task classes.
probs = model.predict_proba(X_val) # Shape: (val_size, 6)

# 1. Hardhat
hardhat_idx = [unique_labels.index(l) for l in hardhat_labels] # indices of Hardhat (1) and NO-Hardhat (3) in model.classes_
probs_hh_subset = probs[hardhat_mask][:, hardhat_idx]
# Normalize probabilities to sum to 1 over the subset
probs_hh_subset_norm = probs_hh_subset / probs_hh_subset.sum(axis=1, keepdims=True)
# Select class with higher probability
preds_hh_constrained = np.array(hardhat_labels)[np.argmax(probs_hh_subset_norm, axis=1)]
acc_hh_const = np.mean(y_val_hh == preds_hh_constrained)
print(f"\nHardhat Task-Constrained Accuracy: {acc_hh_const:.4f}")
print("Confusion Matrix (Constrained):")
print(confusion_matrix(y_val_hh, preds_hh_constrained, labels=hardhat_labels))

# 2. Safety Vest
vest_idx = [unique_labels.index(l) for l in vest_labels] # indices of Safety Vest (5) and NO-Safety Vest (4)
probs_vest_subset = probs[vest_mask][:, vest_idx]
probs_vest_subset_norm = probs_vest_subset / probs_vest_subset.sum(axis=1, keepdims=True)
preds_vest_constrained = np.array(vest_labels)[np.argmax(probs_vest_subset_norm, axis=1)]
acc_vest_const = np.mean(y_val_vest == preds_vest_constrained)
print(f"\nSafety Vest Task-Constrained Accuracy: {acc_vest_const:.4f}")
print("Confusion Matrix (Constrained):")
print(confusion_matrix(y_val_vest, preds_vest_constrained, labels=vest_labels))

# 3. Gloves
glove_idx = [unique_labels.index(l) for l in glove_labels] # indices of Gloves (0) and NO-Gloves (2)
probs_glove_subset = probs[glove_mask][:, glove_idx]
probs_glove_subset_norm = probs_glove_subset / probs_glove_subset.sum(axis=1, keepdims=True)
preds_glove_constrained = np.array(glove_labels)[np.argmax(probs_glove_subset_norm, axis=1)]
acc_glove_const = np.mean(y_val_glove == preds_glove_constrained)
print(f"\nGloves Task-Constrained Accuracy: {acc_glove_const:.4f}")
print("Confusion Matrix (Constrained):")
print(confusion_matrix(y_val_glove, preds_glove_constrained, labels=glove_labels))
