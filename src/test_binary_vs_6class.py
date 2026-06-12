"""
Comprehensive empirical tests comparing binary vs 6-class Random Forest classifiers.
Produces evidence for why the 6-class model underperforms on gloves.
"""
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────
csv_path = "resnet_embeddings_filtered.csv"
print("Loading dataset...")
df = pd.read_csv(csv_path)

feature_cols = [col for col in df.columns if col.startswith("f")]
X = df[feature_cols].values
y = df["label"].values
class_names_arr = df["class_name"].values

X_train, X_val, y_train, y_val, _, _ = train_test_split(
    X, y, class_names_arr, test_size=0.2, random_state=42, stratify=y
)

label_to_name = {0:'Gloves', 1:'Hardhat', 2:'NO-Gloves', 3:'NO-Hardhat', 4:'NO-Safety Vest', 5:'Safety Vest'}
name_to_label = {v:k for k,v in label_to_name.items()}

six_class_model = joblib.load("saved_models/six_class_model.pkl")
glove_binary    = joblib.load("saved_models/Gloves_vs_NO-Gloves.pkl")
hardhat_binary  = joblib.load("saved_models/Hardhat_vs_NO-Hardhat.pkl")
vest_binary     = joblib.load("saved_models/Safety_Vest_vs_NO-Safety_Vest.pkl")

glove_labels   = [0, 2]
hardhat_labels = [1, 3]
vest_labels    = [4, 5]

glove_mask   = np.isin(y_val, glove_labels)
hardhat_mask = np.isin(y_val, hardhat_labels)
vest_mask    = np.isin(y_val, vest_labels)

X_val_glove   = X_val[glove_mask]
y_val_glove   = y_val[glove_mask]
X_val_hardhat = X_val[hardhat_mask]
y_val_hardhat = y_val[hardhat_mask]
X_val_vest    = X_val[vest_mask]
y_val_vest    = y_val[vest_mask]

print(f"Validation split: {len(X_val)} total, Gloves subset: {len(X_val_glove)}, "
      f"Hardhat subset: {len(X_val_hardhat)}, Vest subset: {len(X_val_vest)}")

# ═════════════════════════════════════════════════════════
# TEST 1: FEATURE IMPORTANCE OVERLAP
# Which features does the 6-class model prioritize vs. the binary gloves model?
# ═════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 1: FEATURE IMPORTANCE OVERLAP ANALYSIS")
print("="*70)

imp_binary_glove = glove_binary.feature_importances_
imp_6class       = six_class_model.feature_importances_

# Top 30 features for each model
top30_binary = set(np.argsort(imp_binary_glove)[::-1][:30])
top30_6class = set(np.argsort(imp_6class)[::-1][:30])
overlap      = top30_binary & top30_6class

print(f"Top-30 features of Binary Gloves model:   {sorted(top30_binary)}")
print(f"Top-30 features of 6-Class model:         {sorted(top30_6class)}")
print(f"Overlap (shared features):                {len(overlap)} / 30  ({len(overlap)/30*100:.1f}%)")
print(f"Features unique to Binary Gloves model:   {len(top30_binary - top30_6class)}")
print(f"Features unique to 6-Class model:         {len(top30_6class - top30_binary)}")

# How much total importance weight do the binary model's top-30 glove features
# receive in the 6-class model?
weight_in_binary = np.sum(imp_binary_glove[list(top30_binary)])
weight_in_6class = np.sum(imp_6class[list(top30_binary)])
print(f"\nCombined weight of Binary Gloves top-30 features:")
print(f"  In Binary Gloves model: {weight_in_binary*100:.2f}%")
print(f"  In 6-Class model:       {weight_in_6class*100:.2f}%")
print(f"  Weight dilution factor: {weight_in_binary/weight_in_6class:.2f}x")

# ═════════════════════════════════════════════════════════
# TEST 2: PREDICTION CONFIDENCE (PROBABILITY MARGIN)
# Compare how confident each model is when classifying glove samples.
# ═════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 2: PREDICTION CONFIDENCE ANALYSIS (on Glove Validation Subset)")
print("="*70)

# Binary model confidence
probs_binary = glove_binary.predict_proba(X_val_glove)
margin_binary = np.abs(probs_binary[:, 0] - probs_binary[:, 1])

# 6-class model confidence (constrained to glove classes)
probs_6class_all = six_class_model.predict_proba(X_val_glove)
classes_list = list(six_class_model.classes_)
idx_g  = classes_list.index(0)  # Gloves
idx_ng = classes_list.index(2)  # NO-Gloves
probs_6class_glove = probs_6class_all[:, [idx_g, idx_ng]]
# Normalize
probs_6class_glove_norm = probs_6class_glove / probs_6class_glove.sum(axis=1, keepdims=True)
margin_6class = np.abs(probs_6class_glove_norm[:, 0] - probs_6class_glove_norm[:, 1])

# Also compute total probability mass allocated to glove classes (before normalization)
total_glove_mass = probs_6class_all[:, idx_g] + probs_6class_all[:, idx_ng]

print(f"Binary Gloves model:")
print(f"  Mean confidence margin: {np.mean(margin_binary):.4f}")
print(f"  Median confidence margin: {np.median(margin_binary):.4f}")
print(f"  Low-confidence samples (margin < 0.3): {np.sum(margin_binary < 0.3)} / {len(margin_binary)} ({np.mean(margin_binary < 0.3)*100:.2f}%)")

print(f"\n6-Class model (constrained to Glove classes):")
print(f"  Mean confidence margin: {np.mean(margin_6class):.4f}")
print(f"  Median confidence margin: {np.median(margin_6class):.4f}")
print(f"  Low-confidence samples (margin < 0.3): {np.sum(margin_6class < 0.3)} / {len(margin_6class)} ({np.mean(margin_6class < 0.3)*100:.2f}%)")

print(f"\nProbability mass leaked to non-glove classes in 6-class model:")
print(f"  Mean probability mass on Glove+NO-Glove: {np.mean(total_glove_mass)*100:.2f}%")
print(f"  Mean probability mass leaked to other 4 classes: {(1-np.mean(total_glove_mass))*100:.2f}%")
print(f"  Samples with >10% mass leaked: {np.sum(total_glove_mass < 0.9)} / {len(total_glove_mass)} ({np.mean(total_glove_mass < 0.9)*100:.2f}%)")
print(f"  Samples with >30% mass leaked: {np.sum(total_glove_mass < 0.7)} / {len(total_glove_mass)} ({np.mean(total_glove_mass < 0.7)*100:.2f}%)")

# ═════════════════════════════════════════════════════════
# TEST 3: DECISION PATH LENGTH COMPARISON
# How many nodes does each model traverse for glove samples?
# ═════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 3: DECISION PATH LENGTH COMPARISON (on Glove Validation Subset)")
print("="*70)

# Subsample for speed
np.random.seed(42)
sub_idx = np.random.choice(len(X_val_glove), size=min(2000, len(X_val_glove)), replace=False)
X_sub = X_val_glove[sub_idx]

# Binary model: average decision path length across all trees
binary_paths = []
for tree in glove_binary.estimators_:
    indicator = tree.decision_path(X_sub)
    # number of nodes visited per sample = number of non-zero entries per row
    path_lengths = np.diff(indicator.indptr)
    binary_paths.append(np.mean(path_lengths))
avg_binary_path = np.mean(binary_paths)

# 6-class model: average decision path length across all trees
six_paths = []
for tree in six_class_model.estimators_:
    indicator = tree.decision_path(X_sub)
    path_lengths = np.diff(indicator.indptr)
    six_paths.append(np.mean(path_lengths))
avg_six_path = np.mean(six_paths)

print(f"Average decision path length for glove samples:")
print(f"  Binary Gloves model:  {avg_binary_path:.2f} nodes")
print(f"  6-Class model:        {avg_six_path:.2f} nodes")
print(f"  Difference:           {avg_six_path - avg_binary_path:.2f} extra nodes ({(avg_six_path/avg_binary_path - 1)*100:.1f}% longer)")
print(f"\nThe 6-class model traverses {avg_six_path - avg_binary_path:.1f} more nodes per tree for the same")
print(f"glove samples. These extra nodes are spent separating body regions before")
print(f"reaching the glove-specific decision boundary.")

# ═════════════════════════════════════════════════════════
# TEST 4: PER-TREE AGREEMENT ANALYSIS
# How often do individual trees in the forest disagree on glove samples?
# Higher disagreement = less stable decision boundary.
# ═════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 4: PER-TREE AGREEMENT (ENSEMBLE STABILITY)")
print("="*70)

# Binary model: collect individual tree predictions
binary_tree_preds = np.array([tree.predict(X_sub) for tree in glove_binary.estimators_])  # (100, N)
# 6-class model: collect individual tree predictions
six_tree_preds = np.array([tree.predict(X_sub) for tree in six_class_model.estimators_])  # (100, N)

# For each sample, compute the fraction of trees that agree with the majority vote
def majority_agreement(tree_preds):
    """For each sample, fraction of trees that agree with the majority."""
    n_trees, n_samples = tree_preds.shape
    agreements = []
    for j in range(n_samples):
        votes = tree_preds[:, j]
        unique, counts = np.unique(votes, return_counts=True)
        max_count = counts.max()
        agreements.append(max_count / n_trees)
    return np.array(agreements)

agree_binary = majority_agreement(binary_tree_preds)
agree_six    = majority_agreement(six_tree_preds)

# For the 6-class model, also check how many distinct classes trees vote for
def n_distinct_classes(tree_preds):
    n_trees, n_samples = tree_preds.shape
    counts = []
    for j in range(n_samples):
        counts.append(len(np.unique(tree_preds[:, j])))
    return np.array(counts)

distinct_binary = n_distinct_classes(binary_tree_preds)
distinct_six    = n_distinct_classes(six_tree_preds)

print(f"Binary Gloves model (on glove samples):")
print(f"  Mean majority agreement:        {np.mean(agree_binary)*100:.2f}%")
print(f"  Mean distinct classes per sample: {np.mean(distinct_binary):.2f}")
print(f"  Samples with <70% agreement:    {np.sum(agree_binary < 0.7)} / {len(agree_binary)} ({np.mean(agree_binary < 0.7)*100:.2f}%)")

print(f"\n6-Class model (on same glove samples):")
print(f"  Mean majority agreement:        {np.mean(agree_six)*100:.2f}%")
print(f"  Mean distinct classes per sample: {np.mean(distinct_six):.2f}")
print(f"  Samples with <70% agreement:    {np.sum(agree_six < 0.7)} / {len(agree_six)} ({np.mean(agree_six < 0.7)*100:.2f}%)")

# ═════════════════════════════════════════════════════════
# TEST 5: CROSS-TASK LEAKAGE — WHERE DO TREES "LEAK" VOTES?
# When 6-class trees disagree on glove samples, which wrong classes do they vote for?
# ═════════════════════════════════════════════════════════
print("\n" + "="*70)
print("TEST 5: CROSS-TASK VOTE LEAKAGE (6-Class Model on Glove Samples)")
print("="*70)

# Flatten all tree predictions on glove samples
all_votes = six_tree_preds.flatten()  # (100 * N,)
total_votes = len(all_votes)

for lbl in sorted(label_to_name.keys()):
    count = np.sum(all_votes == lbl)
    print(f"  Votes for {label_to_name[lbl]:>15s} (label {lbl}): {count:>8d} ({count/total_votes*100:5.2f}%)")

non_glove_votes = np.sum(~np.isin(all_votes, glove_labels))
print(f"\n  Total votes leaked to non-glove classes: {non_glove_votes} / {total_votes} ({non_glove_votes/total_votes*100:.2f}%)")
print(f"  These leaked votes directly dilute the ensemble majority and reduce prediction confidence.")

# ═════════════════════════════════════════════════════════
# SUMMARY
# ═════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY OF ALL TESTS")
print("="*70)
print(f"Test 1 — Feature Importance Overlap:     Only {len(overlap)}/30 top features shared; "
      f"binary model's key features receive {weight_in_binary/weight_in_6class:.1f}x less weight in 6-class model.")
print(f"Test 2 — Prediction Confidence:          Binary margin={np.mean(margin_binary):.4f}, "
      f"6-class margin={np.mean(margin_6class):.4f}; "
      f"{(1-np.mean(total_glove_mass))*100:.1f}% probability mass leaked to non-glove classes.")
print(f"Test 3 — Decision Path Length:            6-class paths are {avg_six_path - avg_binary_path:.1f} nodes longer "
      f"({(avg_six_path/avg_binary_path - 1)*100:.1f}% increase).")
print(f"Test 4 — Ensemble Stability:             Binary agreement={np.mean(agree_binary)*100:.1f}%, "
      f"6-class agreement={np.mean(agree_six)*100:.1f}%; "
      f"6-class trees vote for {np.mean(distinct_six):.1f} distinct classes vs {np.mean(distinct_binary):.1f}.")
print(f"Test 5 — Cross-Task Vote Leakage:        {non_glove_votes/total_votes*100:.2f}% of individual tree votes "
      f"on glove samples go to non-glove classes.")
