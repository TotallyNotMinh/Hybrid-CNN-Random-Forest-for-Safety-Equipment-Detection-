import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split

# 1. Load data
csv_path = "resnet_embeddings_filtered.csv"
print("Loading dataset...")
df = pd.read_csv(csv_path)

feature_cols = [col for col in df.columns if col.startswith("f")]
X = df[feature_cols].values
y = df["label"].values
class_names = df["class_name"].values

# Recreate stratified validation split (same random state)
_, X_val, _, y_val, _, name_val = train_test_split(
    X, y, class_names, test_size=0.2, random_state=42, stratify=y
)

# Load models
print("Loading trained models...")
six_class_model = joblib.load("saved_models/six_class_model.pkl")
glove_model = joblib.load("saved_models/Gloves_vs_NO-Gloves.pkl")

# Define crop types (body regions)
# Gloves (0, 2) -> Hand
# Hardhat (1, 3) -> Head
# Safety Vest (4, 5) -> Torso
def get_region(lbl):
    if lbl in [0, 2]:
        return 0  # Hand
    elif lbl in [1, 3]:
        return 1  # Head
    elif lbl in [4, 5]:
        return 2  # Torso
    return -1

regions_val = np.array([get_region(l) for l in y_val])

# We will select a subset of 1000 validation samples to speed up decision path tracking
np.random.seed(42)
sample_indices = np.random.choice(len(X_val), size=1000, replace=False)
X_sub = X_val[sample_indices]
regions_sub = regions_val[sample_indices]

# Analyze the first 20 trees in the forest
num_trees_to_analyze = 20
print(f"Analyzing node split paths for the first {num_trees_to_analyze} trees in the 6-class model...")

first_pure_depths = []

for tree_idx in range(num_trees_to_analyze):
    estimator = six_class_model.estimators_[tree_idx]
    tree = estimator.tree_
    n_nodes = tree.node_count
    children_left = tree.children_left
    children_right = tree.children_right
    
    # Calculate depth of each node in the tree using DFS
    node_depths = np.zeros(shape=n_nodes, dtype=int)
    stack = [(0, 0)]  # (node_id, depth)
    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depths[node_id] = depth
        if children_left[node_id] != children_right[node_id]:  # split node
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
            
    # Get decision paths for the subset
    indicator = estimator.decision_path(X_sub)
    
    # We want to identify the purity of all nodes
    # For speed, we populate the region distribution in each node
    node_counts = np.zeros((n_nodes, 3), dtype=int) # counts of Hand, Head, Torso
    
    # indicator is a CSR matrix of shape (n_samples, n_nodes)
    # Convert to CSC format for fast column slicing
    indicator_csc = indicator.tocsc()
    for node_id in range(n_nodes):
        sample_indices_in_node = indicator_csc[:, node_id].indices
        if len(sample_indices_in_node) > 0:
            node_regions = regions_sub[sample_indices_in_node]
            node_counts[node_id, 0] = np.sum(node_regions == 0)
            node_counts[node_id, 1] = np.sum(node_regions == 1)
            node_counts[node_id, 2] = np.sum(node_regions == 2)
            
    # Determine which nodes are "Body-Region Pure" (e.g. >= 95% of samples in the node belong to a single region)
    node_totals = node_counts.sum(axis=1)
    is_pure_node = np.zeros(n_nodes, dtype=bool)
    for node_id in range(n_nodes):
        total = node_totals[node_id]
        if total > 0:
            max_count = np.max(node_counts[node_id])
            if max_count / total >= 0.95:
                is_pure_node[node_id] = True
                
    # For each sample, trace its path and find the depth of the first pure node
    # indicator is CSR, so indicator[i].indices gives the nodes sample i visited
    for i in range(len(X_sub)):
        visited_nodes = indicator[i].indices
        # Sort visited nodes by depth to find the first one
        visited_depths = [(node, node_depths[node]) for node in visited_nodes]
        visited_depths.sort(key=lambda x: x[1])
        
        # Find the first node that is region-pure
        first_pure_depth = None
        for node, depth in visited_depths:
            if is_pure_node[node]:
                first_pure_depth = depth
                break
                
        if first_pure_depth is not None:
            first_pure_depths.append(first_pure_depth)

avg_pure_depth = np.mean(first_pure_depths)
print("\n" + "="*60)
print("EXPERIMENTAL RESULTS: TREE SPLIT PATH ANALYSIS")
print("="*60)
print(f"Average depth of first 'Body-Region Pure' node: {avg_pure_depth:.2f} splits")
print(f"Minimum depth of first pure node observed:       {np.min(first_pure_depths)} splits")
print(f"Maximum depth of first pure node observed:       {np.max(first_pure_depths)} splits")
print("-" * 60)
print("EXPLANATION:")
print(f"In the 6-class model, a tree must perform an average of {avg_pure_depth:.2f} levels of splits")
print("just to separate head, torso, and hand crops from each other.")
print("Only after reaching this depth can the tree begin to optimize compliance classification.")
print("In contrast, the binary Gloves model has 100% hand crops at depth 0 (root),")
print("meaning 100% of its tree capacity is used for compliance from the very first split.")
print("="*60)
