import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap  

# 1. Load data
print("Loading embeddings...")
df = pd.read_csv("resnet_features.csv")
feature_cols = [col for col in df.columns if col.startswith("f")]
X = df[feature_cols].values
class_names = df["class_name"].values

# 2. Stage 1: PCA to 50 dimensions
print("Running PCA (512d -> 50d)...")
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)

# 3. Stage 2: UMAP to 2 dimensions
print("Running UMAP (50d -> 2d)...")
reducer = umap.UMAP(
    n_neighbors=15, 
    min_dist=0.1, 
    metric="cosine",  # Cosine distance is often better for embeddings
    random_state=None,
    n_jobs=-1
)
X_2d = reducer.fit_transform(X_pca)

# 4. Plot & Save Visualization
df_plot = pd.DataFrame(X_2d, columns=["Component 1", "Component 2"])
df_plot["class"] = class_names

# Define contrasting pairs for class vs NO-class labels
colors = {
    "Hardhat": "#FFD700",       # Yellow
    "NO-Hardhat": "#D32F2F",    # Red
    "Safety Vest": "#388E3C",   # Green
    "NO-Safety Vest": "#F57C00",# Orange
    "Gloves": "#1976D2",        # Blue
    "NO-Gloves": "#7B1FA2"      # Purple
}

plt.figure(figsize=(12, 9), dpi=150)
for label, group in df_plot.groupby("class"):
    plt.scatter(
        group["Component 1"], 
        group["Component 2"], 
        s=1, 
        alpha=0.3, 
        label=label, 
        color=colors.get(label, "#757575")
    )

plt.title("UMAP Projection of 512-D ResNet18 Embeddings", fontsize=14, fontweight="bold")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(markerscale=10, loc="upper right")
plt.tight_layout()
plt.savefig("images/umap_embeddings.png")
print("Saved visualization to images/umap_embeddings_non_filtered.png")