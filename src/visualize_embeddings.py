import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
import os

def sample_equal_classes(df, random_state=42):
    class_counts = df["class_name"].value_counts()
    min_count = class_counts.min()
    print(f"Minimum class count in dataset: {min_count}")
    
    sampled_dfs = []
    for cls in class_counts.index:
        cls_df = df[df["class_name"] == cls]
        sampled_cls_df = cls_df.sample(n=min_count, random_state=random_state)
        sampled_dfs.append(sampled_cls_df)
        
    df_sampled = pd.concat(sampled_dfs, ignore_index=True)
    return df_sampled

def process_and_project(df, label_prefix=""):
    feature_cols = [col for col in df.columns if col.startswith("f")]
    X = df[feature_cols].values
    class_names = df["class_name"].values
    
    print(f"[{label_prefix}] Running PCA (512d -> 50d) on {len(X)} points...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    
    print(f"[{label_prefix}] Running UMAP (50d -> 2d)...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=-1
    )
    X_2d = reducer.fit_transform(X_pca)
    return X_2d, class_names

def main():
    # Colors for the 6 classes
    colors = {
        "Hardhat": "#FFD700",       # Vivid Gold
        "NO-Hardhat": "#E53935",    # Soft Red
        "Safety Vest": "#4CAF50",   # Soft Green
        "NO-Safety Vest": "#FB8C00",# Dark Orange
        "Gloves": "#1E88E5",        # Soft Blue
        "NO-Gloves": "#8E24AA"      # Rich Purple
    }
    
    # Create matplotlib subplots side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=150)
    
    # 1. Non-filtered embeddings
    print("=== Processing Non-Filtered Embeddings ===")
    df_raw = pd.read_csv("resnet_features.csv")
    print(f"Raw shape: {df_raw.shape}")
    df_raw_sampled = sample_equal_classes(df_raw)
    print("Raw sampled distribution:")
    print(df_raw_sampled["class_name"].value_counts())
    
    X_2d_raw, classes_raw = process_and_project(df_raw_sampled, "Non-Filtered")
    
    # Plot raw
    ax_raw = axes[0]
    for label in colors.keys():
        mask = classes_raw == label
        ax_raw.scatter(
            X_2d_raw[mask, 0],
            X_2d_raw[mask, 1],
            s=4,
            alpha=0.5,
            label=label,
            color=colors[label],
            edgecolors='none'
        )
    ax_raw.set_title("Non-Filtered Embeddings (Equal Samples per Class)", fontsize=14, fontweight="bold", pad=10)
    ax_raw.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax_raw.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax_raw.grid(True, linestyle="--", alpha=0.3)
    ax_raw.legend(markerscale=5, loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=True)
    
    # Free up memory
    del df_raw, df_raw_sampled, X_2d_raw
    
    # 2. Filtered embeddings
    print("\n=== Processing Filtered Embeddings ===")
    df_filt = pd.read_csv("resnet_embeddings_filtered.csv")
    print(f"Filtered shape: {df_filt.shape}")
    df_filt_sampled = sample_equal_classes(df_filt)
    print("Filtered sampled distribution:")
    print(df_filt_sampled["class_name"].value_counts())
    
    X_2d_filt, classes_filt = process_and_project(df_filt_sampled, "Filtered")
    
    # Plot filtered
    ax_filt = axes[1]
    for label in colors.keys():
        mask = classes_filt == label
        ax_filt.scatter(
            X_2d_filt[mask, 0],
            X_2d_filt[mask, 1],
            s=4,
            alpha=0.5,
            label=label,
            color=colors[label],
            edgecolors='none'
        )
    ax_filt.set_title("Filtered Embeddings (Equal Samples per Class)", fontsize=14, fontweight="bold", pad=10)
    ax_filt.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax_filt.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax_filt.grid(True, linestyle="--", alpha=0.3)
    ax_filt.legend(markerscale=5, loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=True)
    
    plt.suptitle("UMAP Embedding Space Comparison: Non-Filtered vs Outlier-Filtered", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs("images", exist_ok=True)
    out_path = "images/umap_comparison.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    print(f"\nSuccessfully saved comparison plot to {out_path}")

if __name__ == "__main__":
    main()
