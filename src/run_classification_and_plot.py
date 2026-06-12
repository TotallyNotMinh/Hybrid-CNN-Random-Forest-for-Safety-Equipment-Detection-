import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import umap
import joblib
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

def plot_confusion_matrix(ax, cm, class_names, title):
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    # Show ticks and label them with class names
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')
            
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

def plot_umap_errors(ax, X_2d, y_true, preds, class_names, label_to_name, colors, title):
    correct = (preds == y_true)
    unique_labels = np.unique(y_true)
    
    # Plot correct predictions by class
    for label in unique_labels:
        mask_class = (y_true == label)
        mask_correct = mask_class & correct
        class_name = label_to_name[label]
        
        ax.scatter(
            X_2d[mask_correct, 0],
            X_2d[mask_correct, 1],
            s=8,
            alpha=0.6,
            label=f"{class_name} (Correct)",
            color=colors.get(class_name, "#9E9E9E"),
            edgecolors='none'
        )
        
    # Overlay incorrect predictions in vivid red
    incorrect = ~correct
    if np.sum(incorrect) > 0:
        ax.scatter(
            X_2d[incorrect, 0],
            X_2d[incorrect, 1],
            s=30,
            marker='x',
            color='#FF1744',
            label='Misclassified',
            linewidths=1.5
        )
        
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("UMAP Dimension 1", fontsize=10)
    ax.set_ylabel("UMAP Dimension 2", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(markerscale=1.5, loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=8)

def main():
    # 1. Load data
    csv_path = "resnet_embeddings_filtered.csv"
    print(f"Loading filtered embeddings from {csv_path}...")
    df = pd.read_csv(csv_path)
    print("Dataset shape:", df.shape)
    
    # Build a bidirectional lookup between label id and class name
    label_to_name = {}
    for label, name in zip(df["label"], df["class_name"]):
        label_to_name[label] = name
    name_to_label = {v: k for k, v in label_to_name.items()}
    
    # Sample equal class size
    df_sampled = sample_equal_classes(df)
    
    feature_cols = [col for col in df.columns if col.startswith("f")]
    
    # Setup subplots (3 tasks, each gets 1 confusion matrix & 1 UMAP error scatter)
    fig, axes = plt.subplots(3, 2, figsize=(18, 24), dpi=150)
    
    tasks = [
        ("Hardhat", "NO-Hardhat"),
        ("Safety Vest", "NO-Safety Vest"),
        ("Gloves", "NO-Gloves")
    ]
    
    colors = {
        "Hardhat": "#FFD700",       # Vivid Gold
        "NO-Hardhat": "#E53935",    # Soft Red
        "Safety Vest": "#4CAF50",   # Soft Green
        "NO-Safety Vest": "#FB8C00",# Dark Orange
        "Gloves": "#1E88E5",        # Soft Blue
        "NO-Gloves": "#8E24AA"      # Rich Purple
    }
    
    for idx, (class_a_name, class_b_name) in enumerate(tasks):
        print(f"\nEvaluating: {class_a_name} vs {class_b_name}...")
        
        class_a = name_to_label[class_a_name]
        class_b = name_to_label[class_b_name]
        
        # Slices dataset to just these two classes
        task_df = df_sampled[df_sampled["class_name"].isin([class_a_name, class_b_name])]
        
        X_task = task_df[feature_cols].values
        y_task = task_df["label"].values
        
        # Load pre-trained model
        model_name = f"{class_a_name}_vs_{class_b_name}.pkl".replace(" ", "_")
        model_path = os.path.join("saved_models", model_name)
        
        if not os.path.exists(model_path):
            print(f"Warning: Model not found at {model_path}. Skipping.")
            continue
            
        model = joblib.load(model_path)
        
        # Run inference
        preds = model.predict(X_task)
        
        # Calculate Confusion Matrix
        cm = confusion_matrix(y_task, preds, labels=[class_a, class_b])
        plot_confusion_matrix(
            axes[idx, 0], 
            cm, 
            [class_a_name, class_b_name], 
            f"{class_a_name} vs {class_b_name} Confusion Matrix"
        )
        
        # Dimensionality Reduction for Visualization
        print(f"[{class_a_name} vs {class_b_name}] Running PCA (512d -> 50d)...")
        pca = PCA(n_components=50, random_state=42)
        X_pca = pca.fit_transform(X_task)
        
        print(f"[{class_a_name} vs {class_b_name}] Running UMAP (50d -> 2d)...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
            n_jobs=-1
        )
        X_2d = reducer.fit_transform(X_pca)
        
        # Plot UMAP Errors
        plot_umap_errors(
            axes[idx, 1], 
            X_2d, 
            y_task, 
            preds, 
            task_df["class_name"].values, 
            label_to_name, 
            colors, 
            f"{class_a_name} vs {class_b_name} Error Map"
        )
        
    plt.suptitle("PPE Classifier Performance & Error Distribution (UMAP Space)", fontsize=18, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    os.makedirs("images", exist_ok=True)
    out_path = "images/classification_results.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    print(f"\nSuccessfully generated and saved results plot to {out_path}")

if __name__ == "__main__":
    main()
