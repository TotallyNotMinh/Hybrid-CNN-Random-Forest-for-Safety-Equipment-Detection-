import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import umap
import joblib
import os

def plot_confusion_matrix(ax, cm, class_names, title):
    # Normalize the confusion matrix by rows (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Show ticks and label them with class names
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(class_names, fontsize=9)
    
    # Loop over data dimensions and create text annotations
    # We display both the raw count and the percentage
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_str = f"{count}\n({pct:.1f}%)"
            ax.text(j, i, text_str,
                    ha="center", va="center",
                    color="white" if pct/100.0 > thresh else "black",
                    fontsize=9, fontweight='bold')
            
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')

def plot_umap_space(ax, X_2d, y_true, preds, class_names, label_to_name, colors, title):
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
            s=12,
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
            s=40,
            marker='x',
            color='#FF1744',
            label='Misclassified',
            linewidths=1.5
        )
        
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(markerscale=1.5, loc="upper right", frameon=True, facecolor="white", edgecolor="none", shadow=True, fontsize=8)

def main():
    # 1. Load data
    csv_path = "resnet_embeddings_filtered.csv"
    print(f"Loading filtered embeddings from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    feature_cols = [col for col in df.columns if col.startswith("f")]
    X = df[feature_cols].values
    y = df["label"].values
    class_names = df["class_name"].values
    
    # 80/20 stratified split
    _, X_val, _, y_val, _, name_val = train_test_split(
        X, y, class_names, test_size=0.2, random_state=42, stratify=y
    )
    
    # Bidirectional lookups
    label_to_name = {}
    for label, name in zip(df["label"], df["class_name"]):
        label_to_name[label] = name
    name_to_label = {v: k for k, v in label_to_name.items()}
    
    unique_labels = sorted(list(label_to_name.keys()))
    unique_names = [label_to_name[lbl] for lbl in unique_labels]
    
    # 2. Load model
    model_path = os.path.join("saved_models", "six_class_model.pkl")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train_6class.py first.")
        return
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # 3. Predict on full validation set
    print("Running predictions on validation split...")
    preds = model.predict(X_val)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_val, preds, labels=unique_labels)
    
    # 4. Prepare data for UMAP plot (subsample validation set to keep it clean and fast)
    # We sample 500 instances per class from the validation split (or less if class size < 500)
    print("Sampling validation split for UMAP visualization...")
    sampled_indices = []
    np.random.seed(42)
    for lbl in unique_labels:
        lbl_indices = np.where(y_val == lbl)[0]
        sample_size = min(500, len(lbl_indices))
        choice = np.random.choice(lbl_indices, size=sample_size, replace=False)
        sampled_indices.extend(choice)
        
    X_sub = X_val[sampled_indices]
    y_sub = y_val[sampled_indices]
    preds_sub = preds[sampled_indices]
    
    # Dimensionality reduction
    print("Running PCA (512d -> 50d)...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_sub)
    
    print("Running UMAP (50d -> 2d)...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        n_jobs=-1
    )
    X_2d = reducer.fit_transform(X_pca)
    
    # 5. Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 9), dpi=150)
    
    # Left: Confusion Matrix
    plot_confusion_matrix(axes[0], cm, unique_names, "6-Class Classification Confusion Matrix (Validation Split)")
    
    # Right: UMAP Error Map
    colors = {
        "Hardhat": "#FFD700",       # Vivid Gold
        "NO-Hardhat": "#E53935",    # Soft Red
        "Safety Vest": "#4CAF50",   # Soft Green
        "NO-Safety Vest": "#FB8C00",# Dark Orange
        "Gloves": "#1E88E5",        # Soft Blue
        "NO-Gloves": "#8E24AA"      # Rich Purple
    }
    plot_umap_space(
        axes[1], 
        X_2d, 
        y_sub, 
        preds_sub, 
        unique_names, 
        label_to_name, 
        colors, 
        "6-Class UMAP Semantic Embedding Space & Misclassifications"
    )
    
    plt.suptitle("6-Class Multiclass Random Forest Classifier Evaluation", fontsize=18, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs("images", exist_ok=True)
    out_path = "images/six_class_classification_results.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    print(f"\nSuccessfully generated and saved 6-class results plot to {out_path}")

if __name__ == "__main__":
    main()
