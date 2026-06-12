import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def main():
    csv_path = "resnet_embeddings_filtered.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Calculate counts
    counts = df["class_name"].value_counts()
    print("Class counts in filtered dataset:")
    for cls, count in counts.items():
        print(f"  {cls}: {count}")
        
    # Sort classes by frequency in descending order
    counts = counts.sort_values(ascending=False)
    
    # Plotting
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Professional color scheme matching the project color palette
    colors = {
        "Hardhat": "#FFD700",       # Vivid Gold
        "NO-Hardhat": "#E53935",    # Soft Red
        "Safety Vest": "#4CAF50",   # Soft Green
        "NO-Safety Vest": "#FB8C00",# Dark Orange
        "Gloves": "#1E88E5",        # Soft Blue
        "NO-Gloves": "#8E24AA"      # Rich Purple
    }
    
    bar_colors = [colors.get(cls, "#9E9E9E") for cls in counts.index]
    
    bars = plt.bar(counts.index, counts.values, color=bar_colors, edgecolor="none", width=0.6)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + max(counts.values)*0.015,
            f"{height:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
        
    plt.title("Filtered Dataset Class Distribution (resnet_embeddings_filtered.csv)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Number of Instances", fontsize=11)
    plt.xlabel("PPE Class Name", fontsize=11)
    plt.grid(True, axis='y', linestyle="--", alpha=0.3)
    
    # Set y-axis limit to leave room for text
    plt.ylim(0, max(counts.values) * 1.12)
    
    # Hide top and right spines for a clean modern look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    os.makedirs("images", exist_ok=True)
    out_path = "images/class_distribution.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    print(f"Successfully generated and saved class distribution plot to {out_path}")

if __name__ == "__main__":
    main()
