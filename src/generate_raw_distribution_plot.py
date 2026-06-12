import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    crops_dir = "cropped_objects"
    if not os.path.exists(crops_dir):
        print(f"Error: {crops_dir} directory not found.")
        return
        
    print(f"Reading class subfolders inside {crops_dir}...")
    
    # Get all class folders
    class_folders = [d for d in os.listdir(crops_dir) if os.path.isdir(os.path.join(crops_dir, d))]
    
    counts = {}
    total_files = 0
    
    for folder in class_folders:
        folder_path = os.path.join(crops_dir, folder)
        # Count only files (images) inside the folder
        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        counts[folder] = file_count
        total_files += file_count
        
    print("Class counts in cropped_objects folder:")
    for cls, count in counts.items():
        print(f"  {cls}: {count}")
    print(f"Total files: {total_files}")
    
    # Convert to pandas Series for sorting and plotting
    import pandas as pd
    counts_series = pd.Series(counts).sort_values(ascending=False)
    
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
    
    bar_colors = [colors.get(cls, "#9E9E9E") for cls in counts_series.index]
    
    bars = plt.bar(counts_series.index, counts_series.values, color=bar_colors, edgecolor="none", width=0.6)
    
    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + max(counts_series.values)*0.015,
            f"{height:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )
        
    plt.title("Raw Dataset Class Distribution (cropped_objects/ folder)", fontsize=13, fontweight="bold", pad=15)
    plt.ylabel("Number of Instances", fontsize=11)
    plt.xlabel("PPE Class Name", fontsize=11)
    plt.grid(True, axis='y', linestyle="--", alpha=0.3)
    
    # Set y-axis limit to leave room for text
    plt.ylim(0, max(counts_series.values) * 1.12)
    
    # Hide top and right spines for a clean modern look
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    os.makedirs("images", exist_ok=True)
    out_path = "images/raw_class_distribution.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    print(f"Successfully generated and saved raw class distribution plot to {out_path}")

if __name__ == "__main__":
    main()
