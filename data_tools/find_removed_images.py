import pandas as pd
import os

def main():
    features_csv = "resnet_features.csv"
    filtered_csv = "resnet_embeddings_filtered.csv"
    output_txt = "removed_outliers.txt"

    if not os.path.exists(features_csv) or not os.path.exists(filtered_csv):
        print("Error: CSV files not found. Please ensure you are in the correct directory.")
        return

    print("Loading image paths from CSVs...")
    # Using 'usecols' to load only the required columns for maximum efficiency
    df_raw = pd.read_csv(features_csv, usecols=["image_path", "class_name"])
    df_filtered = pd.read_csv(filtered_csv, usecols=["image_path"])

    # Convert to sets for O(1) lookups
    raw_paths = set(df_raw["image_path"])
    filtered_paths = set(df_filtered["image_path"])

    removed_paths = raw_paths - filtered_paths
    print(f"Total original images: {len(raw_paths)}")
    print(f"Total filtered images: {len(filtered_paths)}")
    print(f"Total removed images: {len(removed_paths)}")

    # Filter the original DataFrame to get the class names of the removed images
    df_removed = df_raw[df_raw["image_path"].isin(removed_paths)].copy()

    # Save the full list to a file
    df_removed.to_csv(output_txt, columns=["image_path", "class_name"], index=False)
    print(f"\nSaved the full list of removed images to: {output_txt}")

    # Display a summary table of removals per class
    print("\nSummary of removed images per class:")
    print("-" * 40)
    summary = df_removed["class_name"].value_counts()
    for cls, count in summary.items():
        print(f"{cls:<25} : {count} images")
    print("-" * 40)

if __name__ == "__main__":
    main()
