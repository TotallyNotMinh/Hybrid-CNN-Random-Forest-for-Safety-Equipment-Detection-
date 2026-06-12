import os
import shutil
import pandas as pd

def main():
    list_file = "removed_outliers.txt"
    dest_base = "removed_outliers_images"

    if not os.path.exists(list_file):
        print(f"Error: {list_file} not found. Please run the finder script first.")
        return

    print(f"Reading outlier list from {list_file}...")
    df = pd.read_csv(list_file)

    copied_count = 0
    missing_count = 0

    print("Copying files...")
    for index, row in df.iterrows():
        src_path = row["image_path"]
        class_name = row["class_name"]

        # Normalize path separators for Windows/Unix compatibility
        src_path_normalized = src_path.replace("\\", os.sep).replace("/", os.sep)
        
        if not os.path.exists(src_path_normalized):
            missing_count += 1
            continue

        # Define destination path maintaining class structure
        dest_dir = os.path.join(dest_base, class_name)
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, os.path.basename(src_path_normalized))
        shutil.copy2(src_path_normalized, dest_path)
        copied_count += 1

    print(f"\nDone! Successfully copied {copied_count} images to '{dest_base}/'.")
    if missing_count > 0:
        print(f"Warning: {missing_count} images listed in CSV were not found on disk.")

if __name__ == "__main__":
    main()
