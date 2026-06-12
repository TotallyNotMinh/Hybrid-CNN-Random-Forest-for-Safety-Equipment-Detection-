import os
import shutil
from pathlib import Path
from tqdm import tqdm

# source is the raw downloaded dataset, output is where we put the cleaned version
SOURCE_DATASET = "data"
OUTPUT_DATASET = "ppe_filtered"

# all 14 original classes in the dataset
original_classes = [
    'Fall-Detected',
    'Gloves',
    'Goggles',
    'Hardhat',
    'Ladder',
    'Mask',
    'NO-Gloves',
    'NO-Goggles',
    'NO-Hardhat',
    'NO-Mask',
    'NO-Safety Vest',
    'Person',
    'Safety Cone',
    'Safety Vest'
]

# we only care about these 6 for our PPE compliance task
selected_classes = [
    'Gloves',
    'Hardhat',
    'NO-Gloves',
    'NO-Hardhat',
    'NO-Safety Vest',
    'Safety Vest'
]

# build a mapping from old class id -> new sequential id (0-5)
old_to_new = {
    original_classes.index(name): i
    for i, name in enumerate(selected_classes)
}

print("Class mapping:")
for old_id, new_id in old_to_new.items():
    print(f"{old_id} ({original_classes[old_id]}) -> {new_id}")

# go through train/valid/test and copy over only the images+labels we need
for split in ['train', 'valid', 'test']:

    print(f"\nProcessing {split}...")

    src_images = os.path.join(SOURCE_DATASET, split, "images")
    src_labels = os.path.join(SOURCE_DATASET, split, "labels")

    dst_images = os.path.join(OUTPUT_DATASET, split, "images")
    dst_labels = os.path.join(OUTPUT_DATASET, split, "labels")

    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    label_files = list(Path(src_labels).glob("*.txt"))

    for label_path in tqdm(label_files):

        image_name = label_path.stem

        # try jpg first, then jpeg, then png
        found_image = False

        for ext in [".jpg", ".jpeg", ".png"]:

            src_img = os.path.join(src_images, image_name + ext)

            if os.path.exists(src_img):

                dst_img = os.path.join(dst_images, image_name + ext)

                shutil.copy2(src_img, dst_img)

                found_image = True
                break

        if not found_image:
            continue

        # rewrite the label file keeping only lines for our 6 classes,
        # and remap the class ids to the new 0-5 range
        filtered_lines = []

        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls_id = int(parts[0])

            if cls_id in old_to_new:

                new_cls_id = old_to_new[cls_id]

                parts[0] = str(new_cls_id)

                filtered_lines.append(" ".join(parts))

        dst_label_path = os.path.join(
            dst_labels,
            label_path.name
        )

        with open(dst_label_path, "w") as f:
            f.write("\n".join(filtered_lines))

print("\nFiltered dataset created successfully.")

# write a yaml config so YOLO knows where to find everything
yaml_text = f"""
train: {OUTPUT_DATASET}/train/images
val: {OUTPUT_DATASET}/valid/images
test: {OUTPUT_DATASET}/test/images

nc: {len(selected_classes)}

names:
"""

for cls in selected_classes:
    yaml_text += f"  - {cls}\n"

yaml_path = "ppe_filtered.yaml"

with open(yaml_path, "w") as f:
    f.write(yaml_text)

print(f"\nYAML file saved to: {yaml_path}")
print(f"Filtered dataset saved to: {OUTPUT_DATASET}")
