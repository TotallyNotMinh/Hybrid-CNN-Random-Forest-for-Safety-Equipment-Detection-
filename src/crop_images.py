import os
import cv2
from pathlib import Path
from tqdm import tqdm

# reads from the filtered dataset we built in filter_labels.py
DATASET_PATH = "ppe_filtered"
OUTPUT_PATH = "cropped_objects"

classes = [
    'Gloves',
    'Hardhat',
    'NO-Gloves',
    'NO-Hardhat',
    'NO-Safety Vest',
    'Safety Vest'
]

# make sure a folder exists for each class before we start writing into them
for cls_name in classes:
    os.makedirs(os.path.join(OUTPUT_PATH, cls_name), exist_ok=True)

img_count = 0

for split in ['train', 'valid', 'test']:

    print(f"\nProcessing {split}...")

    image_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    label_files = list(Path(label_dir).glob("*.txt"))

    for label_path in tqdm(label_files):

        image_name = label_path.stem

        # images can be jpg/jpeg/png so try each extension
        image_path = None

        for ext in [".jpg", ".jpeg", ".png"]:

            temp_path = os.path.join(image_dir, image_name + ext)

            if os.path.exists(temp_path):
                image_path = temp_path
                break

        if image_path is None:
            continue

        image = cv2.imread(image_path)

        if image is None:
            continue

        h, w = image.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        obj_idx = 0

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_w = float(parts[3])
            box_h = float(parts[4])

            # YOLO stores coords as fractions of image size, convert to pixels
            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # clamp to image bounds so we don't slice outside the array
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # skip boxes that collapsed to zero area
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # resize everything to 128x128 so the encoder sees a consistent size
            crop = cv2.resize(crop, (128, 128))

            class_name = classes[cls_id]

            save_name = f"{image_name}_{obj_idx}.jpg"

            save_path = os.path.join(
                OUTPUT_PATH,
                class_name,
                save_name
            )

            cv2.imwrite(save_path, crop)

            obj_idx += 1
            img_count += 1

print(f"\nDone. Saved {img_count} cropped objects.")