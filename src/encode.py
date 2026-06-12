import os
import cv2
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

import torchvision.transforms as transforms
from torchvision import models

# reads cropped objects, writes one big feature CSV
DATASET_PATH = "cropped_objects"
OUTPUT_CSV = "resnet_features.csv"

classes = [
    'Gloves',
    'Hardhat',
    'NO-Gloves',
    'NO-Hardhat',
    'NO-Safety Vest',
    'Safety Vest'
]

class_to_id = {
    name: idx
    for idx, name in enumerate(classes)
}
print(class_to_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# load resnet18 and drop the final FC layer so we get the raw 512-dim embedding
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model = torch.nn.Sequential(*list(model.children())[:-1])

model = model.to(device)
model.eval()

# standard imagenet normalisation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

all_features = []
all_labels = []
all_paths = []

for class_name in classes:

    class_dir = os.path.join(DATASET_PATH, class_name)

    image_files = list(Path(class_dir).glob("*"))

    print(f"\nProcessing {class_name} ({len(image_files)} images)")

    for image_path in tqdm(image_files):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        # cv2 loads BGR, resnet expects RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            features = model(image_tensor)

        # comes out as [1, 512, 1, 1], flatten to a 512-d vector
        features = features.squeeze().cpu().numpy()

        all_features.append(features)
        all_labels.append(class_to_id[class_name])
        all_paths.append(str(image_path))

# stack everything into a dataframe with named columns
feature_array = np.array(all_features)

feature_columns = [
    f"f{i}"
    for i in range(feature_array.shape[1])
]

df = pd.DataFrame(
    feature_array,
    columns=feature_columns
)

df["label"] = all_labels
df["class_name"] = [
    classes[label]
    for label in all_labels
]

df["image_path"] = all_paths

df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved feature CSV:")
print(OUTPUT_CSV)

print("\nDataset shape:")
print(df.shape)