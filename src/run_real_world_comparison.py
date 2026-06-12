import os
import cv2
import joblib
import numpy as np
import torch
import pandas as pd
from get_pose import PPEPoseCropper

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# maps numeric label back to human-readable class name
label_to_name = {
    0: 'Gloves',
    1: 'Hardhat',
    2: 'NO-Gloves',
    3: 'NO-Hardhat',
    4: 'NO-Safety Vest',
    5: 'Safety Vest'
}

cropper = PPEPoseCropper(encoder_name="resnet18")
if hasattr(cropper, "model"):
    cropper.model = cropper.model.to(device)
    cropper.model.eval()

# Load models
hardhat_model = joblib.load("saved_models/Hardhat_vs_NO-Hardhat.pkl")
glove_model_base = joblib.load("saved_models/Gloves_vs_NO-Gloves.pkl")
glove_model_tuned = joblib.load("saved_models/Gloves_vs_NO-Gloves_tuned.pkl")

test_folder = "testimage"
supported_ext = [".jpg", ".jpeg", ".png", ".bmp"]
image_files = [
    f for f in os.listdir(test_folder)
    if os.path.splitext(f)[1].lower() in supported_ext
]
image_files.sort()

results = []

for image_name in image_files:
    image_path = os.path.join(test_folder, image_name)
    img = cv2.imread(image_path)
    if img is None:
        continue

    try:
        with torch.no_grad():
            features = cropper.extract_region_features(img)
    except Exception:
        continue

    if features is None:
        continue

    # 1. Helmet predictions (base t=0.5 vs tuned t=0.3)
    helmet_base = "UNKNOWN"
    helmet_tuned = "UNKNOWN"
    try:
        head_feat = features["head"]
        if head_feat is not None:
            head_feat = head_feat.reshape(1, -1)
            probs = hardhat_model.predict_proba(head_feat)[0]
            classes = list(hardhat_model.classes_)
            idx_no_hardhat = classes.index(3)
            
            # Base (argmax / t=0.5)
            pred_base = int(hardhat_model.predict(head_feat)[0])
            helmet_base = label_to_name[pred_base]
            
            # Tuned (t=0.3 for NO-Hardhat)
            if probs[idx_no_hardhat] >= 0.3:
                helmet_tuned = "NO-Hardhat"
            else:
                helmet_tuned = "Hardhat"
        else:
            helmet_base = "HEAD_NOT_VISIBLE"
            helmet_tuned = "HEAD_NOT_VISIBLE"
    except Exception:
        pass

    # 2. Glove predictions (base vs tuned max_features=64)
    gloves_base = "UNKNOWN"
    gloves_tuned = "UNKNOWN"
    try:
        left_hand_feat  = features["left_hand"]
        right_hand_feat = features["right_hand"]

        if left_hand_feat is None and right_hand_feat is None:
            gloves_base = "HAND_NOT_VISIBLE"
            gloves_tuned = "HAND_NOT_VISIBLE"
        else:
            # Base model prediction
            hand_results_base = []
            if left_hand_feat is not None:
                pred_l = int(glove_model_base.predict(left_hand_feat.reshape(1, -1))[0])
                hand_results_base.append(label_to_name[pred_l])
            if right_hand_feat is not None:
                pred_r = int(glove_model_base.predict(right_hand_feat.reshape(1, -1))[0])
                hand_results_base.append(label_to_name[pred_r])
            
            if all(r == "Gloves" for r in hand_results_base):
                gloves_base = "Gloves"
            else:
                gloves_base = "NO-Gloves"
                
            # Tuned model prediction
            hand_results_tuned = []
            if left_hand_feat is not None:
                pred_l = int(glove_model_tuned.predict(left_hand_feat.reshape(1, -1))[0])
                hand_results_tuned.append(label_to_name[pred_l])
            if right_hand_feat is not None:
                pred_r = int(glove_model_tuned.predict(right_hand_feat.reshape(1, -1))[0])
                hand_results_tuned.append(label_to_name[pred_r])
            
            if all(r == "Gloves" for r in hand_results_tuned):
                gloves_tuned = "Gloves"
            else:
                gloves_tuned = "NO-Gloves"
    except Exception:
        pass

    results.append({
        "image": image_name,
        "helmet_base": helmet_base,
        "helmet_tuned": helmet_tuned,
        "gloves_base": gloves_base,
        "gloves_tuned": gloves_tuned
    })

res_df = pd.DataFrame(results)
res_df.to_csv("results/real_world_comparison.csv", index=False)

# Analyze differences
print(f"Processed {len(res_df)} images.")

# Helmet differences
helmet_diff = res_df[res_df["helmet_base"] != res_df["helmet_tuned"]]
print(f"\nHelmet changes (t=0.5 -> t=0.3): {len(helmet_diff)}")
for _, row in helmet_diff.iterrows():
    print(f"  {row['image']}: {row['helmet_base']} -> {row['helmet_tuned']}")

# Gloves differences
gloves_diff = res_df[res_df["gloves_base"] != res_df["gloves_tuned"]]
print(f"\nGlove changes (max_features=sqrt -> max_features=64): {len(gloves_diff)}")
for _, row in gloves_diff.iterrows():
    print(f"  {row['image']}: {row['gloves_base']} -> {row['gloves_tuned']}")
