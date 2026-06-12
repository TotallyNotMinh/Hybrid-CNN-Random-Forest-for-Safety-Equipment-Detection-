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

# Load optimized/baseline models
helmet_model = joblib.load("saved_models/Hardhat_vs_NO-Hardhat.pkl")
vest_model = joblib.load("saved_models/Safety_Vest_vs_NO-Safety_Vest.pkl")
glove_model_tuned = joblib.load("saved_models/Gloves_vs_NO-Gloves_tuned.pkl")

test_folder = "testimage"
output_folder = "annotated_images_tuned"
os.makedirs(output_folder, exist_ok=True)

supported_ext = [".jpg", ".jpeg", ".png", ".bmp"]
image_files = [
    f for f in os.listdir(test_folder)
    if os.path.splitext(f)[1].lower() in supported_ext
]
image_files.sort()

def get_label_color(value):
    present = {"Hardhat", "Safety Vest", "Gloves"}
    missing = {"NO-Hardhat", "NO-Safety Vest", "NO-Gloves"}
    if value in present:
        return (50, 210, 50)    # green (BGR)
    elif value in missing:
        return (50, 50, 220)    # red (BGR)
    else:
        return (160, 160, 160)  # grey

def shorten(value):
    mapping = {
        "Hardhat":           "Helmet  OK",
        "NO-Hardhat":        "Helmet  MISSING",
        "Safety Vest":       "Vest    OK",
        "NO-Safety Vest":    "Vest    MISSING",
        "Gloves":            "Gloves  OK",
        "NO-Gloves":         "Gloves  MISSING",
        "HEAD_NOT_VISIBLE":  "Helmet  not visible",
        "TORSO_NOT_VISIBLE": "Vest    not visible",
        "HAND_NOT_VISIBLE":  "Gloves  not visible",
        "NO_POSE":           "no pose detected",
        "FEATURE_ERROR":     "feature error",
        "LOAD_ERROR":        "load error",
        "ERROR":             "error",
        "UNKNOWN":           "unknown",
    }
    return mapping.get(value, value)

def draw_annotation(img, helmet, vest, gloves):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    padding    = 12
    line_gap   = 32

    rows = [
        (shorten(helmet), get_label_color(helmet)),
        (shorten(vest),   get_label_color(vest)),
        (shorten(gloves), get_label_color(gloves)),
    ]

    max_text_w = max(
        cv2.getTextSize(text, font, font_scale, thickness)[0][0]
        for text, _ in rows
    )

    panel_w = max_text_w + padding * 2
    panel_h = padding + line_gap * len(rows) + padding // 2

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, (text, color) in enumerate(rows):
        y = padding + (i + 1) * line_gap - 6
        cv2.putText(img, text, (padding, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return img

def draw_no_pose(img):
    label = "NO POSE DETECTED"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    padding = 10

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (tw + padding * 2, th + padding * 2), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, label, (padding, th + padding - 2), font, scale, (160, 160, 160), thick, cv2.LINE_AA)
    return img

POSE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7),
    (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (27, 31), (28, 32)
]

def draw_pose_landmarks(img, result):
    if result is None or not result.pose_landmarks:
        return
    h, w = img.shape[:2]
    for pose_landmarks in result.pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose_landmarks]
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
        for (px, py) in pts:
            cv2.circle(img, (px, py), 5, (0, 200, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

predictions = []

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
        draw_pose_landmarks(img, cropper.last_result)
        draw_no_pose(img)
        cv2.imwrite(os.path.join(output_folder, image_name), img)
        predictions.append({"image": image_name, "helmet": "NO_POSE", "vest": "NO_POSE", "gloves": "NO_POSE"})
        continue

    helmet_result = "UNKNOWN"
    vest_result   = "UNKNOWN"
    glove_result  = "UNKNOWN"

    # 1. Helmet — predict with threshold 0.3 for class 3 (NO-Hardhat)
    try:
        head_feat = features["head"]
        if head_feat is not None:
            head_feat = head_feat.reshape(1, -1)
            probs = helmet_model.predict_proba(head_feat)[0]
            classes = list(helmet_model.classes_)
            idx_no_hardhat = classes.index(3)
            # If probability of NO-Hardhat is >= 0.3, predict NO-Hardhat, else Hardhat
            if probs[idx_no_hardhat] >= 0.3:
                helmet_result = "NO-Hardhat"
            else:
                helmet_result = "Hardhat"
        else:
            helmet_result = "HEAD_NOT_VISIBLE"
    except Exception as e:
        helmet_result = "ERROR"

    # 2. Vest — predict baseline
    try:
        torso_feat = features["torso"]
        if torso_feat is not None:
            torso_feat = torso_feat.reshape(1, -1)
            vest_pred = int(vest_model.predict(torso_feat)[0])
            vest_result = label_to_name[vest_pred]
        else:
            vest_result = "TORSO_NOT_VISIBLE"
    except Exception:
        vest_result = "ERROR"

    # 3. Gloves — classify each hand with tuned model
    try:
        left_hand_feat  = features["left_hand"]
        right_hand_feat = features["right_hand"]

        if left_hand_feat is None and right_hand_feat is None:
            glove_result = "HAND_NOT_VISIBLE"
        else:
            hand_results = []
            if left_hand_feat is not None:
                left_pred = int(glove_model_tuned.predict(left_hand_feat.reshape(1, -1))[0])
                hand_results.append(label_to_name[left_pred])
            if right_hand_feat is not None:
                right_pred = int(glove_model_tuned.predict(right_hand_feat.reshape(1, -1))[0])
                hand_results.append(label_to_name[right_pred])

            if all(r == "Gloves" for r in hand_results):
                glove_result = "Gloves"
            else:
                glove_result = "NO-Gloves"
    except Exception:
        glove_result = "ERROR"

    # Draw and save
    draw_pose_landmarks(img, cropper.last_result)
    draw_annotation(img, helmet_result, vest_result, glove_result)
    cv2.imwrite(os.path.join(output_folder, image_name), img)
    predictions.append({"image": image_name, "helmet": helmet_result, "vest": vest_result, "gloves": glove_result})

# Save tuned predictions
pred_df = pd.DataFrame(predictions)
pred_df.to_csv("results/ppe_predictions_tuned.csv", index=False)
print("Saved predictions to results/ppe_predictions_tuned.csv")
print("Done processing real-world images with tuned models.")
