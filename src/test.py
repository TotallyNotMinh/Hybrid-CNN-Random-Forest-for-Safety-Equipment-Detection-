from get_pose import PPEPoseCropper
import os
import cv2
import joblib
import numpy as np
import torch

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

# PPEPoseCropper handles both the mediapipe pose detection and the resnet encoding
cropper = PPEPoseCropper(
    encoder_name="resnet18"
)

# push the internal encoder to GPU if we have one
if hasattr(cropper, "model"):

    cropper.model = cropper.model.to(device)

    cropper.model.eval()

# load the three binary classifiers we trained
helmet_model = joblib.load(
    "saved_models/Hardhat_vs_NO-Hardhat.pkl"
)

vest_model = joblib.load(
    "saved_models/Safety_Vest_vs_NO-Safety_Vest.pkl"
)

glove_model = joblib.load(
    "saved_models/Gloves_vs_NO-Gloves.pkl"
)

test_folder   = "testimage"
output_folder = "annotated_images"

os.makedirs(output_folder, exist_ok=True)

supported_ext = [".jpg", ".jpeg", ".png", ".bmp"]

image_files = [
    f for f in os.listdir(test_folder)
    if os.path.splitext(f)[1].lower() in supported_ext
]

image_files.sort()

if len(image_files) == 0:
    raise ValueError("No images found in testimage folder.")


# =============================================================================

def get_label_color(value):
    """green if PPE is present, red if missing, grey for anything else"""
    present = {"Hardhat", "Safety Vest", "Gloves"}
    missing = {"NO-Hardhat", "NO-Safety Vest", "NO-Gloves"}
    if value in present:
        return (50, 210, 50)    # green (BGR)
    elif value in missing:
        return (50, 50, 220)    # red (BGR)
    else:
        return (160, 160, 160)  # grey for NO_POSE / errors / not visible


def shorten(value):
    """trim long class names so they fit cleanly on the panel"""
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
    """Draw a semi-transparent status panel in the top-left corner."""

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

    # figure out how wide the panel needs to be
    max_text_w = max(
        cv2.getTextSize(text, font, font_scale, thickness)[0][0]
        for text, _ in rows
    )

    panel_w = max_text_w + padding * 2
    panel_h = padding + line_gap * len(rows) + padding // 2

    # dark semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # write each line in its colour
    for i, (text, color) in enumerate(rows):
        y = padding + (i + 1) * line_gap - 6
        cv2.putText(img, text, (padding, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return img


def draw_no_pose(img):
    """Minimal label for images where no person was detected."""

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


# pairs of landmark indices that should be connected by a line
# matches the standard MediaPipe pose topology
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
    """Draw the pose skeleton directly with cv2 — dots for landmarks,
    lines for connections. Drawn before the panel so text stays on top."""

    if result is None or not result.pose_landmarks:
        return

    h, w = img.shape[:2]

    for pose_landmarks in result.pose_landmarks:

        # convert normalised coords to pixel positions
        pts = [
            (int(lm.x * w), int(lm.y * h))
            for lm in pose_landmarks
        ]

        # draw the connecting lines first so dots sit on top of them
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)

        # draw each landmark as a filled green dot with a white outline
        for (px, py) in pts:
            cv2.circle(img, (px, py), 5, (0, 200, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)


# =============================================================================

saved = 0
skipped = 0

for image_name in image_files:

    image_path = os.path.join(test_folder, image_name)

    print(f"Processing: {image_name}")

    img = cv2.imread(image_path)

    if img is None:
        print(f"  Could not load — skipping.")
        skipped += 1
        continue

    # run pose detection and encode each body region
    try:

        with torch.no_grad():

            features = cropper.extract_region_features(img)

    except Exception as e:

        print(f"  Feature extraction failed: {e}")
        skipped += 1
        continue

    # no person found — draw the skeleton anyway (will be empty) and label it
    if features is None:

        print(f"  No pose detected.")
        draw_pose_landmarks(img, cropper.last_result)
        draw_no_pose(img)
        cv2.imwrite(os.path.join(output_folder, image_name), img)
        saved += 1
        continue

    helmet_result = "UNKNOWN"
    vest_result   = "UNKNOWN"
    glove_result  = "UNKNOWN"

    # helmet — use the head crop features
    try:

        head_feat = features["head"]

        if head_feat is not None:

            head_feat    = head_feat.reshape(1, -1)
            
            # Predict probability instead of class directly
            probs = helmet_model.predict_proba(head_feat)[0]
            
            # Find the index for NO-Hardhat (label 3)
            no_hardhat_idx = np.where(helmet_model.classes_ == 3)[0][0]
            
            # Apply safety threshold t=0.3 to reduce false negatives
            if probs[no_hardhat_idx] >= 0.3:
                helmet_pred = 3  # NO-Hardhat
            else:
                helmet_pred = 1  # Hardhat
                
            helmet_result = label_to_name[helmet_pred]

        else:

            helmet_result = "HEAD_NOT_VISIBLE"

    except Exception as e:

        print(f"  Helmet prediction failed: {e}")
        helmet_result = "ERROR"

    # vest — use the torso crop features
    try:

        torso_feat = features["torso"]

        if torso_feat is not None:

            torso_feat  = torso_feat.reshape(1, -1)
            vest_pred   = int(vest_model.predict(torso_feat)[0])
            vest_result = label_to_name[vest_pred]

        else:

            vest_result = "TORSO_NOT_VISIBLE"

    except Exception as e:

        print(f"  Vest prediction failed: {e}")
        vest_result = "ERROR"

    # gloves — classify each hand independently.
    # we only call it OK if every visible hand is wearing a glove.
    # if even one hand comes back NO-Gloves, the worker fails the check.
    try:

        left_hand_feat  = features["left_hand"]
        right_hand_feat = features["right_hand"]

        if left_hand_feat is None and right_hand_feat is None:

            glove_result = "HAND_NOT_VISIBLE"

        else:

            hand_results = []

            if left_hand_feat is not None:
                left_pred = int(glove_model.predict(left_hand_feat.reshape(1, -1))[0])
                hand_results.append(label_to_name[left_pred])

            if right_hand_feat is not None:
                right_pred = int(glove_model.predict(right_hand_feat.reshape(1, -1))[0])
                hand_results.append(label_to_name[right_pred])

            # both hands must have gloves — if any hand fails, the whole check fails
            if all(r == "Gloves" for r in hand_results):
                glove_result = "Gloves"
            else:
                glove_result = "NO-Gloves"

    except Exception as e:

        print(f"  Glove prediction failed: {e}")
        glove_result = "ERROR"

    print(f"  Helmet: {helmet_result} | Vest: {vest_result} | Gloves: {glove_result}")

    # draw landmarks first, then the annotation panel on top so text stays readable
    draw_pose_landmarks(img, cropper.last_result)
    draw_annotation(img, helmet_result, vest_result, glove_result)
    cv2.imwrite(os.path.join(output_folder, image_name), img)
    saved += 1

print(f"\nDone. Annotated {saved} image(s), skipped {skipped}.")