# PPE Detection — Worker Safety Compliance Classifier

A machine learning pipeline that detects whether workers in images are wearing required Personal Protective Equipment (PPE): **hardhat**, **safety vest**, and **gloves**. The pipeline trains binary Random Forest classifiers on ResNet18 embeddings extracted from a YOLO-labelled dataset, then runs inference on new images using MediaPipe pose estimation to locate and crop the relevant body regions before classifying each one.

Each output image gets two layers of annotation drawn on it: the full **MediaPipe pose skeleton** (33 landmarks + connections) and a **colour-coded status panel** showing the helmet / vest / gloves result.

---

## How It Works

The project is split into two pipelines that share the same trained models.

### Training Pipeline

```
data/                        raw YOLO dataset (14 classes)
    ↓  filter_labels.py
ppe_filtered/                only the 6 PPE classes we care about
    ↓  crop_images.py
cropped_objects/             128×128 crops of each object, sorted by class
    ↓  encode.py
resnet_features.csv          512-dim ResNet18 embeddings for every crop
    ↓  remove_outliers.py
resnet_embeddings_filtered.csv   same, with bad crops removed
    ↓  train.py
saved_models/                3 trained Random Forest .pkl files
```

### Inference Pipeline

```
testimage/                   images you want to run through the system
    ↓  test.py  (uses get_pose.py internally)
        MediaPipe detects body pose
        → pose skeleton (33 landmarks + connections) drawn on image
        → head / torso / hand regions cropped
        → ResNet18 encodes each crop → 512-dim vector
        → Random Forest classifies each region
        → colour-coded status panel drawn on top
annotated_images/            annotated copies, originals untouched
```

---

## System Architecture & Methodology

This project combines deep learning feature extraction with classical machine learning to create a fast, robust classifier.

### 1. Pose-Driven Cropping (MediaPipe)
Instead of training an object detector to find small items like gloves or helmets directly, the pipeline uses **human pose estimation**. By first locating the 33 body landmarks, it can deterministically crop the exact regions where PPE should be. This provides a strong structural prior and makes the system highly robust to scale changes and partial occlusions.

The core extraction logic (`get_pose.py`) dynamically scales the bounding boxes based on the **shoulder width** (Euclidean distance between landmark 11 and 12) to ensure crops remain proportionally correct regardless of how close the person is to the camera:

*   **Head (Hardhat)**: Finds the bounding box encompassing the nose (0), eyes (2, 5), and ears (7, 8). Crucially, it extends the top of this box upward by **0.9× shoulder width** to ensure it captures a hardhat sitting high on the head, and widens the sides by 0.35× shoulder width.
*   **Torso (Safety Vest)**: Creates a bounding box containing the shoulders (11, 12) and hips (23, 24), padded by 0.3× shoulder width in all directions to capture bulky safety vests.
*   **Hands (Gloves)**: Creates tight bounding boxes around the wrist (15/16), pinky knuckle (17/18), index knuckle (19/20), and thumb (21/22), padded by 0.2× shoulder width.

### 2. Feature Extraction (ResNet18)
Rather than training a Convolutional Neural Network from scratch, we use a pre-trained **ResNet18** model with its final classification head removed. Passing the 128×128 body region crops through this network yields a dense, 512-dimensional semantic representation (embedding) of the image.

### 3. Outlier Removal (Centroid-Distance)
Raw bounding boxes from datasets often contain noisy crops (e.g., a "helmet" crop that is actually empty background). `remove_outliers.py` cleans the dataset by grouping embeddings by class, calculating the multi-dimensional centroid for each class, and removing any samples that fall further than `mean + 3×std` from the centroid.

### 4. Binary Classification (Random Forests & Decision Trees)
Instead of one large multi-class model, the pipeline trains **three separate binary Random Forest classifiers** (one each for helmet, vest, and gloves).

A Random Forest is an ensemble of many individual **Decision Trees**. In this architecture, instead of the decision trees looking at raw image pixels, they look at the **512-dimensional feature vector** produced by ResNet18. During training, the decision trees automatically find the optimal thresholds across these 512 abstract dimensions that separate the presence of PPE from its absence.

Because the trees operate on rich semantic features rather than pixels, they train in seconds, strongly resist overfitting, and allow each model to focus purely on the binary decision boundary of its specific PPE item (e.g. `Hardhat` vs `NO-Hardhat`).

---

## Annotation

Each output image has two layers drawn on it, in this order:

### 1. Pose skeleton

The full MediaPipe pose skeleton is drawn first — 33 landmarks as **green dots** (with a white outline) connected by **white lines**. This makes it easy to see which body regions were detected and used for classification.

### 2. Status panel

A semi-transparent panel is drawn on top in the top-left corner showing the PPE result for all three items:

| Colour | Meaning |
|--------|---------|
| 🟢 Green | PPE item detected |
| 🔴 Red | PPE item missing |
| ⚫ Grey | Could not determine (no pose, region not visible, or error) |

Example of what a fully annotated image looks like:

```
┌──────────────────────────┐
│ Helmet  OK               │  ← green
│ Vest    MISSING          │  ← red
│ Gloves  not visible      │  ← grey
└──────────────────────────┘
        (pose skeleton drawn underneath across the whole image)
```

The panel is drawn after the skeleton so it always stays readable regardless of where the landmarks fall.

---

## File Reference

| File | What it does |
|------|-------------|
| `filter_labels.py` | Filters the raw 14-class YOLO dataset down to 6 PPE classes and remaps the class IDs. Also writes `ppe_filtered.yaml`. |
| `crop_images.py` | Reads the filtered dataset, converts YOLO bounding boxes to pixel coordinates, crops each object and resizes to 128×128, saves to `cropped_objects/`. |
| `encode.py` | Loads each cropped image through a headless ResNet18 (final FC layer removed) to get a 512-dimensional feature vector, saves everything to `resnet_features.csv`. |
| `remove_outliers.py` | For each class, computes the centroid of its feature vectors and removes samples further than `mean + 3×std` from it. Saves clean data to `resnet_embeddings_filtered.csv`. |
| `train.py` | Trains three binary Random Forest classifiers (Hardhat vs NO-Hardhat, Safety Vest vs NO-Safety Vest, Gloves vs NO-Gloves) on the filtered embeddings. Saves each model to `saved_models/`. |
| `get_pose.py` | Defines `PPEPoseCropper` — runs MediaPipe pose detection, crops four body regions (head, torso, left hand, right hand), encodes each with ResNet18, and stores the raw detection result in `self.last_result` for downstream landmark drawing. |
| `test.py` | Runs inference on all images in `testimage/`. For each image: detect pose → draw skeleton → encode regions → predict PPE status → draw status panel → save to `annotated_images/`. |

---

## Glove Detection Logic

Gloves are the only PPE item where two regions (left hand and right hand) are checked independently. Each visible hand is classified by its own `predict()` call — there is no averaging or blending of features.

The result is determined by the strictest rule: **all visible hands must be wearing gloves** for the check to pass.

| Hands detected | Left result | Right result | Final result |
|----------------|-------------|--------------|-------------|
| Both | Gloves | Gloves | ✅ `Gloves` |
| Both | Gloves | NO-Gloves | ❌ `NO-Gloves` |
| Both | NO-Gloves | NO-Gloves | ❌ `NO-Gloves` |
| Left only | Gloves | — | ✅ `Gloves` |
| Left only | NO-Gloves | — | ❌ `NO-Gloves` |
| Neither | — | — | ⚫ `HAND_NOT_VISIBLE` |

Helmet and vest use a single region each (head and torso respectively), so those are always a single prediction.

---

## Classes

The original dataset has 14 classes. We keep 6:

| New ID | Class Name |
|--------|-----------|
| 0 | Gloves |
| 1 | Hardhat |
| 2 | NO-Gloves |
| 3 | NO-Hardhat |
| 4 | NO-Safety Vest |
| 5 | Safety Vest |

---

## Setup

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd Project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# or
source .venv/bin/activate   # Linux / macOS
```

### 3. Install PyTorch with CUDA support

Install PyTorch first, separately, using the official selector at https://pytorch.org/get-started/locally/.

The version used during development was **PyTorch 2.12.0 with CUDA 12.6**:

```bash
pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

If you don't have a CUDA GPU, install the CPU build instead:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the MediaPipe pose model

The first time you run `test.py` (or anything that uses `get_pose.py`), it will automatically download `pose_landmarker.task` (~9 MB) from Google's servers and save it to the project directory. No manual step needed.

---

## Running the Training Pipeline

Run each script in order:

```bash
# 1. filter the dataset
python src/filter_labels.py

# 2. crop objects from images
python src/crop_images.py

# 3. extract ResNet18 embeddings
python src/encode.py

# 4. remove outlier samples
python src/remove_outliers.py

# 5. train the classifiers
python src/train.py
```

> **Note:** `encode.py` can take a while on large datasets — GPU is strongly recommended.

---

## Dataset Structure

The training pipeline expects the raw dataset to be placed in a `data/` folder with the standard YOLO split layout:

```
data/
├── train/
│   ├── images/
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── labels/
│       ├── img_001.txt
│       ├── img_002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Images** can be `.jpg`, `.jpeg`, or `.png`. Each image must have a matching `.txt` label file with the same stem name in the corresponding `labels/` folder.

**Label files** follow the YOLO format — one object per line:

```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalised to `[0, 1]` relative to the image dimensions. Example:

```
3 0.512 0.374 0.185 0.231
1 0.801 0.210 0.094 0.178
```

The dataset originally contains 14 classes. `filter_labels.py` will discard the ones we don't need and remap the remaining 6 to new sequential IDs automatically — you do not need to pre-process the labels yourself.

**Original class IDs** (as they appear in the raw label files):

| Original ID | Class Name |
|-------------|-----------|
| 0 | Fall-Detected |
| 1 | Gloves |
| 2 | Goggles |
| 3 | Hardhat |
| 4 | Ladder |
| 5 | Mask |
| 6 | NO-Gloves |
| 7 | NO-Goggles |
| 8 | NO-Hardhat |
| 9 | NO-Mask |
| 10 | NO-Safety Vest |
| 11 | Person |
| 12 | Safety Cone |
| 13 | Safety Vest |

---

## Running Inference

Put your test images inside the `testimage/` folder, then run:

```bash
python src/test.py
```

The annotated copies will be saved to `annotated_images/`. The originals in `testimage/` are not modified.

---

## Model Performance (on validation split)

| Task | Accuracy |
|------|----------|
| Gloves vs NO-Gloves | 98.5% |
| Hardhat vs NO-Hardhat | 96.6% |
| Safety Vest vs NO-Safety Vest | 96.8% |

---

## Project Structure

```text
Project/
├── data/                        # raw YOLO dataset (not included in repo)
├── ppe_filtered/                # filtered 6-class dataset
├── cropped_objects/             # 128×128 per-class crops
├── saved_models/                # trained .pkl classifiers
├── testimage/                   # input images (read-only, never modified)
├── annotated_images/            # annotated output images
├── resnet_features.csv          # raw embeddings
├── resnet_embeddings_filtered.csv  # cleaned embeddings
├── ppe_filtered.yaml            # YOLO dataset config
├── pose_landmarker.task         # MediaPipe model weights
├── src/                         # All python scripts
│   ├── filter_labels.py
│   ├── crop_images.py
│   ├── encode.py
│   ├── remove_outliers.py
│   ├── train.py
│   ├── get_pose.py
│   ├── test.py
│   └── ... (analysis scripts)
├── requirements.txt
└── README.md
```
