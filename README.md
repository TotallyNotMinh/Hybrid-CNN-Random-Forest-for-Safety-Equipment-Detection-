# A Pose-Directed Semantic Embedding Pipeline for PPE Compliance Auditing

This repository contains the code for our final undergraduate project in Classical Machine Learning. It implements a hybrid computer vision pipeline to automate **Personal Protective Equipment (PPE) compliance auditing** (Hardhats, Safety Vests, and Gloves) in industrial environments.

Rather than relying on computationally heavy, end-to-end deep object detectors (which struggle with small scale, deformability, and class imbalance), we explicitly decoupled the problem into **Pose Estimation**, **Deep Feature Extraction**, and **Classical Machine Learning**.

---

## 🌟 Key Achievements

* **Hybrid Architecture:** Built a fast, interpretable pipeline using MediaPipe Pose, a frozen headless ResNet18, and three independent binary Random Forest classifiers.
* **High Accuracy:** Achieved up to **98.5% validation accuracy** and **97.4% real-world deployment accuracy** across 159k+ high-dimensional bounding box crops.
* **Safety-First Imbalance Mitigation:** Handled a severe **17.5:1 class imbalance**. By calibrating the decision threshold to $t=0.3$ for hardhats, we successfully slashed safety-critical false negatives (undetected bare heads) by **67%**.
* **Feature Optimization:** Conducted detailed `max_features` ablation studies, increasing dimensions to 64 for the Gloves task to prevent feature dilution and probability mass leakage.
* **Scalability Proof:** Demonstrated that Random Forests offer superior $O(n)$ scalability over $O(n^2)$ SVM (RBF) baselines when operating on 126k+ 512-dimensional samples.

---

## 🧠 System Architecture

The pipeline is split into three decoupled stages:

### 1. Scale-Invariant Cropping (MediaPipe)
To solve the problem of workers appearing at varying distances from the camera, we use **MediaPipe Pose** to track 33 joint landmarks. The pipeline dynamically scales bounding boxes for the head, torso, and hands proportionally based on the worker's **shoulder width**. This explicitly enforces spatial consistency and uses landmark visibility scores to suppress predictions on occluded body parts.

### 2. High-Dimensional Semantic Encoding (ResNet18)
Instead of training a CNN from scratch, we apply Transfer Learning. The cropped body regions are resized to 224x224 and fed into a **frozen, headless ResNet18**. This acts purely as a fixed feature extractor, projecting raw pixels into a highly discriminative **512-dimensional continuous semantic space**.

### 3. Classification Core (Random Forests)
The core decision-making algorithm is a cascade of **three binary Random Forest classifiers** ($B=100$ trees each). By training separate binary models (e.g., `Hardhat` vs. `NO-Hardhat`), we eliminate the cross-task confusion and probability leakage inherent in standard multiclass models. The models operate directly on the 512D ResNet embeddings.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Project
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # or
   source .venv/bin/activate   # Linux / macOS
   ```

3. **Install PyTorch (CUDA recommended):**
   ```bash
   pip install torch==2.12.0+cu126 torchvision==0.27.0+cu126 --index-url https://download.pytorch.org/whl/cu126
   ```

4. **Install the remaining requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Pipeline

All core machine learning scripts have been organized into the `src/` directory.

### 1. Offline Training Pipeline
Place your raw YOLO dataset in a `data/` folder in the root directory. Then run the pipeline sequentially:

```bash
# 1. Filter YOLO dataset down to our 6 core classes
python src/filter_labels.py

# 2. Extract bounding box crops of PPE objects
python src/crop_images.py

# 3. Pass crops through ResNet18 to generate 512D feature embeddings
python src/encode.py

# 4. Clean noise by removing centroid outliers (mean + 3x std)
python src/remove_outliers.py

# 5. Train the 3 Binary Random Forest Classifiers
# Note: Implements max_features=64 for Gloves task based on ablation study
python src/train.py
```

### 2. Live Inference & Auditing
Place any real-world test images in the `testimage/` folder. 

```bash
python src/test.py
```
The script will output annotated images to the `annotated_images/` folder. It explicitly implements our safety-critical decision threshold ($t=0.3$ for `NO-Hardhat`) using `predict_proba()` to aggressively flag safety violations.

**Output Annotations:**
Each output image features the full MediaPipe skeletal overlay and a color-coded status panel:
* 🟢 **Green**: PPE is compliant
* 🔴 **Red**: PPE is missing (Violation)
* ⚫ **Grey**: Body part not visible / Occluded

---

## 📁 Repository Structure

```text
Project/
├── src/                         # Core Machine Learning Pipeline
│   ├── filter_labels.py         # Dataset filtering
│   ├── crop_images.py           # Bounding box extraction
│   ├── encode.py                # ResNet18 feature extraction
│   ├── remove_outliers.py       # Distance-based data cleaning
│   ├── train.py                 # Random Forest training
│   ├── get_pose.py              # MediaPipe cropping logic
│   ├── test.py                  # Live inference and annotation
│   └── ...                      # Analysis & ablation scripts
├── report_tools/                # Custom scripts used to format the LaTeX report
├── data_tools/                  # Custom one-off data cleaning scripts
├── testimage/                   # Put inference images here
├── annotated_images/            # Annotated inference outputs
├── .gitignore                   # Ignores heavy CSVs, datasets, and cache
├── requirements.txt             
└── README.md
```

*(Note: The heavy dataset folders (`data/`, `ppe_filtered/`, `cropped_objects/`), 512D embedding CSVs, and serialized `.pkl` models are intentionally excluded from version control via `.gitignore` to preserve repository hygiene).*
