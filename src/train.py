import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# load the outlier-filtered embeddings
CSV_PATH = "resnet_embeddings_filtered.csv"

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

# pull out only the feature columns (f0..f511), leave label/class_name aside
feature_cols = [
    col for col in df.columns
    if col.startswith("f")
]

X = df[feature_cols].values
y = df["label"].values
class_names = df["class_name"].values

# 80/20 split, stratified so each class keeps its proportion
X_train, X_val, y_train, y_val, name_train, name_val = train_test_split(
    X,
    y,
    class_names,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Validation size:", len(X_val))

# build a bidirectional lookup between label id and class name
label_to_name = {}

for label, name in zip(df["label"], df["class_name"]):
    label_to_name[label] = name

# reverse mapping
name_to_label = {
    v: k for k, v in label_to_name.items()
}

# we train one binary classifier per PPE item rather than one big multiclass model,
# that way each classifier stays focused on a single yes/no question
tasks = [
    ("Hardhat", "NO-Hardhat"),
    ("Safety Vest", "NO-Safety Vest"),
    ("Gloves", "NO-Gloves")
]

results = []

for class_a_name, class_b_name in tasks:

    print("\n" + "=" * 70)

    print(f"Training: {class_a_name} vs {class_b_name}")

    class_a = name_to_label[class_a_name]
    class_b = name_to_label[class_b_name]

    # slice out just the two classes we care about for this task
    train_mask = np.isin(y_train, [class_a, class_b])

    X_train_pair = X_train[train_mask]
    y_train_pair = y_train[train_mask]

    val_mask = np.isin(y_val, [class_a, class_b])

    X_val_pair = X_val[val_mask]
    y_val_pair = y_val[val_mask]

    if class_a_name == "Gloves" or class_b_name == "Gloves":
        # Based on ablation studies, the Gloves task requires more features per split
        model = RandomForestClassifier(
            n_estimators=100,
            max_features=64,
            random_state=42,
            n_jobs=-1
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

    model.fit(X_train_pair, y_train_pair)

    # spaces in filenames are annoying, replace them
    model_name = f"{class_a_name}_vs_{class_b_name}.pkl".replace(" ", "_")

    save_path = os.path.join("saved_models", model_name)

    joblib.dump(model, save_path)

    print(f"Saved model to: {save_path}")

    preds = model.predict(X_val_pair)

    acc = accuracy_score(y_val_pair, preds)

    print(f"\nAccuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_val_pair,
            preds,
            labels=[class_a, class_b],
            target_names=[class_a_name, class_b_name]
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_val_pair, preds))

    results.append({
        "task": f"{class_a_name} vs {class_b_name}",
        "accuracy": acc
    })

# print a summary sorted by accuracy
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="accuracy",
    ascending=False
)

print("\n" + "=" * 70)
print("BINARY RANDOM FOREST RESULTS")
print("=" * 70)

print(results_df)