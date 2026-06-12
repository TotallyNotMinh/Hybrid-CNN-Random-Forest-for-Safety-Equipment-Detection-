import pandas as pd
import numpy as np

# reads the raw embeddings, writes the cleaned version
INPUT_CSV  = "resnet_features.csv"
OUTPUT_CSV = "resnet_embeddings_filtered.csv"

df = pd.read_csv(INPUT_CSV)

# grab only numeric feature cols and drop the label column from that list
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if "label" in feature_cols:
    feature_cols.remove("label")

print("Number of features:", len(feature_cols))

filtered_groups = []

# filter each class separately so one class's spread doesn't affect another's threshold.
# we compute the centroid of the class, then keep only samples within
# mean + 3*std of the distance distribution — anything further is probably a bad crop.
for label, group in df.groupby("label"):

    X = group[feature_cols].to_numpy(dtype=np.float32)

    centroid = X.mean(axis=0)

    distances = np.linalg.norm(X - centroid, axis=1)

    threshold = distances.mean() + 3 * distances.std()

    filtered_groups.append(
        group[distances <= threshold]
    )

df_filtered = pd.concat(filtered_groups, ignore_index=True)

df_filtered.to_csv(
    OUTPUT_CSV,
    index=False
)

print("Original:", len(df))
print("Filtered:", len(df_filtered))
print("Removed:", len(df) - len(df_filtered))