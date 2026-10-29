import os

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Load preprocessed data
df = pd.read_csv("preprocessed_data.csv")

# Select features for clustering
features = [
    "PM2.5",
    "PM10",
    "NOx",
    "NO2",
    "SO2",
    "VOCs",
    "CO",
    "CO2",
    "CH4",
    "Temperature",
]

# Validate required columns
missing_features = [
    column for column in features
    if column not in df.columns
]

if missing_features:
    raise ValueError(
        f"Kolom berikut tidak ditemukan: {missing_features}"
    )

# Ensure selected features are numeric
X = df[features].apply(pd.to_numeric, errors="coerce")

# Replace infinite values and remove invalid rows
X = X.replace([float("inf"), float("-inf")], float("nan"))

valid_indices = X.dropna().index
X = X.loc[valid_indices].copy()
df = df.loc[valid_indices].copy()

# Validate dataset size
if len(X) < 6:
    raise ValueError(
        "Dataset minimal harus memiliki 6 baris valid."
    )

# Create a pipeline so the scaler is included in the saved model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    (
        "kmeans",
        KMeans(
            n_clusters=5,
            random_state=42,
            n_init=10,
        ),
    ),
])

# Enable MLflow autologging
mlflow.sklearn.autolog()

# MLflow Project manages the active run.
# Do not create another start_run() here.
pipeline.fit(X)

# Predict clusters using the complete pipeline
y_pred = pipeline.predict(X)

# Access the trained K-Means model
model = pipeline.named_steps["kmeans"]

# Evaluate clustering
inertia = float(model.inertia_)

if 1 < len(set(y_pred)) < len(y_pred):
    silhouette = float(
        silhouette_score(
            pipeline.named_steps["scaler"].transform(X),
            y_pred,
        )
    )
else:
    silhouette = 0.0

# Log additional metrics
mlflow.log_metric("inertia", inertia)
mlflow.log_metric("silhouette_score", silhouette)

# Log feature metadata
mlflow.log_dict(
    {"features": features},
    "feature_columns.json",
)

# Save clustering results
df["Cluster"] = y_pred
df.to_csv("hasil_clustering.csv", index=False)

print("Model training completed successfully.")
print(f"Number of valid rows: {len(X)}")
print(f"Inertia: {inertia:.4f}")
print(f"Silhouette Score: {silhouette:.4f}")

active_run = mlflow.active_run()

if active_run is not None:
    print(f"MLflow Run ID: {active_run.info.run_id}")

print("Experiment completed.")