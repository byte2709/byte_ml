# Assignment 9: DBSCAN Density-Based Clustering and Noise Detection
# Purpose: Uniformly detects clusters and noise points without pre-specifying number of clusters

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Load and scale open-access dataset
iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

# 2. Apply DBSCAN Clustering
# eps: maximum distance for neighborhood (tuned to find optimal density)
# min_samples: minimum samples in cluster core (higher = fewer clusters)
eps_val = 0.6
min_samples_val = 5
dbscan = DBSCAN(eps=eps_val, min_samples=min_samples_val)
labels = dbscan.fit_predict(X_scaled)

# 3. Analyze clustering structure and noise points
unique_labels = set(labels)
n_clusters = len(unique_labels - {-1})
n_noise = np.sum(labels == -1)

print("DBSCAN Clustering Results:")
print(f"Parameters: eps = {eps_val}, min_samples = {min_samples_val}")
print(f"Estimated Number of Clusters: {n_clusters}")
print(f"Estimated Number of Noise Points: {n_noise}")

# Cluster distribution
print("\nCluster Distribution:")
for cluster_id in sorted(unique_labels):
    count = np.sum(labels == cluster_id)
    label_name = "Noise" if cluster_id == -1 else f"Cluster {cluster_id}"
    print(f"{label_name}: {count} samples")

# Silhouette score (computed on non-noise points)
non_noise_mask = (labels != -1)
if len(set(labels[non_noise_mask])) > 1:
    sil_score = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
    print(f"\nSilhouette Score (excluding noise): {sil_score:.4f}")

# 4. 2D PCA projection for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5.5))
colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_labels), 1)))

for cluster_id, color in zip(sorted(unique_labels), colors):
    mask = (labels == cluster_id)
    if cluster_id == -1:
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c='black', marker='x', s=60, label=f'Noise ({n_noise})')
    else:
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], color=color, marker='o', s=50, label=f'Cluster {cluster_id}')

plt.title(f"DBSCAN Clustering (eps={eps_val}, min_samples={min_samples_val})")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
