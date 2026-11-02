# Assignment 8: K-Means Clustering with Optimal Cluster Selection

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 1. Load and scale open-access dataset
iris = load_iris()
X = iris.data
X_scaled = StandardScaler().fit_transform(X)

# 2. Evaluate K-Means across cluster numbers (Elbow and Silhouette methods)
k_range = range(2, 11)
wcss_all = []
k_all = range(1, 11)
silhouette_scores = []

# WCSS for k=1 to 10
for k in k_all:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    wcss_all.append(kmeans.inertia_)
    if k >= 2:
        sil_score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil_score)

print("K-Means Cluster Optimization Analysis:")
for k, inertia in zip(k_all, wcss_all):
    if k == 1:
        print(f"K = {k:2d} | WCSS (Inertia): {inertia:8.2f} | Silhouette Score: N/A")
    else:
        print(f"K = {k:2d} | WCSS (Inertia): {inertia:8.2f} | Silhouette Score: {silhouette_scores[k - 2]:.4f}")

# 3. Determine optimal K based on maximum Silhouette Score
optimal_k = list(k_range)[silhouette_scores.index(max(silhouette_scores))]
print(f"\nOptimal K identified by Silhouette Method: {optimal_k}")

# 4. Train final model with optimal K
final_kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(X_scaled)

print("\nCluster Distribution (Data points per cluster):")
for cluster_id in range(optimal_k):
    count = (cluster_labels == cluster_id).sum()
    print(f"Cluster {cluster_id}: {count} samples")

# 5. Visualizations: Elbow Curve and Silhouette Scores
plt.figure(figsize=(11, 4.5))

plt.subplot(1, 2, 1)
plt.plot(list(k_all), wcss_all, marker='o', color='b')
plt.title("Elbow Method (WCSS vs K)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.xticks(list(k_all))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(list(k_range), silhouette_scores, marker='s', color='g')
plt.title("Silhouette Method (Score vs K)")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_range))
plt.grid(True)

plt.tight_layout()
plt.show()
