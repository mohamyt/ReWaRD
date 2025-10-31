import random
from tqdm import tqdm
import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.special import softmax
import matplotlib.pyplot as plt
from umap import UMAP

# -----------------------------
# CONFIGURATION
# -----------------------------
num_clusters = 256   # Number of clusters for K-means
top_k = 5            # Number of closest clusters for pseudo-labels
subset_for_umap = 100000  # Max samples to use for UDP map visualization
seed = 1
latent_file = "latent_vectors.h5"
pseudo_label_file = "data/pseudo_labels_ldl.h5"
udp_map_file = "data/udp_map.npy"

np.random.seed(seed)
random.seed(seed)


# LOAD GENERATED LATENT VECTORS (from generate_latent_vectors.py)
print(f"Loading latent vectors from '{latent_file}'...")
with h5py.File(latent_file, "r") as f:
    latent_vectors = np.array(f["latent_vectors"], dtype=np.float32)

num_samples, dim = latent_vectors.shape
print(f"Loaded {num_samples:,} samples with {dim} dimensions")

# NORMALIZE LATENT VECTORS
print("Normalizing latent vectors...")
latent_vectors /= np.linalg.norm(latent_vectors, axis=1, keepdims=True)

#K-MEANS CLUSTERING
print("Applying K-means clustering...")
kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init="auto", verbose=0)
cluster_labels = kmeans.fit_predict(latent_vectors)
cluster_centers = kmeans.cluster_centers_


#COMPUTE SILHOUETTE SCORE
print("Computing Silhouette score (may take a minute)...")
sample_size = min(10000, num_samples)
sil_score = silhouette_score(latent_vectors[:sample_size], cluster_labels[:sample_size])
print(f" Silhouette Score: {sil_score:.4f}")


#ASSIGN PSEUDO-LABELS
print("Assigning soft pseudo-labels...")
pseudo_labels = np.zeros((num_samples, num_clusters), dtype=np.float32)
batch_size = 10000 

for i in tqdm(range(0, num_samples, batch_size), desc="Assigning pseudo-labels"):
    end = min(i + batch_size, num_samples)
    distances = np.linalg.norm(latent_vectors[i:end, None] - cluster_centers, axis=2)
    top_k_indices = np.argpartition(distances, top_k, axis=1)[:, :top_k]
    top_k_distances = np.take_along_axis(distances, top_k_indices, axis=1)
    top_k_weights = softmax(-top_k_distances, axis=1)
    for j, indices in enumerate(top_k_indices):
        pseudo_labels[i + j, indices] = top_k_weights[j]

with h5py.File(pseudo_label_file, 'w') as f:
    f.create_dataset('pseudo_labels', data=pseudo_labels)
print(f" Pseudo-labels saved to '{pseudo_label_file}'")


#GENERATE UDP MAP (UMAP)
print("Generating UDP map with UMAP...")

# Subsample
if num_samples > subset_for_umap:
    print(f"Dataset too large for full UMAP ({num_samples:,} samples). Using subset of {subset_for_umap:,}.")
    idx = np.random.choice(num_samples, subset_for_umap, replace=False)
    latent_subset = latent_vectors[idx]
    cluster_subset = cluster_labels[idx]
else:
    latent_subset = latent_vectors
    cluster_subset = cluster_labels

# UMAP reduction
reducer = UMAP(
    n_components=2,
    random_state=seed,
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean"
)
latent_2d = reducer.fit_transform(latent_subset)
np.save(udp_map_file, latent_2d)
print(f" UDP (UMAP) map saved to '{udp_map_file}'")


#PLOT UDP MAP

plt.figure(figsize=(10, 8))
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=cluster_subset, cmap="tab20", s=2, alpha=0.7)
plt.title(f"UDP Map (UMAP 2D Projection) - {num_clusters} clusters\nSilhouette={sil_score:.3f}")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("data/udp_map.png", dpi=300)
plt.show()

print(" UDP visualization saved to 'data/udp_map.png'")
