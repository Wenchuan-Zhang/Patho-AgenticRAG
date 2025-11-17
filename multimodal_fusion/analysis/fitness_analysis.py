import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib.patches import Rectangle


def select_discriminative_instances(
    semantics_matrix: np.ndarray, n_clusters: int
) -> np.ndarray:
    """
    Cluster instances and select discriminative ones.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(semantics_matrix.T)
    selected = []
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        vars_ = np.var(semantics_matrix[:, idx], axis=0)
        max_var = np.max(vars_)
        candidates = idx[np.where(vars_ == max_var)[0]]
        if candidates.size > 1:
            avgs = np.mean(semantics_matrix[:, candidates], axis=0)
            sel = candidates[np.argmin(avgs)]
        else:
            sel = int(candidates)
        selected.append(int(sel))
    return np.array(sorted(selected))


# Load the semantic matrix
sem_mat = np.load("sem_mat.npy")

print(f"Semantic matrix shape: {sem_mat.shape}")
print(f"Matrix range: [{sem_mat.min():.3f}, {sem_mat.max():.3f}]")

# Set parameters
n_clusters_highlight = 5

# Perform clustering
kmeans = KMeans(n_clusters=n_clusters_highlight, random_state=0)
labels = kmeans.fit_predict(sem_mat.T)
selected_instances = select_discriminative_instances(sem_mat, n_clusters_highlight)

print(f"Selected instances: {selected_instances}")

# Create the two requested figures
fig, axes = plt.subplots(1, 2, figsize=(8, 3))

# Figure 1: Matrix with Selected Instances
ax1 = axes[0]
im1 = ax1.imshow(sem_mat, cmap="viridis", aspect="auto")

# Highlight selected instances with red vertical lines
for idx in selected_instances:
    ax1.axvline(x=idx, color="red", linewidth=2, alpha=0.8)

ax1.set_xlabel("Instances")
ax1.set_ylabel("Individuals")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Add legend for selected instances
red_patch = Rectangle((0, 0), 1, 1, facecolor="red", alpha=0.8)
ax1.legend([red_patch], ["Selected Instances"], loc="upper right")

# Figure 2: Clustering Results with Different Colors
ax2 = axes[1]

# Create cluster color map using seaborn palette
colors = sns.color_palette("viridis", n_clusters_highlight)
cluster_colors = np.zeros((sem_mat.shape[0], sem_mat.shape[1], 3))

# Assign colors based on cluster labels
for i in range(sem_mat.shape[1]):
    cluster_id = labels[i]
    cluster_colors[:, i, :] = colors[cluster_id]  # RGB values

im2 = ax2.imshow(cluster_colors, aspect="auto")

# Highlight selected instances
for idx in selected_instances:
    ax2.axvline(x=idx, color="red", linewidth=3, alpha=0.9)

ax2.set_xlabel("Instances")
ax2.set_ylabel("Individuals")

# Add legend for selected instances
red_patch = Rectangle((0, 0), 1, 1, facecolor="red", alpha=0.9)
ax2.legend([red_patch], ["Selected Instances"], loc="upper right")

plt.tight_layout()
plt.show()

# Print cluster information
print("\n=== Clustering Results ===")
print(f"Number of clusters: {n_clusters_highlight}")
for cid in range(n_clusters_highlight):
    cluster_instances = np.where(labels == cid)[0]
    selected_in_cluster = [i for i in selected_instances if i in cluster_instances]
    print(
        f"Cluster {cid}: {len(cluster_instances)} instances, Selected: {selected_in_cluster}"
    )


