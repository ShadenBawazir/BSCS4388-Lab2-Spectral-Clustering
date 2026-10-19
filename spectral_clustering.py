"""
BSCS 4388 - Computer Vision
Lab 2: Image Clustering - Spectral Clustering
Dataset 1: Generated data (make_circles, 1000 samples, 2 circles)
Dataset 2: Real image (skimage.data.coins)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import SpectralClustering
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image as sk_image
from skimage import data
from skimage.transform import rescale
from skimage.filters import gaussian 

# Step 1 - Generate the dataset
X, y_true = make_circles(
    n_samples=1000, factor=0.5, noise=0.05, random_state=42
)

# Step 2 - Plot the dataset before applying Spectral Clustering
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c="gray", s=20)
plt.title("Generated Data - Before Spectral Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.tight_layout()
plt.savefig("spectral_data_before.png", dpi=150)
plt.show()

# Step 3 - Apply Spectral Clustering
spectral = SpectralClustering(
    n_clusters=2, affinity="nearest_neighbors", random_state=42
)
labels = spectral.fit_predict(X)

# Step 4 - Plot the dataset after applying Spectral Clustering
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=20)
plt.title("Generated Data - After Spectral Clustering (2 clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.axis("equal")
plt.tight_layout()
plt.savefig("spectral_data_after.png", dpi=150)
plt.show()

# Dataset 2: Real Image (coins)

# Step 1 - Load the image
coins = data.coins()

# Step 2 - Show the image before applying Spectral Clustering
plt.figure(figsize=(8, 6))
plt.imshow(coins, cmap="gray")
plt.title("Coins Image - Before Spectral Clustering")
plt.axis("off")
plt.tight_layout()
plt.savefig("spectral_image_before.png", dpi=150)
plt.show()

# Step 3 - Apply Spectral Clustering on neighbouring pixels
smooth_coins = gaussian(coins, sigma=2)
small_coins = rescale(smooth_coins, 0.2, anti_aliasing=False)

# Build a graph: each pixel is connected to its neighbouring pixels.
graph = sk_image.img_to_graph(small_coins)

# Convert brightness differences into similarities.
graph.data = np.exp(-10 * graph.data / graph.data.std())

img_labels = spectral_clustering(
    graph, n_clusters=25, assign_labels="discretize", random_state=42
)
segmented = img_labels.reshape(small_coins.shape)

# Step 4 - Display each region with its own colour
plt.figure(figsize=(8, 6))
plt.imshow(segmented, cmap="nipy_spectral", interpolation="nearest")
plt.title("Coins Image - After Spectral Clustering (25 regions)")
plt.axis("off")
plt.tight_layout()
plt.savefig("spectral_image_after.png", dpi=150)
plt.show()
