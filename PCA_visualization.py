import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from matplotlib import cm

# ✅ Scientific style fonts (works in Colab without real LaTeX)
plt.rcParams.update({
    "text.usetex": False,  # Use mathtext instead of external LaTeX
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# Load data
with open("known_face_encodings (5).pkl", "rb") as f:
    known_face_encodings = pickle.load(f)

with open("known_face_names (5).pkl", "rb") as f:
    known_face_names = pickle.load(f)

# Label encoding
le = LabelEncoder()
labels = le.fit_transform(known_face_names)
unique_labels = np.unique(labels)

# PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(known_face_encodings)

# Colors
colors = cm.get_cmap('tab20', len(unique_labels))

# Plot
plt.figure(figsize=(12, 8))
for i, label in enumerate(unique_labels):
    idx = labels == label
    plt.scatter(reduced_embeddings[idx, 0], reduced_embeddings[idx, 1],
                color=colors(i), alpha=0.7, edgecolors="k")

# Labels + title (LaTeX-style)
plt.title(r"PCA - Embeddings Visualization", fontsize=16)
plt.xlabel(r"Principal Component 1")
plt.ylabel(r"Principal Component 2")

# Ticks
plt.tick_params(axis="both", which="major", labelsize=10)

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save as high-quality vector PDF
plt.savefig("/content/pca_embeddings_fixed_2.pdf", format='pdf', dpi=1200)
plt.show()
