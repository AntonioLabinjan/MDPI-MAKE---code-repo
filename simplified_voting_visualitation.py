!!apt-get update -qq
!apt-get install -y texlive-latex-base texlive-latex-extra texlive-fonts-recommended dvipng cm-super


import numpy as np
import matplotlib.pyplot as plt

# Use LaTeX fonts
plt.rcParams.update({
    "text.usetex": True,         # Enable LaTeX
    "font.family": "serif",      # Use serif fonts
    "font.serif": ["Computer Modern Roman"],  # Default LaTeX font
    "axes.labelsize": 15,        # Axis labels
    "axes.titlesize": 17,        # Title
    "legend.fontsize": 13,       # Legend
    "xtick.labelsize": 13,
    "ytick.labelsize": 13
})

# Generate random embeddings and target
embeddings = np.random.rand(1500, 2)
target = np.random.rand(1, 2)[0]

# Compute L2 distances
distances = np.linalg.norm(embeddings - target, axis=1)

# K1 and K2 neighbors
k1_indices = distances.argsort()[1:6]
k2_indices = distances.argsort()[6:15]

# Prepare figure
plt.figure(figsize=(10, 8))

# Scatter points
plt.scatter(embeddings[k1_indices, 0], embeddings[k1_indices, 1],
            color='green', label=r'$K_1$ neighbors (1--5)', s=50)
plt.scatter(embeddings[k2_indices, 0], embeddings[k2_indices, 1],
            color='blue', label=r'$K_2$ neighbors (6--15)', s=50)
plt.scatter(target[0], target[1], color='black',
            label=r'Recognized face (target)', s=70)

# Lines to neighbors
for idx in np.concatenate((k1_indices, k2_indices)):
    plt.plot([target[0], embeddings[idx, 0]],
             [target[1], embeddings[idx, 1]],
             color='gray', alpha=0.5)

# Dummy scatter for legend
plt.scatter([], [], color='gray', alpha=0.5, label=r'Voting connections')

# 📌 Auto zoom
all_x = np.concatenate(([target[0]], embeddings[k1_indices, 0], embeddings[k2_indices, 0]))
all_y = np.concatenate(([target[1]], embeddings[k1_indices, 1], embeddings[k2_indices, 1]))

margin = 0.1
x_min, x_max = all_x.min(), all_x.max()
y_min, y_max = all_y.min(), all_y.max()
x_range = x_max - x_min
y_range = y_max - y_min
plt.xlim(x_min - margin * x_range, x_max + margin * x_range)
plt.ylim(y_min - margin * y_range, y_max + margin * y_range)

# Labels and title (with LaTeX)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title(r"\textbf{Simplified Voting Visualization}", fontsize=16)

plt.legend(loc='upper right')
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Save as vector PDF
plt.savefig("voting_visualization_final.pdf", format="pdf")
plt.show()
