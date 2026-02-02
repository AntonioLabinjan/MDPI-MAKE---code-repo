import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Hardcoded values
TP = 2702
FP = 991
TN = 46577
FN = 991

# Confusion matrix
cm = np.array([[TP, FP],
               [FN, TN]])

plt.figure(figsize=(3.2, 2.8), dpi=300, facecolor='white')
ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                 xticklabels=["TP", "FP"],
                 yticklabels=["FN", "TN"],
                 annot_kws={"size": 10},
                 linewidths=0.5, linecolor='black', square=True)

ax.tick_params(axis='both', which='major', labelsize=9)

plt.tight_layout()
plt.savefig("confusion_matrix_for_paper.pdf", bbox_inches='tight', facecolor='white')
plt.show()
