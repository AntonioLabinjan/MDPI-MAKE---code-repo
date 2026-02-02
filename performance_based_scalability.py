import matplotlib.pyplot as plt

# === DATA ===
num = list(range(1, 51))      # Number of classes (e.g. [1, 2, 3, ..., 50])
precision = [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.9981,0.9982,0.9982,0.9983,0.9985,0.9985,0.9986,0.9986, 0.9987, 0.9974, 0.9939, 0.9942, 0.9946, 0.9709, 0.9711, 0.9082, 0.8601, 0.8614, 0.8371, 0.7596, 0.7456, 0.7300, 0.7296, 0.7157, 0.7386, 0.7248, 0.7282, 0.7399, 0.7390, 0.7419, 0.7482, 0.7487, 0.7482, 0.7300]  # Preciznost za svaku klasu
recall = [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.9981, 0.9982,0.9982,0.9983,0.9985,0.9985,0.9986,0.9986, 0.9987, 0.9974, 0.9939, 0.9942, 0.9946, 0.9709, 0.9711, 0.9082, 0.8601, 0.8614, 0.8371, 0.7596, 0.7456, 0.7300, 0.7296, 0.7157, 0.7386, 0.7248, 0.7282, 0.7399, 0.7390, 0.7419, 0.7482, 0.7487, 0.7482, 0.7300]     # Opoziv za svaku klasu
f1 = [1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,0.9981, 0.9982,0.9982,0.9983,0.9985,0.9985,0.9986,0.9986, 0.9987, 0.9974, 0.9939, 0.9942, 0.9946, 0.9662, 0.9665, 0.8924, 0.8324, 0.8334, 0.8122, 0.7191, 0.7456, 0.6771, 0.6766, 0.6562, 0.6843,0.6676 , 0.6722, 0.6892, 0.6895, 0.6933, 0.7008, 0.6980, 0.6958, 0.6773 ]         # F1 rezultat za svaku klasu

plt.figure(figsize=(14, 6))

plt.plot(num, precision, marker='o', color='blue', label='Precision')
plt.plot(num, recall, marker='s', color='green', label='Recall')
plt.plot(num, f1, marker='p', color='magenta', label='F1-Score')

plt.title("Performance Metrics vs Number of Classes")
plt.xlabel("Number of Classes")
plt.ylabel("Metric Value")
plt.xticks(num, rotation=60)
plt.ylim(0.5, 1.05)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()

# === Saving into PDF ===
plt.savefig("numClass_scalability.pdf", format='pdf')

plt.show()

# === Plotting ===
fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

# Precision
axs[0].plot(num, precision, marker='o', color='blue')
axs[0].set_title("Precision vs Number of Classes")
axs[0].set_xlabel("Number of Classes")
axs[0].set_ylabel("Metric Value")
axs[0].grid(True, axis='y', linestyle='--', alpha=0.6)
axs[0].set_ylim(0.5, 1.05)

# Recall
axs[1].plot(num, recall, marker='s', color='green')
axs[1].set_title("Recall vs Number of Classes")
axs[1].set_xlabel("Number of Classes")
axs[1].grid(True, axis='y', linestyle='--', alpha=0.6)

# F1-Score
axs[2].plot(num, f1, marker='p', color='magenta')
axs[2].set_title("F1-Score vs Number of Classes")
axs[2].set_xlabel("Number of Classes")
axs[2].grid(True, axis='y', linestyle='--', alpha=0.6)

# Adjustment of plots
for ax in axs:
    ax.set_xticks(num[::2])
    ax.tick_params(axis='x', rotation=60)

plt.tight_layout()
plt.savefig("numClass_scalability_separated.pdf", format='pdf')
plt.show()
