import matplotlib.pyplot as plt
import numpy as np

num_classes = np.arange(1, 51)
loading_times = [
    15, 15, 16, 16, 17, 17, 18, 18, 19, 19,
    20, 20, 21, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 30, 31, 32, 33, 39, 39, 40,
    41, 42, 43, 44, 45, 46, 48, 49, 50, 52,
    53, 54, 55, 56, 57, 59, 60, 61, 62, 63
]

# Create the plot
plt.figure(figsize=(8, 4))
plt.plot(num_classes, loading_times, marker='o', color='blue', label='Loading time')
plt.title("Correlation between the number of classes and loading time")
plt.xlabel("Number of classes")
plt.ylabel("Loading time (s)")
plt.legend()
plt.grid(True)

# Save to vector PDF
output_path = "/mnt/data/loading_time_vs_classes.pdf"
plt.savefig(output_path, format='pdf')

output_path
