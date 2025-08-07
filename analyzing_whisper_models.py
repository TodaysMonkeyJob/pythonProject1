import matplotlib.pyplot as plt
import numpy as np

# Experiment Settings
experiments = np.arange(1, 25)  # X-axis: Number of experiments
models = ['Tiny', 'Whisper-1 (lang=None)', 'Whisper-1 (lang=en)', 'Large-2']
models2 = ['Tiny', 'Whisper-1 (lang=None)', 'Whisper-1 (lang=en)']


# Data for Visualization
accuracy = [
    [86.96, 90.47, 88.23, 84.61, 100.0, 90.90, 93.75, 90.91, 92.86, 100.0, 91.67, 100.0, 92.86, 90.91, 92.86, 91.67, 93.33, 92.31, 92.86, 100.0, 92.86, 91.67, 93.33, 92.31],
    [91.30, 95.24, 94.11, 92.31, 100.0, 100.0, 87.50, 100.0, 100.0, 100.0, 100.0, 100.0, 92.86, 100.0, 92.86, 100.0, 93.33, 92.31, 92.86, 92.86, 92.86, 100.0, 93.33, 92.86],
    [95.65, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 90.91, 85.71, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 91.67, 91.67, 100.0],
    [95.65, 100.0, 94.11, 100.0, 100.0, 100.0, 100.0, 90.91, 92.86, 90.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 92.86, 92.86, 100.0, 100.0, 100.0, 92.86],
]

processing_time = [
    [0.84, 0.74, 0.66, 0.75, 0.66, 0.81, 0.76, 0.82, 0.74, 0.89, 0.78, 0.69, 0.81, 0.76, 0.85, 0.72, 0.84, 0.89, 0.95, 0.91, 0.85, 0.72, 0.84, 0.89],
    [1.64, 1.19, 1.05, 1.87, 2.25, 1.92, 2.61, 2.33, 1.87, 1.94, 2.13, 1.76, 2.22, 1.85, 2.09, 1.89, 2.02, 1.94, 1.98, 2.01, 2.09, 1.89, 2.02, 1.94],
    [1.49, 1.47, 1.51, 1.98, 2.12, 1.95, 2.47, 2.42, 2.15, 2.21, 2.35, 2.09, 2.15, 1.89, 2.15, 1.95, 2.17, 2.08, 2.12, 2.14, 2.15, 1.95, 1.95, 2.17],
    [26.89, 24.14, 24.78, 23.67, 28.45, 21.98, 26.54, 25.78, 22.36, 24.83, 27.22, 23.11, 22.89, 23.47, 27.48, 22.78, 23.56, 25.67, 24.59, 25.32, 27.48, 22.78, 23.56, 25.67],

]

memory_usage = [
    [251.12, 257.52, 245.32, 233.18, 256.11, 200.33, 238.55, 245.32, 235.78, 243.22, 250.98, 247.28, 239.92, 234.74, 236.95, 240.12, 242.25, 237.87, 238.94, 240.74, 236.95, 240.12, 242.25, 237.87],
    [243.13, 251.93, 228.25, 210.99, 230.43, 223.12, 229.78, 240.12, 228.45, 225.77, 232.59, 225.84, 229.34, 230.81, 228.9, 235.64, 233.53, 239.0, 234.15, 233.23, 228.9, 235.64, 233.53, 239.0],
    [245.25, 227.18, 238.32, 220.47, 241.2, 231.74, 230.45, 241.87, 243.32, 230.5, 235.3, 244.2, 240.85, 240.53, 239.12, 242.31, 241.51, 240.37, 240.25, 242.4, 239.12, 242.31, 241.51, 240.37],
    [260.82, 259.28, 235.49, 243.45, 270.52, 230.84, 245.29, 260.08, 250.93, 240.87, 245.45, 245.93, 248.64, 245.97, 250.54, 247.85, 244.98, 246.22, 249.75, 250.84, 250.54, 247.85, 244.98, 246.22],
]

# Accuracy Comparison Plot
plt.figure(figsize=(14, 7))
for i, label in enumerate(models):
    plt.plot(experiments, accuracy[i], marker='o', linestyle='-', label=label)
plt.title("Accuracy Score Comparison Across Models", fontsize=16)
plt.xlabel("Experiment #", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Processing Time Comparison Plot
plt.figure(figsize=(14, 7))
for i, label in enumerate(models):
    plt.plot(experiments, processing_time[i], marker='o', linestyle='-', label=label)
plt.title("Processing Time Comparison Across Models", fontsize=16)
plt.xlabel("Experiment #", fontsize=12)
plt.ylabel("Time (s)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Memory Usage Comparison Plot
plt.figure(figsize=(14, 7))
for i, label in enumerate(models):
    plt.plot(experiments, memory_usage[i], marker='o', linestyle='-', label=label)
plt.title("Memory Usage Comparison Across Models", fontsize=16)
plt.xlabel("Experiment #", fontsize=12)
plt.ylabel("Memory Usage (MB)", fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Combined Comparison
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Accuracy
axs[0].set_title("Accuracy Comparison", fontsize=14)
for i, label in enumerate(models):
    axs[0].plot(experiments, accuracy[i], label=label, marker='o')
axs[0].set_ylabel("Accuracy (%)")
axs[0].legend()
axs[0].grid(True)

# Processing Time
axs[1].set_title("Processing Time Comparison", fontsize=14)
for i, label in enumerate(models):
    axs[1].plot(experiments, processing_time[i], label=label, marker='o')
axs[1].set_ylabel("Time (s)")
axs[1].legend()
axs[1].grid(True)

# Memory Usage
axs[2].set_title("Memory Usage Comparison", fontsize=14)
for i, label in enumerate(models):
    axs[2].plot(experiments, memory_usage[i], label=label, marker='o')
axs[2].set_ylabel("Memory (MB)")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Combined Comparison
fig, axs = plt.subplots(3, 1, figsize=(12, 18))

# Accuracy
axs[0].set_title("Accuracy Comparison", fontsize=14)
for i, label in enumerate(models2):
    axs[0].plot(experiments, accuracy[i], label=label, marker='o')
axs[0].set_ylabel("Accuracy (%)")
axs[0].legend()
axs[0].grid(True)

# Processing Time
axs[1].set_title("Processing Time Comparison", fontsize=14)
for i, label in enumerate(models2):
    axs[1].plot(experiments, processing_time[i], label=label, marker='o')
axs[1].set_ylabel("Time (s)")
axs[1].legend()
axs[1].grid(True)

# Memory Usage
axs[2].set_title("Memory Usage Comparison", fontsize=14)
for i, label in enumerate(models2):
    axs[2].plot(experiments, memory_usage[i], label=label, marker='o')
axs[2].set_ylabel("Memory (MB)")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()