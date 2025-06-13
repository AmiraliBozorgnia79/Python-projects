import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Define dimension list and number of samples
dimension_list = [1, 2, 5, 10, 20, 50, 100]
N = 1000
# To store results
avg_distances = []
std_distances = []

for d in dimension_list:
    print(f"\nDimension: {d}")
    np.random.seed(42)
    random_array = np.random.rand(N, d)

    # Initialize distance matrix
    euc_dis = np.zeros((N, N))

    # Nested loop to compute pairwise distances
    for i in range(N):
        for j in range(N):
            if i != j:
                euc_dis[i, j] = linalg.norm(random_array[i] - random_array[j])
            else:
                euc_dis[i, j] = np.inf  # Avoid self-distance

    # ✅ Move these OUTSIDE the inner loops
    nearest_distances = euc_dis.min(axis=1)
    avg_distances.append(nearest_distances.mean())
    std_distances.append(nearest_distances.std())

    print(f"  → Avg NN distance: {nearest_distances.mean():.4f}")
    print(f"  → Std NN distance: {nearest_distances.std():.4f}")

#  Plotting
# Compute CV
cv = np.array(std_distances) / np.array(avg_distances)

# Plot all three in subplots
plt.figure(figsize=(15, 4))

# Plot 1: Average NN Distance
plt.subplot(1, 3, 1)
plt.plot(dimension_list, avg_distances, marker='o', color='blue')
plt.title('Average Nearest Neighbor Distance')
plt.xlabel('Dimension (d)')
plt.ylabel('Average Distance')
plt.grid(True)
# Plot 2: Standard Deviation
plt.subplot(1, 3, 2)
plt.plot(dimension_list, std_distances, marker='o', color='orange')
plt.title('Standard Deviation of NN Distance')
plt.xlabel('Dimension (d)')
plt.ylabel('Standard Deviation')
plt.grid(True)
# Plot 3: Coefficient of Variation
plt.subplot(1, 3, 3)
plt.plot(dimension_list, cv, marker='o', color='green')
plt.title('Coefficient of Variation (CV)')
plt.xlabel('Dimension (d)')
plt.ylabel('CV = std / mean')
plt.grid(True)
plt.tight_layout()
plt.show()
