# Problem 2: Local Density of States for Electrons

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

folder = "Local_density_of_states_near_band_edge"
heatmap_folder = os.path.join(folder, "local_density_of_states_heatmap")
heightmap_folder = os.path.join(folder, "local_density_of_states_height")
os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(heightmap_folder, exist_ok=True)

data_dict = {}

for i in range(10):  # Levels 0 to 9
    filename = f"{folder}/local_density_of_states_for_level_{i}.txt"
    try:
        data_dict[f"data_{i}"] = np.genfromtxt(filename, delimiter=",")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# Convert dictionary entries into Pandas DataFrames
for key, value in data_dict.items():
    globals()[key] = pd.DataFrame(value)

# Select a sub-region (e.g., central 10x10 region)
subregion_means = {}
region_size = 10

# a) Generate a 2-dimensional heatmap depicting the local 
# electron density.
print("Generating heatmaps...")
for i in range(9):
    key = f"data_{i}"
    if key in globals():
        df = globals()[key]
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, cmap="inferno", cbar=True)
        plt.title(f"Local Electron Density - {key}")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.savefig(os.path.join(heatmap_folder, f"heatmap_{i}.png"))
        plt.close()

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("Local Electron Density Heatmaps")

for i, ax in enumerate(axes.flat):
    key = f"data_{i}"
    if key in globals():
        df = globals()[key]
        sns.heatmap(df, cmap="inferno", cbar=True, ax=ax)
        ax.set_title(f"{key}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")

plt.tight_layout()
plt.show()
print("Heatmaps generated successfully.")

# b) Create a 2-dimensional surface plot where the height 
# profile represents the local density of states.
print("Generating 3D surface plots...")
for i in range(9):
    key = f"data_{i}"
    if key in globals():
        df = globals()[key]
        x = np.arange(df.shape[1])
        y = np.arange(df.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = df.values
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f"Local Density of States - {key}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Density")
        plt.savefig(os.path.join(heightmap_folder, f"heightmap_{i}.png"))
        plt.close()

fig, axes = plt.subplots(3, 3, figsize=(15, 15), subplot_kw={'projection': '3d'})
fig.suptitle("Local Density of States Surface Plots")

for i, ax in enumerate(axes.flat):
    key = f"data_{i}"
    if key in globals():
        df = globals()[key]
        x = np.arange(df.shape[1])
        y = np.arange(df.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = df.values
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(f"{key}")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Density")

plt.tight_layout()
plt.show()
print("3D surface plots generated successfully.")

# c) Select a local sub-region and quantitatively illustrate the 
# changes while offering some physical speculations.
print("Analyzing subregion changes...")
for i in range(9):
    key = f"data_{i}"
    if key in globals():
        df = globals()[key]
        center_x, center_y = df.shape[1] // 2, df.shape[0] // 2
        subregion = df.iloc[center_y - region_size//2:center_y + region_size//2, center_x - region_size//2:center_x + region_size//2]
        subregion_means[key] = subregion.mean().mean()

plt.figure(figsize=(8, 6))
plt.plot(list(subregion_means.keys()), list(subregion_means.values()), marker='o', linestyle='-')
plt.xlabel("File Index")
plt.ylabel("Average Local Density of States in Subregion")
plt.title("Changes in Local Density of States Across Levels")
plt.show()
print("Subregion analysis completed.")
