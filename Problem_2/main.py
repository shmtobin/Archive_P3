# Problem 2: Local Density of States for Electrons

import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

folder = "Local_density_of_states_near_band_edge"
heatmap_folder = os.path.join(folder, "local_density_of_states_heatmap")
heightmap_folder = os.path.join(folder, "local_density_of_states_height")
os.makedirs(heatmap_folder, exist_ok=True)
os.makedirs(heightmap_folder, exist_ok=True)

data_dict = {}

for i in range(10):  # levels 0 to 9
    filename = f"{folder}/local_density_of_states_for_level_{i}.txt"
    try:
        data_dict[f"data_{i}"] = np.genfromtxt(filename, delimiter=",")
    except Exception as e:
        print(f"Error loading {filename}: {e}")

# convert dictionary entries into Pandas DataFrames
for key, value in data_dict.items():
    globals()[key] = pd.DataFrame(value)

# select a sub-region (e.g., central 10x10 region)
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
plt.savefig("Plots/Local_Electron_Density_Heatmaps.png", bbox_inches='tight')
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
plt.savefig("Plots/Local_Density_of_States_Surface_Plots.png", bbox_inches='tight')
plt.show()
print("3D surface plots generated successfully.")

# c) Select a local sub-region and quantitatively illustrate the 
# changes while offering some physical speculations.

subregion_stats = {}

for i in range(9):
    key = f"data_{i}"
    if key in globals():
        df = globals()[key]
        # define the central subregion (10x10 region)
        center_x, center_y = df.shape[1] // 2, df.shape[0] // 2
        subregion = df.iloc[center_y - region_size//2 : center_y + region_size//2, 
                              center_x - region_size//2 : center_x + region_size//2]
        # calculate subregion mean and standard deviation
        sub_mean = subregion.values.mean()
        sub_std = subregion.values.std()
        # calculate the global (full matrix) mean for comparison
        global_mean = df.values.mean()
        # compute the difference between subregion and global means
        diff = sub_mean - global_mean
        # compute the ratio of subregion to global mean
        ratio = sub_mean / global_mean
        # save all computed values
        subregion_stats[key] = {"mean": sub_mean, "std": sub_std, 
                                  "global_mean": global_mean, "diff": diff, "ratio": ratio}

# prepare lists of metrics for plotting across energy levels (files 0-8)
levels = list(range(9))
means = [subregion_stats[f"data_{i}"]["mean"] for i in levels]
stds = [subregion_stats[f"data_{i}"]["std"] for i in levels]
diffs = [subregion_stats[f"data_{i}"]["diff"] for i in levels]
ratios = [subregion_stats[f"data_{i}"]["ratio"] for i in levels]

plt.figure(figsize=(8, 6))
plt.errorbar(levels, means, yerr=stds, fmt='o-', capsize=5, color='red', ecolor='black')
plt.xlabel("Energy Level Index")
plt.ylabel("Subregion Mean Density (with Std Dev)")
plt.title("Subregion Mean Density with Variability Across Energy Levels")
plt.grid(True)
plt.savefig("Plots/Subregion_Mean_Density_with_Variability.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(8, 6))
plt.bar(levels, diffs, color='skyblue')
plt.xlabel("Energy Level Index")
plt.ylabel("Difference (Subregion Mean - Global Mean)")
plt.title("Deviation of Subregion Density from Global Average")
plt.grid(axis='y')
plt.savefig("Plots/Deviation_of_Subregion_Density_from_Global_Average.png", bbox_inches='tight')
plt.show()


print("Extended subregion analysis completed.")
print("\nObservations & Speculations:")
print("- The error bar plot indicates not only the average local density within the subregion but also the spread (variability) of electron density. A high standard deviation at certain levels may suggest localized fluctuations or inhomogeneities, which could be linked to electron localization phenomena or variations in the local potential landscape.")
print("- The bar plot showing the difference between subregion and global mean density helps to identify energy levels where the central region is either significantly enhanced or suppressed compared to the overall material. Such deviations could point to regions of localized states or potential wells/barriers influencing electron behavior.")
print("- Additionally, the ratio plot (if considered) provides a normalized measure of how pronounced the subregion density is relative to the entire system, which might be useful for comparing across different materials or simulation conditions.")

# wanted to test out clustering algorithm and found DBSCAN

# dictionary to store clustering results for each level
clustering_results = {}

# define a multiplier for the threshold (you can adjust this)
threshold_multiplier = 1.0

print("Performing clustering analysis on high-density regions...")
for i in range(9):
    key = f"data_{i}"
    if key in globals():
        df = globals()[key]
        density_array = df.values
        # Define a threshold: here we use global mean + threshold_multiplier*std
        global_mean = density_array.mean()
        global_std = density_array.std()
        threshold = global_mean + threshold_multiplier * global_std
        
        # get indices (row, col) of pixels above the threshold
        indices = np.argwhere(density_array > threshold)
        
        # if no pixels exceed threshold, record zero clusters
        if len(indices) == 0:
            clustering_results[key] = {"num_clusters": 0, "cluster_sizes": [], "indices": indices, "labels": None, "threshold": threshold}
            continue
        
        # run DBSCAN on these indices.
        # eps: maximum distance for two points to be considered neighbors (in pixels).
        # min_samples: minimum number of points to form a cluster.
        clustering = DBSCAN(eps=1.5, min_samples=3)
        cluster_labels = clustering.fit_predict(indices)
        
        # exclude noise points (label == -1) when counting clusters.
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        num_clusters = len(unique_labels)
        
        # compute cluster sizes (number of points in each cluster)
        cluster_sizes = []
        for label in unique_labels:
            size = np.sum(cluster_labels == label)
            cluster_sizes.append(size)
        
        clustering_results[key] = {"num_clusters": num_clusters, 
                                   "cluster_sizes": cluster_sizes,
                                   "indices": indices,
                                   "labels": cluster_labels,
                                   "threshold": threshold}

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle("DBSCAN Clustering of High-Density Regions in Each Energy Level")

for i, ax in enumerate(axes.flat):
    key = f"data_{i}"
    if key in clustering_results:
        res = clustering_results[key]
        indices = res["indices"]
        labels = res["labels"]
        
        # if no points, note it
        if indices.shape[0] == 0 or labels is None:
            ax.text(0.5, 0.5, "No high-density regions", horizontalalignment='center', verticalalignment='center')
            ax.set_title(f"Level {i}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            continue
        
        # get unique labels for plotting
        unique_labels = set(labels)
        # create a color palette; noise points (-1) will be colored black.
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            class_member_mask = (labels == k)
            xy = indices[class_member_mask]
            if k == -1:
                # noise is plotted in black
                col = 'k'
                marker = 'x'
            else:
                marker = 'o'
            ax.plot(xy[:, 1], xy[:, 0], marker, markerfacecolor=col, markeredgecolor='k', markersize=6, alpha=0.6)
        
        ax.set_title(f"Level {i}: {res['num_clusters']} clusters")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
plt.tight_layout()
plt.savefig("Plots/DBSCAN_Clustering_of_High_Density_Regions.png", bbox_inches='tight')
plt.show()

levels = list(range(9))
num_clusters_list = [clustering_results[f"data_{i}"]["num_clusters"] for i in levels]

plt.figure(figsize=(8, 6))
plt.plot(levels, num_clusters_list, marker='o', linestyle='-', color='magenta')
plt.xlabel("Energy Level Index")
plt.ylabel("Number of Clusters")
plt.title("Number of High-Density Clusters Across Energy Levels")
plt.grid(True)
plt.savefig("Plots/Number_of_High_Density_Clusters.png", bbox_inches='tight')
plt.show()

avg_cluster_sizes = []
for i in levels:
    cs = clustering_results[f"data_{i}"]["cluster_sizes"]
    if cs:
        avg_cluster_sizes.append(np.mean(cs))
    else:
        avg_cluster_sizes.append(0)

plt.figure(figsize=(8, 6))
plt.plot(levels, avg_cluster_sizes, marker='s', linestyle='--', color='teal')
plt.xlabel("Energy Level Index")
plt.ylabel("Average Cluster Size (pixels)")
plt.title("Average Size of High-Density Clusters Across Energy Levels")
plt.grid(True)
plt.savefig("Plots/Average_Size_of_High_Density_Clusters.png", bbox_inches='tight')
plt.show()