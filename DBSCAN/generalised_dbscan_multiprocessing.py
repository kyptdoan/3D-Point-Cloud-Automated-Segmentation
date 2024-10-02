import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import open3d as o3d
import cv2
from skimage.color import rgb2lab
import math
import matplotlib.pyplot as plt

# Define a cluster as a set of points which have both near positions and near colors
# Distance used in dbscan is a sum of spatial and color distance (with hyper parameters)
# Color distance is calculated using CIELAB color space

def load_as_array(path):
    # Loads the data from an .xyz file into a numpy array.
    # Also returns a boolean indicating whether per-point labels are available.
    data_array = np.loadtxt(path, comments='//') # raw data as np array, of shape (nx6) or (nx8) if labels are available.
    return data_array

def load_as_o3d_cloud(path):
    # Loads the data from an .xyz file into an open3d point cloud object.
    # Extract spatial information and convert to a numpy array
    data = load_as_array(path)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,0:3])
    pc.colors = o3d.utility.Vector3dVector(data[:,3:6])
    coordinates = np.asarray(pc.points)
    return data, pc, coordinates

# DBSCAN IMPLEMENTATION WITH MODIFIED DISTANCE
UNCLASSIFIED = False
NOISE = -1

def rgb_to_lab(rgb_array):
    """ Convert RGB values to CIELAB color space. """
    # Normalize the RGB values to [0, 1] range
    rgb_array_normalized = rgb_array / 255.0
    # Convert RGB to LAB
    lab_array = rgb2lab(rgb_array_normalized)
    return lab_array

def calculate_dist(p_xyz, q_xyz, p_rgb, q_rgb):
    spatial_dist = np.sqrt(np.sum((p_xyz - q_xyz) ** 2))
    p_lab = rgb_to_lab(p_rgb)
    q_lab = rgb_to_lab(q_rgb)
    color_dist = np.sqrt(np.sum((p_lab - q_lab) ** 2))
    return spatial_dist + 0.01*color_dist

def _eps_neighborhood(p_xyz, q_xyz, p_rgb, q_rgb, eps):
    return calculate_dist(p_xyz, q_xyz, p_rgb, q_rgb) < eps

def _region_query(data, point_id, eps):
    n_points = data.shape[0]  # Number of points (rows)
    seeds = []
    for i in range(n_points):
        if _eps_neighborhood(data[point_id, :3], data[i, :3], data[point_id,3:6], data[i,3:6], eps): 
            seeds.append(i)
    return seeds

def _expand_cluster(data, classifications, point_id, cluster_id, eps, min_points):
    seeds = _region_query(data, point_id, eps)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id
            
        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(data, current_point, eps)
            if len(results) >= min_points:
                for result_point in results:
                    if classifications[result_point] == UNCLASSIFIED or \
                       classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True
        
def dbscan(data, eps, min_points):
    """
    Inputs:
    data - A NumPy array where each row is a point, and the first three columns are XYZ
    eps - Maximum distance two points can be to be regionally related
    min_points - The minimum number of points to make a cluster
    
    Outputs:
    classifications - an array with either a cluster id number or dbscan.NOISE (None) for each
    row in m.
    
    """
    cluster_id = 1
    n_points = data.shape[0]  # Number of points (rows)
    classifications = [UNCLASSIFIED] * n_points
    for point_id in range(n_points):
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(data, classifications, point_id, cluster_id, eps, min_points):
                cluster_id += 1
    # numbers of clusters & noise
    n_clusters = len(set(classifications)) - (1 if -1 in classifications else 0)
    n_noise = list(classifications).count(-1)
    return classifications, n_clusters, n_noise

def visualise_segmented_pc(classifications, pc, n_clusters, n_noise):
    # Generate random colors
    np.random.seed(20)
    
    if n_noise != 0:
        n_colors = n_clusters + 1
    else:
        n_colors = n_clusters
    
    color_list = np.random.rand(n_colors, 3)      
    segmented_colors = np.zeros_like(data[:,3:6])
    for i in range(n_clusters):
        segmented_colors[np.array(classifications).flatten() == i-1] = color_list[i]
    pc.colors = o3d.utility.Vector3dVector(segmented_colors)
    o3d.visualization.draw_geometries([pc])

if __name__ == '__main__':
    print(mp.cpu_count())
    # Please specify where the data set is located:
    data_dir = 'C:\\Users\\LENOVO\\Desktop\\Project\\'
    
    all_files = sorted(os.listdir(data_dir))
    scans = [fname for fname in all_files if fname.endswith(".xyz")]
    
    # Take file name from user and check for the file existence
    existence = False
    while True:
        file_name = input("Please type in your chosen file (including extension field): ")
        for file in scans:
            if file_name == file:
                existence = True
                break
        if existence: 
            break
        else:
            print("Your chosen file does not exist in this repository. Please choose another one.")
            continue
    
    # Loads the data from an .xyz file into an open3d point cloud object.
    data, pc, coordinates = load_as_o3d_cloud(data_dir + file_name)
    
    # o3d.visualization.draw_geometries([down_pc])
    # print(down_coordinates)
    classifications, n_clusters, n_noise = dbscan(data, 3.5, 50)

    visualise_segmented_pc(classifications, pc, n_clusters, n_noise)
    print(n_clusters)
    print(n_noise)
    