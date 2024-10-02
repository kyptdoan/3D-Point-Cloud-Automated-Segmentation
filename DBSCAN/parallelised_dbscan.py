import numpy as np
import math
from scipy.spatial import KDTree
from skimage.color import rgb2lab
import multiprocessing as mp
import copy
import os
import open3d as o3d

def load_as_array(path):
    # Loads the data from an .xyz file into a numpy array.
    # Also returns a boolean indicating whether per-point labels are available.
    data_array = np.loadtxt(path, comments='//') # raw data as np array, of shape (nx6) or (nx8) if labels are available.
    return data_array

def load_as_o3d_cloud(path):
    # Loads the data from an .xyz file into an open3d point cloud object.
    # Extract spatial information and convert to a numpy array
    data = load_as_array(path)
    points = data
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,0:3])
    pc.colors = o3d.utility.Vector3dVector(data[:,3:6])
    coordinates = np.asarray(pc.points)
    return data, pc, coordinates

def rgb_to_lab(rgb_array):
    """ Convert RGB values to CIELAB color space. """
    # Normalize the RGB values to [0, 1] range
    rgb_array_normalized = rgb_array / 255.0
    # Convert RGB to LAB
    lab_array = rgb2lab(rgb_array_normalized)
    return lab_array

# def visualise_segmented_pc(clusters, pc):
    # Generate random colors
    np.random.seed(20)
    
    n_clusters = len(clusters)
    
    color_list = np.random.rand(n_clusters, 3)      
    segmented_colors = np.zeros_like(data[:,3:6])
            
    pc.colors = o3d.utility.Vector3dVector(segmented_colors)
    o3d.visualization.draw_geometries([pc])

UNCLASSIFIED = -2
NOISE = -1


if __name__ == '__main__':
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
    
    points = [Point(x, y, z, r, g, b) for x, y, z, r, g, b in data]
    # print(points)
    clustered = DBSCAN(Points(points), n_pred, 15, w_card)
    print(clustered)
    # visualise_segmented_pc(clustered, pc)
# Example usage (assuming rgb_to_lab function is defined elsewhere):
# points_data = [Point(0,0,0,255,0,0), Point(1,1,1,255,0,0), ... ]
# clusters = DBSCAN(points_data, n_pred, min_card=3, w_card=w_card)