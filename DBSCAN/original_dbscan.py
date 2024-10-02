import os
import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.cluster import DBSCAN
import copy
import matplotlib.pyplot as plt
import cv2
import sys
import time

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
    colors = np.asarray(pc.colors)
    return data[:,0:6], pc, coordinates, colors

def perform_dbscan(data):
    dbscan = DBSCAN(eps=10, min_samples=200).fit(pd.DataFrame(data))
    # dbscan = DBSCAN(eps=epsilon, min_samples=min_pts, metric=custom_distance)
    labels = dbscan.labels_
    # numbers of clusters & noise
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    return labels, n_clusters, n_noise

def visualise_segmented_pc(segmented_labels, pc, n_clusters, n_noise):
    # Generate random colors
    np.random.seed(20)
    
    if n_noise != 0:
        n_colors = n_clusters + 1
    else:
        n_colors = n_clusters
    
    color_list = np.random.rand(n_colors, 3)      
    segmented_colors = np.zeros_like(data[:,3:6])
    for i in range(n_clusters):
        segmented_colors[np.array(segmented_labels).flatten() == i-1] = color_list[i]
    pc.colors = o3d.utility.Vector3dVector(segmented_colors)
    o3d.visualization.draw_geometries([pc])
    
def save_data_as_xyz(data, segmented_labels, path):
    # To save your own data in the same format as we used, you can use this function.    
    # Concatenate the original data with the segmented labels
    labeled_data = np.hstack((data, segmented_labels.reshape(-1,1)))
    
    with open(path, 'w') as f:
        f.write("//X Y Z R G B class instance\n")       
        np.savetxt(f, labeled_data, fmt=['%.6f', '%.6f', '%.6f','%d', '%d', '%d', '%d'])
    return

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
    data, pc, coordinates, colors = load_as_o3d_cloud(data_dir + file_name)
    
    # runtime_report = []
    # for i in range (0, 3):
        # start = time.time()
    segmented_labels, n_clusters, n_noise = perform_dbscan(data)
        # end = time.time()
        # runtime_report.append(end - start)
    # mean_runtime = sum(runtime_report)/3
    # print(mean_runtime)
    #with open("C:\\Users\\LENOVO\\Desktop\\Project\\Strawberry-Project-\\Runtime_Evaluation\\runtime_report.txt", 'a') as file:
            #file.write(f"Average runtime: {mean_runtime} \n")
    visualise_segmented_pc(segmented_labels, pc, n_clusters, n_noise)
    print(n_clusters)
    print(n_noise)
    #save_data_as_xyz(data, segmented_labels, "C:\\Users\\LENOVO\\Desktop\\Project\\Strawberry-Project-\\DBSCAN\\DBSCAN_iou\\A2_20220608_DBSCAN_iou.xyz")