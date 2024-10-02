import os
import numpy as np
import open3d as o3d
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
    # Extract color information and convert to a numpy array
    data = load_as_array(path)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,0:3])
    pc.colors = o3d.utility.Vector3dVector(data[:,3:6])
    colors = np.asarray(pc.colors)
    return data, data[:,0:6], pc, np.float32(colors)

def perform_kmeans_clustering(colors, k):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.2)
    compactness, segmented_labels, centers = cv2.kmeans(colors, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return compactness, segmented_labels, np.uint8(centers)

def visualise_segmented_pc(segmented_labels, pc, k):
    # Generate random colors
    np.random.seed(20)
    color_list = np.random.rand(k, 3)
    segmented_colors = np.zeros_like(data[:,3:6])
    for i in range(k):
        segmented_colors[segmented_labels.flatten() == i] = color_list[i]
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
    segmentation_type = 0
    while True:
        print("Please choose segmentation type: 1. Semantic / 2. Instance")
        segmentation_type = input()
        if (segmentation_type != '1') and (segmentation_type != '2'):
            print("Please type 1 or 2.")
            continue
        else:
            break
    
    data_dir = 'C:\\Users\\LENOVO\\Desktop\Project\\'
    
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
        
    annotated_data, data, pc, colors = load_as_o3d_cloud(data_dir + file_name)
    
    # Semantic Segmentation:
    if segmentation_type == '1': 
        compactness, segmented_labels, centers = perform_kmeans_clustering(colors, 9)
        # Calculate runtime
        #runtime_report = []
        #for i in range (0, 30):
            #start = time.time()
            #compactness, segmented_labels, centers = perform_kmeans_clustering(colors, 9)
            #end = time.time()
            #runtime_report.append(end-start)
        #mean_runtime = sum(runtime_report)/30
        #print(mean_runtime)
        visualise_segmented_pc(segmented_labels, pc, 9)
        print(segmented_labels)
        # write average runtime result into file
        #with open("C:\\Users\\LENOVO\\Desktop\\Project\\Strawberry-Project-\\Runtime_Evaluation\\runtime_report.txt", 'a') as file:
            #file.write(f"Average runtime: {mean_runtime} ")
        # save_data_as_xyz(data, segmented_labels,"C:\\Users\\LENOVO\\Desktop\\Project\\Strawberry-Project-\\K-MEANS\\A2_20220512_rgb_kmeans_9.xyz")
    elif segmentation_type == '2':
        k = int(np.max(annotated_data[:,7:8]))
        compactness, segmented_labels, centers = perform_kmeans_clustering(colors, k)
        visualise_segmented_pc(segmented_labels, pc, k)
        print(k)
        print(segmented_labels)
        # save_data_as_xyz(data, segmented_labels,"C:\\Users\\LENOVO\\Desktop\\Project\\Strawberry-Project-\\K-MEANS\\A2_20220608_rgb_kmeans_17.xyz")

