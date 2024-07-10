import os
import numpy as np
import open3d as o3d

def load_as_array(path):
    # Loads the data from an .xyz file into a numpy array.
    # Also returns a boolean indicating whether per-point labels are available.
    data_array = np.loadtxt(path, comments='//') # raw data as np array, of shape (nx6) or (nx8) if labels are available.
    labels_available = data_array.shape[1] == 8
    return data_array, labels_available

def detach_instance(data, cls):
    # Find positions of points (their row numbers in .xyz file) in a specific class/instance 
    instance_max = 0
    class_list = []
    instance_list = []
    
    for row in range (0, data.shape[0]):
        # Find and save positions of points in the 'cls' class into a list
        if data[row,6] == cls:
            class_list.append(row)
    
    for index in class_list:
        # Find the highest instance label contained in 'cls' class
        if data[index,7] >= instance_max:
            instance_max = int(data[index,7])
                
    for instance in range (1, instance_max + 1, 1):
        # Find and save positions of points in each instance of the 'cls' class into a nested list
        point_list = []
        for index in class_list:
            if data[index, 7] == instance:
                point_list.append(index)
        instance_list.append(point_list)        
    return class_list, instance_list

def generate_colors(numbers):
    # Generates a list of random colors
    np.random.seed(40)
    return np.random.rand(numbers, 3)

def segment(class_segmentation):
    # Paints points with different colors based on type of segmentation
                 
    # Convert colors to a NumPy array
    colors = np.asarray(pc.colors)
          
    if class_segmentation:
        # Perform class segmentation
        # Create a list of random colors
        color_list = generate_colors(9)
        # Change the color of the points 
        for cls in range (1, 10):
            class_list, instance_list = detach_instance(data, cls) 
            for index in class_list:
                colors[index] = color_list[cls-1]
    else:
        # Perform instance segmentation
        for cls in range (1, 10):
            class_list, instance_list = detach_instance(data, cls) 
            # Create a list of random colors
            color_list = generate_colors((len(instance_list)))
            # Change the color of the points
            i = 0  
            for instance in instance_list:
                for index in instance:
                    colors[index] = color_list[i]
                i += 1
                        
    # Convert the modified colors back to Vector3dVector
    pc.colors = o3d.utility.Vector3dVector(colors)         
    return pc, labels_available, labels

def toggle_visualisation(vis):
    # The callback function
    global class_segmentation
    class_segmentation = not class_segmentation
    if class_segmentation:
        segment(class_segmentation)
    else:
        segment(class_segmentation)
    vis.update_geometry(pc)
    vis.update_renderer()
    return False
    

def save_data_as_xyz(data, labels_available, path):
    # To save your own data in the same format as we used, you can use this function.
    # Edit as needed with more or fewer columns.
    with open(path, 'w') as f:
        f.write("//X Y Z R G B class instance\n")
        if labels_available:        
            np.savetxt(f, data, fmt=['%.6f', '%.6f', '%.6f','%d', '%d', '%d', '%d', '%d'])
        else:
            np.savetxt(f, data, fmt=['%.6f', '%.6f', '%.6f','%d', '%d', '%d'])
    return


if __name__ == '__main__':
    '''Example usage of the functions defined in this file'''
    # Please specify where the data set is located:
    data_dir = 'C:\\Users\\LENOVO\\Desktop\\Intern\\'
    
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
    data, labels_available = load_as_array(data_dir + file_name)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(data[:,0:3])
    pc.colors = o3d.utility.Vector3dVector(data[:,3:6]/255)   
    labels = None
    if labels_available:
        labels = data[:, 6:]     
    
    # Save data back to a file with the same format
    save_data_as_xyz(data, labels_available, "C:\\Users\\LENOVO\\Desktop\\Intern\\saved_file.xyz")
    
    # Initilise the visulisation setting
    class_segmentation = True
    segment(class_segmentation)
    
    # Set up O3d visuliser
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pc)
    vis.register_key_callback(ord(" "), toggle_visualisation)

    # Visualize the point cloud with segmentation:
    vis.run()
    vis.destroy_window()
    