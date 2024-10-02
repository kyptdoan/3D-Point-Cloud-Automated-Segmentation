import numpy as np
import copy
from scipy.optimize import linear_sum_assignment

def load_as_array(path):
    # Loads the data from an .xyz file into a numpy array.
    # Also returns a boolean indicating whether per-point labels are available.
    data_array = np.loadtxt(path, comments='//') # raw data as np array, of shape (nx6) or (nx8) if labels are available.
    return data_array

def extract_candidates(data, id_col):
    """
    Extract and store position of points belong to each cluster in GT and RS files
    Args:
        data (_np_array_): _GT/RS array read from input files_

    Returns:
        _candidates_dict_: _a dictionary contains cluster ids as keys and points' positions as values_
    """
    candidates_dict = dict()
    id_list = [data[0][id_col]]
    
    for i in range (data.shape[0]):
        if data[i][id_col] not in id_list:
            id_list.append(data[i][id_col])
            
    for id in id_list:
        candidates_dict[id] = []
    
    for row_idx in range (data.shape[0]):
        candidates_dict[data[row_idx][id_col]].append(row_idx)
        
    return candidates_dict, id_list

def calculate_iou(gt_candidate_idx, rs_candidate_idx):
    """
    This canculate iou value of a gt candidate and a rs candidate
    Args:
        gt_candidate_idx (_list_): _a list contains lists of positions corresponding to a cluster in GT._
        rs_candidate_idx (_list_): _a list contains lists of positions corresponding to a cluster in GT._
    
    Returns:
        iou (_float_): the iou value.
    """
    # Convert lists to sets for faster operations
    gt_set = set(gt_candidate_idx)
    rs_set = set(rs_candidate_idx)
    
    # Calculate intersection and union using set operations
    intersection = len(gt_set & rs_set)
    union = len(gt_set | rs_set)
    
    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    return iou

def create_iou_matrix(gt_data, rs_data, gt_id_col, rs_id_col):
    """
    This create a NxM matrix (N: number of GT clusters, M: number of RS clusters), whose each entries is an iou value of a GT-RS candidate pair.
    Args:
        gt_data (_np_array_): _an array contains GT data read from GT file._
        rs_data (_np_array_): _an array contains RS data read from RS file._
    Returns:
        iou_matrix (_np_array_): _the matrix contains iou values._
    """
    # Extract candidates from GT and RS data
    gt_candidates_dict, gt_id_list = extract_candidates(gt_data, gt_id_col)
    rs_candidates_dict, rs_id_list = extract_candidates(rs_data, rs_id_col)
    
    # Preallocate an IoU matrix
    iou_matrix = np.zeros((len(gt_candidates_dict), len(rs_candidates_dict)))
    
    # Vectorized computation of IoU values
    for i, gt_candidate in enumerate(gt_candidates_dict.values()):
        for j, rs_candidate in enumerate(rs_candidates_dict.values()):
            iou_matrix[i, j] = calculate_iou(gt_candidate, rs_candidate)
    
    # Modify the IoU values for the Hungarian algorithm
    return iou_matrix, gt_id_list, rs_id_list

if __name__ == '__main__':
    # Specify paths
    ground_truth_path = "C:\\Users\\LENOVO\\Desktop\\Project\\A2_20220608_a.xyz"
    annotated_result_path = "C:\\Users\\LENOVO\\Desktop\\Project\\Strawberry-Project-\\DBSCAN\\DBSCAN_iou\\A2_20220608_DBSCAN_iou.xyz"
    
    # Load GT and result into numpy arrays
    ground_truth_data = load_as_array(ground_truth_path)
    annotated_result_data = load_as_array(annotated_result_path)
    
    while True:
        print("Please choose segmentation type ('1': semantic/'2': instance): ")
        segmentation_type = input()
        if segmentation_type != '1' and segmentation_type != '2':
            print("Please press '1' or '2'.")
            continue
        elif segmentation_type == '1':
            gt_id_col = 6
            rs_id_col = 6
            break
        elif segmentation_type == '2':
            gt_id_col = 7
            rs_id_col = 6
            break
    
    # IoU matrix
    iou_matrix, gt_id_list, rs_id_list = create_iou_matrix(ground_truth_data, annotated_result_data, gt_id_col, rs_id_col)
    
    # Find assigments that gives max iou
    # print(ori_iou_matrix)
    row_idx, col_idx = linear_sum_assignment(iou_matrix, True)
    
    # Write outputs into a text file
    with open("C:\\Users\\LENOVO\\Desktop\\Project\\Strawberry-Project-\\DBSCAN\\DBSCAN_iou\\DBSCAN_iou.txt", 'a') as file:
        file.write(f"GT cluster ids: ")
        for id in gt_id_list:
            file.write(f"{id} ")
        
        file.write("\n")
        
        file.write(f"RS cluster ids: ")
        for id in rs_id_list:
            file.write(f"{id} ")
        
        file.write("\n")
        
        file.write(f"Assignments (GT_id : RS_id):\n")
        for i in range (len(row_idx)):
            file.write(f"{gt_id_list[row_idx[i]]} : {rs_id_list[col_idx[i]]}\n")
        
        file.write(f"Maximum mean IoU: {iou_matrix[row_idx, col_idx].sum() / len(gt_id_list)}\n")
            
    print(iou_matrix)
    print(gt_id_list)
    print(rs_id_list)
    print(row_idx)
    print(col_idx)
    print(iou_matrix[row_idx, col_idx].sum() / len(gt_id_list))      