"""
Utils for reading NIFTI files
"""
import numpy as np
import nibabel as nib

# filters and returns a list of strings from 'str_list' that contain the substring 'substr'
find_first_match = lambda substr, str_list: [x for x in str_list if substr in x][0]
# Loads the header and data array of a NIfTI file specified by 'file_path' using nibabel.
read_nii_head_and_data = lambda file_path: [nib.load(file_path).header, nib.load(file_path).get_fdata().astype(np.int16)]
read_nii_head_and_data_floatValue = lambda file_path: [nib.load(file_path).header, nib.load(file_path).get_fdata()]

def crop_nparray(original_matrix : np.ndarray, target_shape : tuple = (160, 192, 128)) -> np.ndarray:
    """Crop the center part of the original matrix to the target shape.

    Args:
        original_matrix: The original NumPy array to be cropped.
        target_shape: The target shape to crop the matrix to (default is (160, 192, 128)).

    Returns:
        np.ndarray: The cropped matrix.
    """
    # Calculate the start indices for cropping the center part
    start_indices = [(original_dim - target_dim) // 2 for original_dim, target_dim in zip(original_matrix.shape, target_shape)]
    # Calculate the end indices for cropping the center part
    end_indices = [start_idx + target_dim for start_idx, target_dim in zip(start_indices, target_shape)]
    # Crop the matrix to the center part based on calculated indices
    cropped_matrix = original_matrix[start_indices[0]:end_indices[0], start_indices[1]:end_indices[1], start_indices[2]:end_indices[2]]
    return cropped_matrix

def min_max_normalize(data : np.ndarray) -> np.ndarray:
    """
    Normalize the data to the range of [0, 1] using min-max normalization.
    """
    data_min = np.min(data)
    data_max = np.max(data)
    data_norm = (data - data_min) / (data_max - data_min) if data_max !=  data_min else np.ones_like(data)
    return data_norm