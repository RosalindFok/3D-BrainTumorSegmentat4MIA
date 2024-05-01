"""
process data

Note:
    2015
    2017{ 
        url = https://www.med.upenn.edu/sbia/brats2017/data.html
        Brats17TrainingData: the same as 2018 Training{
            HGG(210)
            LGG( 75)
        }
        : no seg.
    }
    2018{
        url = https://www.med.upenn.edu/sbia/brats2018/data.html
        MICCAI_BraTS_2018_Data_Training: HGG(glioblastoma), LGG(lower grade glioma){
            HGG(210) {
                t1: native
                t1ce: T1 Contrast Enhanced
                t2: T2-weighted
                flair: T2 Fluid Attenuated Inversion Recovery
                seg: segmentation
            }
            LGG( 75){
                t1, t1ce, t2, flair, seg
            }
            shape: (240, 240, 155)
        }
        MICCAI_BraTS_2018_Data_Validation and MICCAI_BraTS_2018_Data_Testing_LFang: no seg.
    }
"""

import re
import os
import time
import h5py
import numpy as np
import nibabel as nib
from tqdm import tqdm

from load_path import hdf5_path_list, H_subjects_list, L_subjects_list

# filters and returns a list of strings from 'str_list' that contain the substring 'substr'
find_first_match = lambda substr, str_list: [x for x in str_list if substr in x][0]
# Loads the header and data array of a NIfTI file specified by 'file_path' using nibabel.
read_nii_head_and_data = lambda file_path: [nib.load(file_path).header, nib.load(file_path).get_fdata().astype(np.int16)]


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


def raw2hdf5(tag : str, subtag : str, sublist : list[str], hdf5_file : h5py._hl.files.File) -> None:
    group = hdf5_file.create_group(subtag)
    for index, subject_path in enumerate(tqdm(sublist, desc=''.join([tag,'_',subtag]), leave=True)):
        all_files = [os.path.join(subject_path, x) for x in os.listdir(subject_path)]
        
        # multimodal scans: {0:flair, 1:t1, 2:t1ce, 3:t2, 4:seg}
        [head, data_flair] = read_nii_head_and_data(file_path=find_first_match(substr=''.join(['_', 'flair','.']), str_list=all_files))
        [head, data_t1   ] = read_nii_head_and_data(file_path=find_first_match(substr=''.join(['_', 't1'   ,'.']), str_list=all_files))
        [head, data_t1ce ] = read_nii_head_and_data(file_path=find_first_match(substr=''.join(['_', 't1ce' ,'.']), str_list=all_files))
        [head, data_t2   ] = read_nii_head_and_data(file_path=find_first_match(substr=''.join(['_', 't2'   ,'.']), str_list=all_files))
        [head, data_seg  ] = read_nii_head_and_data(file_path=find_first_match(substr=''.join(['_', 'seg'  ,'.']), str_list=all_files))
        # Input Matrix
        data_flair = min_max_normalize(crop_nparray(data_flair))
        data_t1    = min_max_normalize(crop_nparray(data_t1))
        data_t1ce  = min_max_normalize(crop_nparray(data_t1ce))
        data_t2    = min_max_normalize(crop_nparray(data_t2))
        # Output Target Matrix
        data_seg   = crop_nparray(data_seg)
        # Concatenate Input and Output Target Matrices and save as hdf5 
        data       = np.array([data_flair, data_t1, data_t1ce, data_t2, data_seg], dtype=np.float16)
        hdf5_data  = group.create_dataset(str(index), data.shape, dtype=data.dtype)
        hdf5_data[:] = data

for hdf5_path, subjects_list in zip(hdf5_path_list, [H_subjects_list, L_subjects_list]):
    tag = re.search(r'(.*?)\.hdf5', hdf5_path).group(1)[-3:] # HGG LGG

    train_list = subjects_list[len(subjects_list)//7*0 : len(subjects_list)//7*5]
    valid_list = subjects_list[len(subjects_list)//7*5 : len(subjects_list)//7*6]
    test_list  = subjects_list[len(subjects_list)//7*6 : len(subjects_list)//7*7]

    if not os.path.exists(path=hdf5_path):
        start_time = time.time()
        with h5py.File(name=hdf5_path, mode='w') as hdf5_file:
            for subtag, sublist in zip(['train', 'valid', 'test'], [train_list, valid_list, test_list]):
                raw2hdf5(tag=tag, subtag=subtag, sublist=sublist, hdf5_file=hdf5_file)
        end_time = time.time()
        print(f'It took {round((end_time-start_time)/60, 2)} minutes to write {hdf5_path}')