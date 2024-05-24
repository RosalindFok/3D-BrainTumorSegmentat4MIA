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
from tqdm import tqdm

from load_path import hdf5_path_list, H_subjects_2018_list, L_subjects_2017_2018_list
from read_nii_utils import read_nii_head_and_data, find_first_match, min_max_normalize, crop_nparray

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

for hdf5_path, subjects_list in zip(hdf5_path_list, [H_subjects_2018_list, L_subjects_2017_2018_list]):
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