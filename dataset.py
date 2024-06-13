"""
load torch Dataset:
    2017 and 2018: read from hdf5 file
    2024: read from nii.gz file
"""
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

from read_nii_utils import find_first_match, read_nii_head_and_data_floatValue, crop_nparray, min_max_normalize

# If you read hdf5 file via torch Dataset in Windows, it will raise TypeError("h5py objects cannot be pickled") and TypeError: h5py objects cannot be pickled
class BraTSDataset_from_hdf5(Dataset):
    def __init__(self, hdf5_path : str, tag : str) -> None:
        '''
        tag = {train, valid, test}
        '''
        super().__init__()
        self.hdf5_path = hdf5_path
        self.tag = tag

    def open_hdf5(self):
        self.hdf5_file = h5py.File(self.hdf5_path, 'r')
        self.content = self.hdf5_file[self.tag]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, 'content'):
            self.open_hdf5()
        content = self.content[str(idx)][:]
        content = torch.Tensor(content)
        return content[:4], content[-1]
    
    def __len__(self) -> int:
        if not hasattr(self, 'content'):
            self.open_hdf5()
        return len(self.content.keys())
    
    def __del__(self) -> None:
        if hasattr(self, 'content'):
            self.hdf5_file.close()

# Read from raw data is easier than read from hdf5 file
class BraTSDataset_from_nii(Dataset):
    # https://www.synapse.org/#!Synapse:syn52939291/wiki/625694
    def __init__(self, subjects_list : list) -> None:
        super().__init__()
        self.nii_path_list = [[os.path.join(each_subject, file_name) for file_name in os.listdir(each_subject) ] for each_subject in subjects_list]
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        all_files = self.nii_path_list[idx]
        # multimodal scans: [t1c, t1n, t2f, t2w], [seg]
        [head, data_t1c] = read_nii_head_and_data_floatValue(file_path=find_first_match(substr=''.join(['t1c','.']), str_list=all_files))
        [head, data_t1n] = read_nii_head_and_data_floatValue(file_path=find_first_match(substr=''.join(['t1n','.']), str_list=all_files))
        [head, data_t2f] = read_nii_head_and_data_floatValue(file_path=find_first_match(substr=''.join(['t2f','.']), str_list=all_files))
        [head, data_t2w] = read_nii_head_and_data_floatValue(file_path=find_first_match(substr=''.join(['t2w','.']), str_list=all_files))
        [head, data_seg] = read_nii_head_and_data_floatValue(file_path=find_first_match(substr=''.join(['seg','.']), str_list=all_files))
        # Input Matrix
        data_t1c = min_max_normalize(crop_nparray(data_t1c))
        data_t1n = min_max_normalize(crop_nparray(data_t1n))
        data_t2f = min_max_normalize(crop_nparray(data_t2f))
        data_t2w = min_max_normalize(crop_nparray(data_t2w))
        # Output Target Matrix
        data_seg   = crop_nparray(data_seg)
        return torch.Tensor(np.array([data_t1c, data_t1n, data_t2f, data_t2w])), torch.Tensor(data_seg)

    def __len__(self) -> int:
        return len(self.nii_path_list)
