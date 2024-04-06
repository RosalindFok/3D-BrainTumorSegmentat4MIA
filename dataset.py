"""
load torch Dataset
"""
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, hdf5_path : str, tag : str, transform = None) -> None:
        super().__init__()
        self.hdf5_path = hdf5_path
        self.tag = tag
        self.content = None

    def open_hdf5(self):
        hdf5_file = h5py.File(self.hdf5_path, 'r')
        self.content = hdf5_file[self.tag]

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        if self.content is None:
            self.open_hdf5()
        content = self.content[str(idx)][:]
        content = torch.Tensor(content)
        return content[:4], content[-1]
    
    def __len__(self) -> int:
        if self.content is None:
            self.open_hdf5()
        return len(self.content.keys())
    
    def __del__(self) -> None:
        if self.content is None:
            self.hdf5_file.close()

