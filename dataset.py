"""
load torch Dataset
"""
import h5py
import torch
from torch.utils.data import Dataset

# If you read hdf5 file via torch Dataset in Windows, it will raise TypeError("h5py objects cannot be pickled") and TypeError: h5py objects cannot be pickled
class BraTSDataset(Dataset):
    def __init__(self, hdf5_path : str, tag : str) -> None:
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

