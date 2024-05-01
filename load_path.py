"""
load path
"""

import os
import yaml

brain_tumor_datasets_path = os.path.join('..', 'dataset', 'brain_tumor_datasets')
data_2017_path = os.path.join(brain_tumor_datasets_path, '2017')
data_2018_path = os.path.join(brain_tumor_datasets_path, '2018')
MICCAI_BraTS_2017_Data_Training_path = os.path.join(data_2017_path, 'Brats17TrainingData')
MICCAI_BraTS_2018_Data_Training_path = os.path.join(data_2018_path, 'MICCAI_BraTS_2018_Data_Training')
H_subjects_list = [
    os.path.join(HGG_path, x) 
    for HGG_path in [
        # os.path.join(MICCAI_BraTS_2017_Data_Training_path, 'HGG'), 
        os.path.join(MICCAI_BraTS_2018_Data_Training_path, 'HGG')
    ] 
    for x in os.listdir(HGG_path)
]
L_subjects_list = [
    os.path.join(LGG_path, x) 
    for LGG_path in [
        os.path.join(MICCAI_BraTS_2017_Data_Training_path, 'LGG'), 
        os.path.join(MICCAI_BraTS_2018_Data_Training_path, 'LGG')
    ] 
    for x in os.listdir(LGG_path)
]
hdf5_path_list = [os.path.join('..', ''.join([tag,'.hdf5'])) for tag in ['HGG', 'LGG']]
save_npz_path = os.path.join('..', 'saved_npz')
if not os.path.exists(save_npz_path): os.makedirs(save_npz_path)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file) 