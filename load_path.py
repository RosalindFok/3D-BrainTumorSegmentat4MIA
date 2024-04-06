"""
load path
"""

import os
import yaml

YEAR = str(2018)

brain_tumor_datasets_path = os.path.join('..', 'dataset', 'brain_tumor_datasets')
data_2018_path = os.path.join(brain_tumor_datasets_path, '2018')
MICCAI_BraTS_2018_Data_Training_path = os.path.join(data_2018_path, 'MICCAI_BraTS_2018_Data_Training')
HGG_path = os.path.join(MICCAI_BraTS_2018_Data_Training_path, 'HGG')
LGG_path = os.path.join(MICCAI_BraTS_2018_Data_Training_path, 'LGG')
H_subjects_list = [os.path.join(HGG_path, x) for x in os.listdir(HGG_path)]
L_subjects_list = [os.path.join(LGG_path, x) for x in os.listdir(LGG_path)]
hdf5_path_list = [os.path.join('..', ''.join([YEAR,'_',tag,'.hdf5'])) for tag in ['HGG', 'LGG']]

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)