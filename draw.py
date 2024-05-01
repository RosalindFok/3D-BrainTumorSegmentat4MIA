"""
Plot on Windows using Matplotlib
"""

import re
import os
import numpy as np
import matplotlib.pyplot as plt

HGG_WT_output_path = os.path.join('results', 'HGG_WT.out')
assert os.path.exists(HGG_WT_output_path), f'Error: file {HGG_WT_output_path} not found.'
with open(HGG_WT_output_path, 'r') as file:
    HGG_WT_output = file.read()

lines = HGG_WT_output.split('\n')
train_losses = []
valid_losses = []
for line in lines:
    train_loss = re.search(r'Train Loss=([\d.]+)', line)
    valid_loss = re.search(r'Valid Loss=([\d.]+)', line)
    if train_loss and valid_loss:
        train_losses.append(float(train_loss.group(1)[:-1]))
        valid_losses.append(float(valid_loss.group(1)[:-1]))

assert len(train_losses) == len(valid_losses), 'Train and valid loss should have the same length.'

x = np.arange(len(train_losses))
train_y = np.array(train_losses, dtype=np.float32)
valid_y = np.array(valid_losses, dtype=np.float32)
plt.title('Train and Valid Losses', fontdict={'family':'Times New Roman','size':20})
plt.xlabel('Epoch', fontdict={'family':'Times New Roman','size':16})
plt.ylabel('Loss', fontdict={'family':'Times New Roman','size':16})
plt.plot(x, train_y, label='Train Loss')
plt.plot(x, valid_y, label='Valid Loss')
plt.legend(loc='upper right', prop={'family':'Times New Roman','size':16})
plt.savefig(os.path.join('results', 'train_valid_loss.svg'), bbox_inches='tight', pad_inches=1)
plt.close()